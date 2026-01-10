import os
import numpy as np
import time
import multiprocessing
from multiprocessing.managers import DictProxy




from dataclasses import dataclass, field
from typing import Any, Callable

class ResourceExhaustionException(Exception):
    pass


@dataclass
class process_state:
    resource_cooldowns: dict[str, float]
    progress: int



@dataclass
class resource_cooldown():
    """
    Args:
        fixed_ms: int. Amount of time, in milliseconds, between subsequent runs of a step/dataset that involve this resource.
        cooldown_expiry_time: Don't set. float, a time.monotonic() time, in seconds, when any non-fixed cooldown on this resource expires. Should really only be set programmatically.
    """
    fixed_ms:int
    cooldown_expiry_time:float=0

@dataclass
class ProcessStateFunctions():
    set_progress_func: Callable[[int], None]
    set_resource_cooldown_func: Callable[[str, int], None]
    rate_limit_resource_names: list[str]


@dataclass
class step():
    """
    Args:
        inp: an ordered list of tuples of (directory, extension)
        function:  a callable, which in general shouldn't return anything
        kwargs: a dictionary {"argname":"argvaue"}, same for all branches
        out: an ordered list of tuples, for the outputs.
        special_kwargs: a (similarly) ordered list of string argnames. PP will fill them with the paths to the directory/XXXXX.extension's in inp and out.
        resource_penalties: keyed by resource types and valued by the penalty float. One needs not put all resource types unless you want a penalty.
        process_state_functions_kwarg: str, optional, the kwarg that will be filled with a ProcessStateFunctions dataclass to report progress, cooldowns, and resurce names.
        step_id: str, optional. Must be filled in order to use undo_steps (for routers). Steps left unnamed will be filled with str(step_index).

    """
    inp: list[tuple[str,str]]
    out: list[tuple[str,str]]
    function:Callable
    special_kwargs:list[str]
    resource_penalties:dict[str, float]
    kwargs:dict[str, Any] = field(default_factory=dict)
    process_state_functions_kwarg: str = ""
    step_id: str = ""


class Parallel():
    def __init__(self, pipeline_map: list[step | list[step] | list[step | list [step]]], resource_limits:dict[str, int], resource_timeout:int=1000, new_instance_timeout:int=20,resource_cooldowns:dict[str, resource_cooldown]={})->None:
        """
        Args:
            pipeline_map: A list of steps. You can nest them, if that is easier, with a limit of 3 layers of total depth. (No nesting is one layer).
            resource_limits: dict[str, int] A dictionary keyed by resource types and valued by the limits on the resources.
            resource_timeout: int=1000, The time in milliseconds, after a resource exhaustion, before we can attempt to try again.
            new_instance_timeout:int=20, The time in milliseconds before any attempt to start a new step. 
            resource_cooldowns:dict[str, resource_cooldown]={}, optional. Keyed by resources. Valued by resource_cooldown's. Only need to touch resource_cooldown.fixed_ms. 
        """
        self._unnest_steps(pipeline_map)
        self.resource_timeout = resource_timeout
        self.new_instance_timeout = new_instance_timeout

        self._setup_resource(resource_limits, resource_cooldowns)
        self._set_datasets()

        self._create_manager_and_lock()
        self._initialize_state_dict()
        

        
    def _unnest_steps(self, pipeline_map: list[step | list[step] | list[step | list [step]]])->None:
        """ Takes a pipeline map which either a list of steps, a list of steps and list of steps, up to a depth of three.
            Returns None, and sets self.pipeline_map to a flattened list of steps.
        """
        self.pipeline_map:list[step] = []
        for val in pipeline_map:
            if isinstance(val, step):
                self.pipeline_map.append(val)
            elif isinstance(val, list):
                for val2 in val:
                    if isinstance(val2, step):
                        self.pipeline_map.append(val2)
                    elif isinstance(val2, list):
                        for val3 in val2:
                            if isinstance(val3, step):
                                self.pipeline_map.append(val3)
                            else:
                                raise RecursionError("rmPL: PipelineMap list is too deeply nested or invalid. Limit of 3-depth.")
                    else:
                        raise ValueError("rmPL: PipelineMap list is invalid.")
            else:
                raise ValueError("rmPL: PipelineMap list is invalid.")

    
    def _init_step_ids(self)->None:
        """ Fill in empty step IDs with str(step_index). Must be called after self.pipeline_map has been set and unnested.
        """
        for index, step_i in enumerate(self.pipeline_map):
            if not step_i.step_id:
                step_i.step_id = str(index)
        return

    def _create_manager_and_lock(self)->None:
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()


    def _initialize_state_dict(self)->None:
        """
        Must be called after self._create_manager_and_lock
        This creates and sets the outer and inner dict for the ProxyDict, which is keyed by all datasets, and then sets the inner dict with the step_indices.
        """

        temp_dict = {}

        for dataset in self.datasets:
            temp_dict[dataset] = {}
            for step_index, _ in enumerate(self.pipeline_map):
                temp_dict[dataset][step_index] = process_state({}, 0)

        self.state_dict = self.manager.dict(temp_dict)
        return
    

    @staticmethod
    def _set_state_dict_progress(state_dict:DictProxy, lock, dataset:str, step_index:int, progress:int)->None:
        """ Sets the progress of a dataset of a step with the state_dict DictProxy properly.

        Args:
            state_dict: a DictProxy. Should be self.state_dict.
            lock: a Lock. Should be self.lock
            dataset: str
            step_index: int
            progress: int
        """
        
        with lock:
            this_dataset:dict[int, process_state] = state_dict[dataset]
            this_dataset[step_index].progress = progress
            state_dict[dataset] = this_dataset
        return

    
    @staticmethod
    def _set_state_dict_cooldowns(state_dict:DictProxy, lock, dataset:str, step_index:int, resource:str, cooldown_ms: int)->None:
        with lock:
            this_dataset:dict[int, process_state] = state_dict[dataset]
            this_dataset[step_index].resource_cooldowns[resource] = cooldown_ms
            state_dict[dataset] = this_dataset
        return

    
    @staticmethod
    def _get_process_state_functions(pipeline_map:list[step], state_dict:DictProxy, lock, dataset:str, step_index:int)->ProcessStateFunctions:
        def set_progress_func(progress:int)->None:
            """
            Sets the progress of the current process.

            Args:
                progress:int from 0 to 100. Progress will automatically be set to 100 upon completion, so you need not do that to mark as complete.
            """
            Parallel._set_state_dict_progress(state_dict, lock, dataset, step_index, progress)
            return
        def set_resource_cooldown_func(resource:str, cooldown_ms:int):
            """
            Reports resource cooldowns (eg, rate limits) to the main process.

            Args:
                resource:str, the same resource as defined in the resource_limits, etc. 
                cooldown_ms: int, the amount of time, in milliseconds, that we should wait starting now, before starting another step that uses this resource.
            """
            Parallel._set_state_dict_cooldowns(state_dict, lock, dataset, step_index, resource, cooldown_ms)
            return

        # now get the resource names.
        resource_names:list[str] = list(pipeline_map[step_index].resource_penalties.keys())
        resource_names:list[str] = [resource for resource in resource_names if resource.lower() != 'overall']

        return ProcessStateFunctions(set_progress_func=set_progress_func, set_resource_cooldown_func=set_resource_cooldown_func, rate_limit_resource_names=resource_names)
        
    
    def _update_resource_cooldowns(self, snapshot:dict[str, dict[int, process_state]])->None:
        """ Needs to be in a self.lock. Modifies snapshot in place to clear resource cooldowns.
        """
        reference_time:float = time.monotonic()
        
        for dataset in snapshot.keys():
            for step_index in snapshot[dataset].keys():
                dataset_step_state = snapshot[dataset][step_index]
                
                for resource in dataset_step_state.resource_cooldowns.keys():
                    this_resource_cooldown_ms:float = dataset_step_state.resource_cooldowns[resource]
                    new_resource_time :float = (this_resource_cooldown_ms/1000) + reference_time
                    if new_resource_time > self.resource_cooldowns[resource].cooldown_expiry_time:
                        self.resource_cooldowns[resource].cooldown_expiry_time = new_resource_time
                    # Reset it so we don't do this again
                    snapshot[dataset][step_index].resource_cooldowns[resource] = 0.0
        return

    
    def _handle_process_progress(self, snapshot:dict[str, dict[int, process_state]])->None:
        """
        Needs to be in a self.lock. Modifies snapshot in place. Adds p-locks to finished processes, and reduces the utilization.
        """
        for dataset in snapshot.keys():
            for step_index in snapshot[dataset].keys():
                dataset_step_state:process_state = snapshot[dataset][step_index]


                # Deal with progress
                if dataset_step_state.progress == 100:
                    # Create a p-lock and set to 110.
                    self._create_p_file(self.pipeline_map, step_index, dataset, 'p-lock')
                    dataset_step_state.progress = 110

                    # remove this utilization
                    for resource, penalty in self.pipeline_map[step_index].resource_penalties.items():
                        self.resource_utilization[resource] -= penalty


    def _snapshot_to_state_dict(self, snapshot:dict[str, dict[int, process_state]])->None:
        """
        Needs to be in a self.lock, just updates the self.state_dict from a copy of it.
        """
        for dataset in snapshot.keys():
            self.state_dict[dataset] = snapshot[dataset]
        return
    


    def _propagate_state_dict(self)->None:
        """ Checks processes' cooldown reports and propagates them to self.resource_cooldowns.
            Checks progress on processes, creating p-locks when the task is at 100. """
        with self.lock:
            
            snapshot:dict[str, dict[int, process_state]] = dict(self.state_dict)
            self._update_resource_cooldowns(snapshot)
            self._handle_process_progress(snapshot)
            self._snapshot_to_state_dict(snapshot)
        return


    def log(self, message:str)->None:
        print(message)


    def _set_datasets(self)->None:
        """
        Goes into the inp of the first step in the pipeline
        and then extracts all legal datasets (those starting without
        dots or underscores), and sets these as a list. We will sort
        them. We will take from the first entry in the list.
        """

        directory_to_check:str = self.pipeline_map[0].inp[0][0]
        extension_to_check:str = '.' + self.pipeline_map[0].inp[0][1]
        files_in_dir:list[str] = os.listdir(directory_to_check)

        # Let's get only that which is of the correct extension
        files_in_dir: list[str] = [file for file in files_in_dir if os.path.splitext(file)[1] == extension_to_check]

        # Let's get only the basenames without extensions now
        files_in_dir: list[str] = [os.path.splitext(os.path.basename(file))[0] for file in files_in_dir]

        # And let's get only that which does not start with a dot or an underscore
        files_in_dir: list[str] = [file for file in files_in_dir if file[0] not in ['.', '_']]

        files_in_dir.sort()

        self.datasets: list[str] = files_in_dir
        

    def _setup_resource(self, resource_limits:dict[str, int], resource_cooldowns:dict[str, resource_cooldown])->None:
        """
        Initializes resource utilization and resource last_utilized. Also fills empty
        cooldowns with zeros. 
        """
        self.resource_limits:dict = resource_limits
        
        self.resource_utilization:dict = {}
        self.resources_last_utilized:dict = {}
        for key in resource_limits.keys():
            self.resource_utilization[key] = 0.0
            self.resources_last_utilized[key] = -np.inf

            if key not in resource_cooldowns.keys():
                resource_cooldowns[key] = resource_cooldown(0)

        self.resource_cooldowns = resource_cooldowns


    def _get_pipeline_set(self, directory:str)->set[str]:
        if not os.path.exists(os.path.join(directory, '.pipeline')):
            os.makedirs(os.path.join(directory, '.pipeline'), exist_ok=True)
        return set(os.listdir(os.path.join(directory, '.pipeline')))


    def _get_dataset_filename(self, dir_ext_tuple:tuple[str,str], dataset:str)->str:
        directory = dir_ext_tuple[0]
        extension = dir_ext_tuple[1]
        return os.path.join(directory, f'{dataset}.{extension}')


    def _has_p_lock(self, dir_ext_tuple:tuple[str,str], dataset:str)->tuple[bool,bool]:
        """Returns a tuple of bools, the first is whether the p_lock exists, the
        second is whether the file itself exists. """

        dataset_filename = self._get_dataset_filename(dir_ext_tuple, dataset)
        has_p_lock:bool = (os.path.basename(dataset_filename) + ".p-lock") in self._get_pipeline_set(directory=dir_ext_tuple[0])
        exists:bool = os.path.exists(dataset_filename)
        return has_p_lock, exists


    def _has_p_log(self, dir_ext_tuple:tuple[str,str], dataset:str)->bool:
        dataset_basename = os.path.basename(self._get_dataset_filename(dir_ext_tuple, dataset))
        has_p_log: bool = (dataset_basename + ".p-log") in self._get_pipeline_set(directory=dir_ext_tuple[0])
        return has_p_log

    def _needs_p_lock(self, dir_ext_tuple:tuple[str, str]):
        """ Returns whether or not a dir_ext_tuple needs a p-lock by checking if this dir_ext_tuple is the result of a step.
            It only checks the directory.
        """

        for step_i in self.pipeline_map:
            for det in step_i.out:
                if det[0] == dir_ext_tuple[0]:
                    return True
        return False


    def _is_legal_dataset_step(self, step_index:int, dataset:str)->bool:
        """
        Returns whether it is legal to work on *this* particular *dataset* for this step.
        It assumes the step itself is legal, which it may not be. That should be checked,
        separately, perhaps through _get_legal_steps()
        """
        # First thing to do is to check whether the prerequisites are finished.
        for dir_ext_tuple in self.pipeline_map[step_index].inp:
            # Note that the first step (the external input) prerequisites need not a p_lock.
            has_p_lock:tuple[bool, bool] = self._has_p_lock(dir_ext_tuple, dataset)
            needs_p_lock: bool = self._needs_p_lock(dir_ext_tuple)
            if not (False not in has_p_lock or (has_p_lock[1]==True and not needs_p_lock)):
                return False
        
        # Next, just ensure that nobody has already started working on this one.
        for dir_ext_tuple in self.pipeline_map[step_index].out:
            if self._has_p_log(dir_ext_tuple, dataset) or self._has_p_lock(dir_ext_tuple, dataset)[0]:
                return False
        return True
   

    def _get_legal_steps(self)->list[int]:
        """
        Checks for resource exhaustion. Returns a sorted list of step indices. Returns an empty list if none.
        Checks the resource_cooldown.
        """
        legal_step_i:list[int] = []
        for index, the_step in enumerate(self.pipeline_map):
            i_legal = True
            for key in the_step.resource_penalties.keys():
                # Check current resource usage for exhaustion
                if the_step.resource_penalties[key] + self.resource_utilization[key] > self.resource_limits[key]:
                    i_legal = False

                # Check both fixed and variable resource cooldowns.
                current_time:float = time.monotonic()
                last_utilized_time:float = self.resources_last_utilized[key]
                if not (last_utilized_time + (self.resource_cooldowns[key].fixed_ms/1000) < current_time):
                    i_legal = False # the fixed cooldown has not passed
                if not (self.resource_cooldowns[key].cooldown_expiry_time < current_time):
                    i_legal = False # the varying cooldown has not passed

            if i_legal:
                legal_step_i.append(index)
        return legal_step_i


    def _find_legal_task(self)->tuple[int, str]|None:
        """
        Returns the index of the step and the string of the dataset of a legal task to complete.
        If no legal tasks, returns None. If resource exhaustion, through a ResourceExhaustionException.
        """
        
        #First let's see which steps are legal for us to work on.
        legal_steps:list[int] = self._get_legal_steps()

        if not legal_steps:
            raise ResourceExhaustionException()

        # Now let's just go through those and find the first dataset that is legal to work on.

        for step_i in legal_steps:
            for dataset in self.datasets:
                if self._is_legal_dataset_step(step_i, dataset):
                    return (step_i, dataset)
        return None


    def _is_the_final_step_done(self)->bool:
        """Returns whether or not all datasets are complete
        on the final step
        """
        dir_ext_tuples: list[tuple[str,str]] = self.pipeline_map[-1].out

        for dir_ext_tuple in dir_ext_tuples:
            for dataset in self.datasets:
                has_p_lock:tuple[bool, bool] = self._has_p_lock(dir_ext_tuple, dataset) 
                if False in has_p_lock:
                    return False
        return True


    @staticmethod
    def _create_p_file(pipeline_map:list[step], step_index, dataset, p_ext:str, delete:bool=False)->None:
        """ p_ext is either 'p-lock' or 'p-log'. if delete it deletes the p-file.
        """

        the_step: step = pipeline_map[step_index]
        dir_ext_tuples: list[tuple[str, str]] = the_step.out

        for dir_ext_tuple in dir_ext_tuples:
            log_filename:str = os.path.join(dir_ext_tuple[0], '.pipeline/'+ dataset + '.' + dir_ext_tuple[1] + '.' +  p_ext)
            if not delete:
                with open(log_filename, 'a') as file:
                    file.write('')
            else:
                if os.path.exists(log_filename):
                    os.remove(log_filename)
                else:
                    print(f"rmPL: Can't remove nonexistent log file {log_filename}. Skipping.")



    def _p_launch_func(self, func: Callable)->Callable:
        """ Returns a Callable to a wrapper function which runs func with kwargs and then updates progress when done.
        """
        def wrapped_func(step_index:int, dataset:str, **kwargs)->None: 
            func(**kwargs)
            self._set_state_dict_progress(self.state_dict, self.lock, dataset, step_index, 100)
            

        return wrapped_func


    def _update_resource_utilization(self, step_index:int, starting_task:bool)->None:
        """ Updates the resource utilization and last utilization to reflect starting a run of step step_index.
        Args:
            starting_task : bool, set to true to increase utilization, etc, or false to signify the completion of a task.
        """
        
        the_step: step = self.pipeline_map[step_index]

        # Get resource_penalty: 
        for resource in the_step.resource_penalties.keys():
            
            utilization_change = the_step.resource_penalties[resource]

            if starting_task:
                self.resource_utilization[resource] += utilization_change
                self.resources_last_utilized[resource] = time.monotonic()
            if not starting_task:
                self.resource_utilization[resource] -= utilization_change
        return

    
    def _build_kwargs(self, step_index:int, dataset:str)->dict[str, Any]:
        """ Returns the kwargs (both special and basic) for a dataset and step.
            Adds the ProcessStateFunctions if process_state_functions_kwarg is set.
        """

        the_step:step = self.pipeline_map[step_index]
        
        # Let's build the full kwargs.
        full_kwargs: dict = the_step.kwargs.copy()
        
        for i, inp_dir_ext_tuple in enumerate(the_step.inp):
            kwarg = the_step.special_kwargs[i]
            full_kwargs[kwarg] = os.path.join(inp_dir_ext_tuple[0], dataset + '.' + inp_dir_ext_tuple[1])
        for j, out_dir_ext_tuple in enumerate(the_step.out):
            i = j+len(the_step.inp)
            kwarg = the_step.special_kwargs[i]
            full_kwargs[kwarg] = os.path.join(out_dir_ext_tuple[0], dataset + '.' + out_dir_ext_tuple[1])

        if the_step.process_state_functions_kwarg:
            full_kwargs[the_step.process_state_functions_kwarg] = self._get_process_state_functions(self.pipeline_map, self.state_dict, self.lock, dataset, step_index)

        return full_kwargs


    def _start_multiprocessing_step_dataset(self, step_index:int, dataset:str)->None:
        """ Starts the multiprocessing process for any step and dataset. Does not check anything. Adds a .p-log. Also adds to the resource utilization.
        """
        
        the_step:step = self.pipeline_map[step_index]
        
        full_kwargs:dict[str, Any] = self._build_kwargs(step_index, dataset)
        
        # Let's wrap the function to add the lock when done:
        target_func:Callable = self._p_launch_func(the_step.function)
        p = multiprocessing.Process(target=target_func, args=(step_index, dataset), kwargs=full_kwargs)

        self._create_p_file(self.pipeline_map, step_index, dataset, 'p-log')
        p.start()

        self._update_resource_utilization(step_index, starting_task=True)
        
        self.log(f"rmPP: Started {the_step.function.__name__} for dataset {dataset}.")


        return


    def go(self)->None:
        """ Repeatedly and constantly attempts to run tasks until things are done. It will make many parallel instances
        of itself, until resource exhaustion. For this reason it is recommended to have an 'overall' resource limit. 
        If resources are exhausted it will wait self.resource_timeout:int milliseconds before trying again. It will also wait
        self.new_instance_timeout:int milliseconds generally after attempting to start again.
        """

        stop_reason:str = ""
        
        while not stop_reason:

            self._propagate_state_dict()

            try:
                task = self._find_legal_task()
            except ResourceExhaustionException:
                time.sleep(self.resource_timeout / 1000)
                continue

            if not task:
                # Are we done?
                if self._is_the_final_step_done():
                    stop_reason = "rmPP: All datasets have been completed for the final step."
                continue

            self._start_multiprocessing_step_dataset(task[0], task[1])
            time.sleep(self.new_instance_timeout / 1000)

        print(f"rmPP: stop_reason: {stop_reason}")


    @staticmethod
    def undo_steps(dataset:str, step_ids: list[str], pipeline_map: list[step])->None:
        """
        For a given dataset, this function will delete the p-logs and p-locks of the steps in the list. 
        """

        steps_to_delete: list[step|None] = [step_i if step_i.step_id in step_ids else None for step_i in pipeline_map]

        for step_index, _ in enumerate(steps_to_delete):
            Parallel._create_p_file(pipeline_map, step_index, dataset, 'p-lock', delete=True)
            Parallel._create_p_file(pipeline_map, step_index, dataset, 'p-log', delete=True)
        return
