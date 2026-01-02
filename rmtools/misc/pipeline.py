import os
import time
import multiprocessing
from multiprocessing.managers import DictProxy




"""
Here we will consider and explore the main pipeline.

Let's first consider the french pipeline:

1.) input.mp3 is post processed in pp.py
2.) BREAKING pp.py into split/
3.) We break with splitsisters to splitsisters/
4.) We ask gemini with AI_TS.py (batch_runner?) to reply
5.) We then filter. 


We probably would like something that can manage this for us. Maybe a 'parallel task manager'.
So we first need to setup the tasks to be parallelized, and then start parallelizing.

I like the 'lock' approach. Perhaps we can create a .pipeline subdirectory, and make file.extension.lock, when
that parallel task has started. Then maybe we can fill it with "FINISHED" or something, when it is done. 
Alternatively, we can create a .log when we start, log in there, and then .lock when done. Maybe p-lock and p-log. 
That could be good. I think we'll do that. 

We generally do things very file based. I think that is the way to go, since generally I think this parallel thing
is useful for when things take a long time and probably want to be stored in an intermediate storage.

I think we can also similarly force every output to be in a new subdirectory, as this also helps simplify and we probably can think the separation is a good thing. If storage is a concern, we can note that g-locks, etc. is all that is really needed.

One thing though is that this subtitles thing only needs sort of one 'thing' per step. But I suppose that is part of 'parallel' tasks, since otherwise it would be like an interwoven thing. I'm actually okay with that. 

So, anyway, we could probably make each step a dictionary entry. 

We have like a:

step:dict = {function:my_func, kwargs:kwargs, }


So for example we could have a:

ai_step = {function: ai_ts.start, kwargs:{ }}

Hmm. I'm not immediately sure about how we could actually fit the 0001.txt for example inside. Well Hmm

step:dict keys:

inp:list[str] | str
function
kwargs:dict
special_kwarg:list[str] | str
out:list[str] | str

So for ai_step we could have:

inp: ["spliced/", "spliced_sisters"]
function: ai_ts.go
kwargs: {"model":"gemini-pro"}
special_kwargs: ["spliced_loc", "spliced_sisters_loc"]
out: ["reply/"]

This is almost perfect except we don't have extensions. For example spliced should be .mp3, and reply.txt. 
One way we could do this is by explicitly specifying. That might be good, actually, and is what we will do, through inp:[list[tuple[str, str]]]

This is good and I think we will typedef this. 

So inp: [("spliced/", ".mp3"), ("spliced_sisters", ".mp3")]

Okay we'll named tuple this.

I suppose some steps should have different parallel branch counts. We'll put that in as well.


Now the question is how do we get started. I think all files should just be labeled by their number, no prefixes. 
I suppose we just take the first directory, and just go from there, doing for each file that does not start
with an underscore or a dot. We don't actually need 0006, etc. Not buggs. So I suppose the first thing we should do is just take
an inventory of all the branches we have.

Now that we know which datasets we have, we best get to work. One thing we need to do is know which are elegible. Which I suppose
imposes eligibility requirements. The first thing we should do is make the .p-lock for finished tasks, since those should obviously be
ignored. We can do these searches as set lookups, since there are no duplicates.

The question of "which to do" is not that simple. And sometimes we will want different behaviour, so we'd have to make it somewhat
configurable. One method that is useful is the straight linear vertical. Basically, try to finish datasets as fast as possible. Another method
is the horizontal; finishing steps as fast as possible. And we'd probably want some 'adaptive' stuff, especially when dealing with
rate-limited stuff. In either case, we'd like to know when things are legal to do or not. So we'd ask: is it legal to do step Y of
dataset XXXX? We will check both the prerequisites, and that it has not been started/completed. For now, let's start with the vertical.


Our hacky solution of adding the p-lock when done seems to be buggs, since the fact is that we also need to check to see when they're done to update resource utilization.
So what can we do? And we also have the question of alerting. We've got the resource_cooldowns dict, but we need a way to access it. So utlimately, we need a way to:
1.) Signal during the execution of a task
2.) Signal when done.


It seems that ultimately, this is the same task. 


I think we are looking at this wrong; We do not need to constantly poll. We only need to poll, when we're going to start a new process. 
So in the start new process thing, the first thing we can do is call some sort of _check_on_processes, which can check for:

    a.) Process being done -> update_resources, add a p-lock.
    b.) There is a manager ProxyDict that is key


    We will create a manager dict[str, dict[int, process_state]], where the outer key dataset, the inner key is the step_index,
    and then the progress are ints 0-100 for progress. Right now we'll probably only really use 100 to signify done.
    This can just be checked par hazard. There are some concurrency issues, possibly, but I think that so long as each worker touches their
    own dict, and noone else's dict, everything is okay. 


"""



from dataclasses import dataclass
from typing import Any, Callable, Dict, NamedTuple

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
        cooldown_expiry_time: float, a time.monotonic() time, in seconds, when any non-fixed cooldown on this resource expires. Should really only be set programmatically.
    """
    fixed_ms:int
    cooldown_expiry_time:float

@dataclass
class ProcessStateFunctions():
    set_progress_func: Callable[[int], None]
    set_resource_cooldown_func: Callable[[str, int], None]


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
        process_state_functions_kwarg: str, optional, the kwarg that will be filled with a ProcessStateFunctions dataclass to report progress and cooldowns.

    """
    inp: list[tuple[str,str]]
    out: list[tuple[str,str]]
    function:Callable
    special_kwargs:list[str]
    resource_penalties:dict[str, float]
    kwargs:dict={}
    process_state_functions_kwarg: str = ""


class Parallel():
    def __init__(self, pipeline_map:list[step], resource_limits:dict, resource_timeout:int=1000, new_instance_timeout:int=20,resource_cooldowns:dict[str, resource_cooldown]={})
        #TODO: FIX RESOURCE COOLDOWN DEFUALT THIGN
        self.pipeline_map: list[step] = pipeline_map
        self.resource_timeout = resource_timeout
        self.new_instance_timeout = new_instance_timeout
        self._setup_resource(resource_limits)
        self._set_datasets()

        # We also need a last_utilized time. 
        self.resources_last_utilized: dict[str, float] = {}
        self.resource_cooldowns: dict[str, resource_cooldown] = resource_cooldowns

        self._create_manager_and_lock()
        
    def _create_manager_and_lock(self)->None:
        self.manager = multiprocessing.Manager()
        self.lock = self.manager.Lock()


    def _initialize_state_dict(self)->None:
        """
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
    def _get_process_state_functions(state_dict:DictProxy, lock, dataset:str, step_index:int)->ProcessStateFunctions:
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

        return ProcessStateFunctions(set_progress_func=set_progress_func, set_resource_cooldown_func=set_resource_cooldown_func)
        
    
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
        Needs to be in a self.lock. Modifies snapshot in place. Adds p-locks to finished processes.
        """
        for dataset in snapshot.keys():
            for step_index in snapshot[dataset].keys():
                dataset_step_state:process_state = snapshot[dataset][step_index]


                # Deal with progress
                if dataset_step_state.progress == 100:
                    # Create a p-lock and set to 110.
                    self._create_p_file(self.pipeline_map, step_index, dataset, 'p-lock')
                    dataset_step_state.progress = 110


    def _snapshot_to_state_dict(self, snapshot:dict[str, dict[int, process_state]])->None:
        """
        Needs to be in a self.lock, just updates the self.state_dict from a copy of it.
        """
        for dataset in snapshot.keys():
            self.state_dict[dataset] = snapshot[dataset]
        return
    


    def _process_manager_dict(self)->None:
        """ Thread safe, creates p-locks when things are done, and adjusts resource allocations. Also updates the resource_cooldowns.
        """

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
        

    def _setup_resource(self, resource_limits:dict)->None:
        self.resource_limits:dict = resource_limits
        
        self.resource_utilization:dict = {}
        for key in resource_limits.keys():
            self.resource_utilization[key] = 0.0


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
            if not (False not in has_p_lock or (has_p_lock[1]==True and step_index==0)):
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
    def _create_p_file(pipeline_map:list[step], step_index, dataset, p_ext:str)->None:
        """ p_ext is either 'p-lock' or 'p-log'
        """

        the_step: step = pipeline_map[step_index]
        dir_ext_tuples: list[tuple[str, str]] = the_step.out

        for dir_ext_tuple in dir_ext_tuples:
            log_filename:str = os.path.join(dir_ext_tuple[0], '.pipeline/'+ dataset + '.' + dir_ext_tuple[1] + '.' +  p_ext)
            with open(log_filename, 'a') as file:
                file.write('')


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
            full_kwargs[the_step.process_state_functions_kwarg] = self._get_process_state_functions(self.state_dict, self.lock, dataset, step_index)

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
