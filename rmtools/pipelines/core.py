from dataclasses import dataclass
import sys
import shutil
import os
import numpy as np
import time
import multiprocessing
from multiprocessing.managers import DictProxy
from typing import Any, Callable
from . import Blocks
from . import file_io
from . import state_management

from .rmPL_Types import Block, ResourceExhaustionException, RouterType, process_state, resource_cooldown, ProcessStateFunctions, Step, OnReturnInfoStruct, ParallelOptions, OptionalBlock

def _zipsort(array_to_sort_by:list, *args, reverse:bool=False)-> tuple:
    """
    Takes a list-like array to sort by, and then as many other list-likes
    after, they all have to be of the same length. Sorts all lists based on
    the first list-like, and returns a tuple of all the sorted lists.
    """

    for listlike in args:
        if len(listlike) != len(array_to_sort_by):
            raise ValueError("rmtools.zipsort: arrays of unequal length.")

    combined = sorted(zip(array_to_sort_by, *args), reverse=reverse)
    return tuple(zip(*combined))



class Parallel():
    def __init__(self, parallel_options:ParallelOptions)->None:
        """
        Args:
            parallel_options: ParallelOptions
        """
        self.pipeline_map: list[Step] = Blocks.process_and_flatten_input_pipeline_map(parallel_options.pipeline_map)
        
        self.resource_timeout = parallel_options.resource_timeout_ms
        self.new_instance_timeout = parallel_options.new_instance_timeout_ms

        self._setup_resource(resource_limits=parallel_options.resource_limits, resource_cooldowns=parallel_options.resource_cooldowns)
        self.restrict_datasets:tuple[str, str]|None = parallel_options.restrict_datasets
        self._set_datasets()
        self._clear_existing(parallel_options.clear_existing)

        self._create_manager_and_lock()
        self._initialize_state_dict()
        self._log_cursors = {} # maps (step_index, dataset) -> file byte offset

        self.use_vertical:bool = parallel_options.use_vertical
        self.clear_orphan_p_log:bool = parallel_options.clear_orphan_p_log
        self.redundant_process_limit:float = parallel_options.redundant_process_limit

        self._running_processes:list[tuple[int,str]] = []

    
    
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
        """ Needs to be in a self.lock. Modifies snapshot in place. Adds p-locks to finished processes, and reduces the utilization.
        Processes with -100 progress are 'reset'. In both cases, remove it from self._running_processes.
        """
        for dataset in snapshot.keys():
            for step_index in snapshot[dataset].keys():
                dataset_step_state:process_state = snapshot[dataset][step_index]


                # Deal with progress
                if dataset_step_state.progress == 100:
                    # Create a p-lock and set to 110.
                    file_io.create_p_file(self.pipeline_map, step_index, dataset, 'p-lock')
                    dataset_step_state.progress = 110

                    # remove this utilization
                    for resource, penalty in self.pipeline_map[step_index].resource_penalties.items():
                        self.resource_utilization[resource] -= penalty

                    self._running_processes = [process_tuple for process_tuple in self._running_processes if process_tuple != (step_index, dataset)]

                elif dataset_step_state.progress == -100:
                    # We delete the p-log.
                    file_io.create_p_file(self.pipeline_map, step_index, dataset, 'p-log', delete=True)
                    # Reset progress to zero.
                    dataset_step_state.progress = 0

                    # remove this utilization
                    for resource, penalty in self.pipeline_map[step_index].resource_penalties.items():
                        self.resource_utilization[resource] -= penalty

                    self._running_processes = [process_tuple for process_tuple in self._running_processes if process_tuple != (step_index, dataset)]


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


    def _clear_existing(self, clear_existing:bool=False)->None:
        if clear_existing:
            if not self.datasets:
                self.log("rmPL: No datasets. Can't clear. Logic error? Or no datasets set?")
                return
            for dataset in self.datasets:
                for step_index, step in enumerate(self.pipeline_map):
                    for dir_ext_tuple in step.out:
                        file_io.create_p_file(self.pipeline_map, step_index, dataset, 'p-lock', delete=True)
                        file_io.create_p_file(self.pipeline_map, step_index, dataset, 'p-log', delete=True)
                        filename = file_io.get_dataset_filename(dir_ext_tuple, dataset)
                        if os.path.exists(filename):
                            os.unlink(filename)
                            self.log(f"rmPL: cleared existing file {filename} due to clear_existing=True.")
        return


    def _stream_logs_to_console(self):
        for step_index, dataset in self._running_processes:
            # Match the path logic used in _p_launch_func
            log_path = os.path.join(self.pipeline_map[step_index].out[0][0], ".pipeline", "logs", dataset, "stdout_err.txt")
            
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    # Seek to where we last stopped reading
                    last_pos = self._log_cursors.get((step_index, dataset), 0)
                    f.seek(last_pos)
                    
                    new_data = f.read()
                    if new_data:
                        # Print with a prefix so you know which process it came from
                        prefix = f"[{self.pipeline_map[step_index].step_id} | {dataset}]: "
                        print(prefix + new_data.replace("\n", f"\n{prefix}").strip(prefix))
                        
                    self._log_cursors[(step_index, dataset)] = f.tell()
            else:
                print("Warning: log for stdout_err not found")


    def _set_datasets(self)->None:
        """
        Goes into the inp of the first Step in the pipeline
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
        
        if not files_in_dir:
            self.log("rmPL: No datasets. Check your prerequisites files.")

        if self.restrict_datasets:
            files_in_dir = [file for file in files_in_dir if (file >= self.restrict_datasets[0] and file <= self.restrict_datasets[1])]
            if not files_in_dir:
                self.log("rmPL: No datasets after restricting dataset. Did you mistype it?")

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



    def _has_p_lock(self, dir_ext_tuple:tuple[str,str], dataset:str)->tuple[bool,bool]:
        """Returns a tuple of bools, the first is whether the p_lock exists, the
        second is whether the file itself exists. """

        dataset_filename = file_io.get_dataset_filename(dir_ext_tuple, dataset)
        has_p_lock:bool = (os.path.basename(dataset_filename) + ".p-lock") in self._get_pipeline_set(directory=dir_ext_tuple[0])
        exists:bool = os.path.exists(dataset_filename)
        return has_p_lock, exists

    def _has_p_log(self, dir_ext_tuple:tuple[str,str], dataset:str)->bool:
        dataset_basename = os.path.basename(file_io.get_dataset_filename(dir_ext_tuple, dataset))
        has_p_log: bool = (dataset_basename + ".p-log") in self._get_pipeline_set(directory=dir_ext_tuple[0])
        return has_p_log
    
    
    def _has_p_file(self, dir_ext_tuple:tuple[str,str], dataset:str, p_file:str)->bool:
        """p_file is either "p-lock" or "p_log"."""
        dataset_basename = os.path.basename(file_io.get_dataset_filename(dir_ext_tuple, dataset))
        has_p_file: bool = (dataset_basename + "." + p_file) in self._get_pipeline_set(directory=dir_ext_tuple[0])
        return has_p_file

    
    def _has_its_p_file(self, step_index:int, dataset:str, p_file:str, true_on_any:bool=False)->bool:
        """p_file is either "p-lock" or "p-log". true_on_any means if any p-file is present, returns true.
            if not true_on_any, then to return true it must have *all* of its p-files.
        """
        the_step = self.pipeline_map[step_index]
        
        out=True

        for det in the_step.out:
            if not self._has_p_file(det, dataset, p_file):
                out=False
            else:
                if true_on_any:
                    return True
        return out
        

    def _needs_p_lock(self, dir_ext_tuple:tuple[str, str]):
        """ Returns whether or not a dir_ext_tuple needs a p-lock by checking if this dir_ext_tuple is the result of a Step.
            It only checks the directory.
        """

        for step_i in self.pipeline_map:
            for det in step_i.out:
                if det[0] == dir_ext_tuple[0]:
                    return True
        return False


    def _is_legal_dataset_step(self, step_index:int, dataset:str, ensure_no_duplicates:bool=True)->bool:
        """ Returns whether it is legal to work on *this* particular *dataset* for this step.
        It assumes the step itself is legal, which it may not be. That should be checked,
        separately, perhaps through _get_legal_steps(). Checks for p-logs if ensure_no_duplicates.
        """
        # First thing to do is to check whether the prerequisites are finished.
        for dir_ext_tuple in self.pipeline_map[step_index].inp:
            # Note that the first step (the external input) prerequisites need not a p_lock.
            has_p_lock:tuple[bool, bool] = self._has_p_lock(dir_ext_tuple, dataset)
            needs_p_lock: bool = self._needs_p_lock(dir_ext_tuple)
            if not (False not in has_p_lock or (has_p_lock[1]==True and not needs_p_lock)):
                return False
        
        # Next, just ensure that nobody has already started working on this one.
        if ensure_no_duplicates:
            if (step_index, dataset) in self._running_processes:
                return False
            for dir_ext_tuple in self.pipeline_map[step_index].out:
                if self._has_p_log(dir_ext_tuple, dataset) or self._has_p_lock(dir_ext_tuple, dataset)[0]:
                    return False
        return True
   

    def _get_legal_steps(self)->list[int]:
        """
        Checks for resource exhaustion. Returns a sorted list of Step indices. Returns an empty list if none.
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

    def _list_all_legal_tasks(self)->list[tuple[int, str]]|None:
        """ Iterates through every step and dataset, listing out all tasks legal to complete. Returns None if None.
            Raises ResourceExhaustionException if resources are exhausted. """
        legal_steps:list[int] = self._get_legal_steps()
        
        if not legal_steps:
            raise ResourceExhaustionException()
        
        # Now let's iterate through and see which datasets are also legal.
        out:list[tuple[int,str]] = []
        for step_i in legal_steps:
            for dataset in self.datasets:
                if self._is_legal_dataset_step(step_i, dataset):
                    out.append((step_i, dataset))
        if not out:
            return None
        else:
            return out


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

    def _find_legal_task_vertical(self)->tuple[int, str]|None:
        """ Returns index of Step and string of the dataset of a legal task to complete.
        Returns None if none. Can Raise ResourceExhaustionException. Attempts to follow a 'vertical' path."""
        
        # Let's get all legal tasks.
        legal_tasks:list[tuple[int,str]]|None = self._list_all_legal_tasks()
        if not legal_tasks:
            return None

        # Now we will want to follow the 'vertical' approach. This is simply done by picking the highest legal step_i we can take.
        # legal_tasks is increasing in step, then dataset. Thus this will give us the 'first' dataset at the highest step:
        first_task:tuple[int,str] = max(legal_tasks,key=lambda x:x[0])
        return first_task


    def _are_all_steps_done(self)->bool:
        """Returns whether or not all datasets are complete. 
        """
        for step_i in self.pipeline_map:
            dir_ext_tuples: list[tuple[str,str]] = step_i.out

            for dir_ext_tuple in dir_ext_tuples:
                for dataset in self.datasets:
                    has_p_lock:tuple[bool, bool] = self._has_p_lock(dir_ext_tuple, dataset) 
                    if False in has_p_lock:
                        return False
        return True


    def _get_undone_tasks(self)->list[tuple[int,str]]:
        """ Checks the output p-locks of each step of each task to
            see whether or not they're done. Returns list of 
            (step_index, dataset) of tasks that are not done.
        """

        out:list[tuple[int,str]] = []

        for step_index, step in enumerate(self.pipeline_map):
            for dataset in self.datasets:
                step_dataset_done:bool = False
                for det in step.out:
                    if self._has_p_lock(det, dataset)[0]:
                        step_dataset_done = True
                        break
                if not step_dataset_done:
                    out.append((step_index,dataset))
        return out

    
    def _count_task_instances(self, task:tuple[int,str])->int:
        instances_of_task:list[tuple[int,str]] = [task_i for task_i in self._running_processes if task_i == task]
        instance_count = len(instances_of_task)
        return instance_count
                    
    if False:
        def _clean_and_get_valid_straggler_task(self)->tuple[int,str]|None:
            """ 
            Returns a task tuple (step_index:int, dataset:str) of a straggler task that is eligible to be redundantly run.
            If self.clear_orphan_p_log is set, then it clears p-logs of tasks not currently running.
            """

            """
            An issue we have here is that this just spams repeats of the current because this returns fast. 
            I suppose we should make it such that it looks at the undone task with the least amount of instances.
            """
            undone_tasks:list[tuple[int,str]] = self._get_undone_tasks()
            undone_task_instance_counts:list[int] = [self._count_task_instances(task) for task in undone_tasks]

            if not undone_tasks:
                self.log("rmPL: All tasks are complete according to g-locks. If you want to reset, you should also clear the respective g-locks.")
                return None

            _, undone_tasks = _zipsort(undone_task_instance_counts, undone_tasks, reverse=True)
            


            for task in undone_tasks:

                # clear the p_logs if we're not working on it and clear_orphan_p_log is set to True.
                if self.clear_orphan_p_log:
                    if (self._has_its_p_file(task[0], task[1], 'p-log', true_on_any=True) 
                        and task not in self._running_processes):
                        file_io.create_p_file(self.pipeline_map, task[0], task[1], 'p-log', delete=True)
                
                # We will launch this straggler, *again*, even if it is running, provided that we are within limits.
                instance_count = self._count_task_instances(task)
                valid_to_retry:bool = True

                # check if we have resources, generally, to do this:
                if task[0] not in self._get_legal_steps():
                    valid_to_retry = False
                    continue

                # check if this is legal to do, based on prerequisites
                if not self._is_legal_dataset_step(task[0], task[1], ensure_no_duplicates=False):
                    valid_to_retry = False

                # check if, by starting another instance, we'd be over the self._redundant_process_limit fraction of resources.
                for resource, penalty in self.pipeline_map[task[0]].resource_penalties.items():
                    if ((penalty*(instance_count + 1))/self.resource_limits[resource]) > self.redundant_process_limit:
                        valid_to_retry = False
                        break
        
                if valid_to_retry:
                    self.log(f"rmPL: Redundantly running straggler task: {self.pipeline_map[task[0]].step_id}, dataset {task[1]}")
                    return task

            return None


    if False:
        def _clean_and_get_valid_straggler_task(self)->tuple[int,str]|None:
            """ 
            Returns a task tuple (step_index:int, dataset:str) of a straggler task that is eligible to be redundantly run.
            If self.clear_orphan_p_log is set, then it clears p-logs of tasks not currently running.
            """

            # If redundancy is disabled, do not launch duplicate work.
            if self.redundant_process_limit <= 0:
                return None

            undone_tasks: list[tuple[int, str]] = self._get_undone_tasks()
            undone_task_instance_counts: list[int] = [self._count_task_instances(task) for task in undone_tasks]

            if not undone_tasks:
                self.log("rmPL: All tasks are complete according to g-locks. If you want to reset, you should also clear the respective g-locks.")
                return None

            _, undone_tasks = _zipsort(undone_task_instance_counts, undone_tasks, reverse=True)

            for task in undone_tasks:
                step_index, dataset = task

                # Optionally clear orphan p-logs for tasks that are not currently running.
                if self.clear_orphan_p_log:
                    if self._has_its_p_file(step_index, dataset, 'p-log', true_on_any=True) and task not in self._running_processes:
                        file_io.create_p_file(self.pipeline_map, step_index, dataset, 'p-log', delete=True)

                instance_count = self._count_task_instances(task)

                # Only consider tasks already in flight as candidates for redundant relaunch.
                if instance_count == 0:
                    continue

                # Step must still be legal from a resource perspective.
                if step_index not in self._get_legal_steps():
                    continue

                # Prereqs must still be satisfied; allow duplicate instances.
                if not self._is_legal_dataset_step(step_index, dataset, ensure_no_duplicates=False):
                    continue

                step = self.pipeline_map[step_index]

                # If no penalties are defined, we cannot safely budget redundant work.
                if not step.resource_penalties:
                    continue

                valid_to_retry = True
                for resource, penalty in step.resource_penalties.items():
                    limit = self.resource_limits[resource]

                    # Guard against invalid limits.
                    if limit <= 0:
                        valid_to_retry = False
                        break

                    if ((penalty * (instance_count + 1)) / limit) > self.redundant_process_limit:
                        valid_to_retry = False
                        break

                if valid_to_retry:
                    self.log(f"rmPL: Redundantly running straggler task: {step.step_id}, dataset {dataset}")
                    return task

            return None

    def _clean_and_get_valid_straggler_task(self) -> tuple[int, str] | None:
            """ 
            Returns a task tuple (step_index:int, dataset:str) of a straggler task that is eligible to be redundantly run.
            If self.clear_orphan_p_log is set, then it clears p-logs of tasks not currently running.
            """

            undone_tasks: list[tuple[int, str]] = self._get_undone_tasks()
            
            if not undone_tasks:
                # We check this but don't log every time to avoid spamming the console 
                # while the last few processes are finishing.
                return None

            undone_task_instance_counts: list[int] = [self._count_task_instances(task) for task in undone_tasks]
            _, undone_tasks = _zipsort(undone_task_instance_counts, undone_tasks, reverse=True)

            for task in undone_tasks:
                step_index, dataset = task

                # 1. ORPHAN CLEANING LOGIC
                # If a task has a p-log but is NOT in our running list, it's an orphan (likely from a crash).
                if self.clear_orphan_p_log:
                    if self._has_its_p_file(step_index, dataset, 'p-log', true_on_any=True) and task not in self._running_processes:
                        self.log(f"rmPL: Clearing orphan p-log for step {self.pipeline_map[step_index].step_id}, dataset {dataset}")
                        file_io.create_p_file(self.pipeline_map, step_index, dataset, 'p-log', delete=True)

                # 2. REDUNDANCY LOGIC 
                # If redundancy is disabled (<= 0), we don't return None anymore; 
                # we 'continue' so the loop can clean orphans for the rest of the tasks.
                if self.redundant_process_limit <= 0:
                    continue

                instance_count = self._count_task_instances(task)

                # Only consider tasks already in flight as candidates for redundant relaunch.
                # If instance_count is 0, it's a fresh task that _find_legal_task should handle.
                if instance_count == 0:
                    continue

                # Step must still be legal from a resource perspective.
                if step_index not in self._get_legal_steps():
                    continue

                # Prereqs must still be satisfied; allow duplicate instances.
                if not self._is_legal_dataset_step(step_index, dataset, ensure_no_duplicates=False):
                    continue

                step = self.pipeline_map[step_index]

                # If no penalties are defined, we cannot safely budget redundant work.
                if not step.resource_penalties:
                    continue

                valid_to_retry = True
                for resource, penalty in step.resource_penalties.items():
                    limit = self.resource_limits[resource]

                    # Guard against invalid limits.
                    if limit <= 0:
                        valid_to_retry = False
                        break

                    if ((penalty * (instance_count + 1)) / limit) > self.redundant_process_limit:
                        valid_to_retry = False
                        break

                if valid_to_retry:
                    self.log(f"rmPL: Redundantly running straggler task: {step.step_id}, dataset {dataset}")
                    return task

            return None

    if False:
        def _p_launch_func(self, func: Callable, on_return:list[RouterType])->Callable:
            """ Returns a callable to a wrapper function which runs func with kwargs and then updates progress when done.
            """
            def wrapped_func(step_index:int, dataset:str, **kwargs)->None: 
                return_val: Any = func(**kwargs)

                if on_return:
                    ORIS: OnReturnInfoStruct = OnReturnInfoStruct(self.pipeline_map, dataset, step_index, self.state_dict, self.lock)
                    for router in on_return:
                        router(return_val, ORIS)

                state_management.set_state_dict_progress(self.state_dict, self.lock, dataset, step_index, 100)
                

            return wrapped_func
        


    def _p_launch_func(self, func: Callable, on_return: list[RouterType]) -> Callable:
            def wrapped_func(step_index: int, dataset: str, **kwargs) -> None:
                # 1. Determine the log directory based on the Step's output folder
                # We look at the first output directory defined for this step
                the_step = self.pipeline_map[step_index]
                
                # Use the first output directory defined for this step
                base_out_dir = the_step.out[0][0] 
                log_dir = os.path.join(base_out_dir, ".pipeline", "logs", dataset)

                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, "stdout_err.txt")

                # 2. Redirect to the file
                with open(log_path, "a", buffering=1) as log_file:
                    sys.stdout = log_file
                    sys.stderr = log_file

                    try:
                        return_val: Any = func(**kwargs)

                        if on_return:
                            ORIS: OnReturnInfoStruct = OnReturnInfoStruct(self.pipeline_map, dataset, step_index, self.state_dict, self.lock)
                            for router in on_return:
                                router(return_val, ORIS)

                        state_management.set_state_dict_progress(self.state_dict, self.lock, dataset, step_index, 100)
                    except Exception:
                        print(f"\n--- ERROR IN STEP {the_step.step_id} ---", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                        raise 
                    finally:
                        log_file.flush()

            return wrapped_func


    def _update_resource_utilization(self, step_index:int, starting_task:bool)->None:
        """ Updates the resource utilization and last utilization to reflect starting a run of step step_index.
        Args:
            starting_task : bool, set to true to increase utilization, etc, or false to signify the completion of a task.
        """
        
        the_step: Step = self.pipeline_map[step_index]

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

        the_step:Step = self.pipeline_map[step_index]
        
        # Let's build the full kwargs.
        full_kwargs: dict = the_step.kwargs.copy()
        
        for i, inp_dir_ext_tuple in enumerate(the_step.inp):
            kwarg = the_step.special_kwargs[i]
            # if no kwarg, the function doesn't take this argument.
            # thus we will move to the next 
            if not kwarg:
                continue

            full_kwargs[kwarg] = os.path.join(inp_dir_ext_tuple[0], dataset + '.' + inp_dir_ext_tuple[1])
        for j, out_dir_ext_tuple in enumerate(the_step.out):
            i = j+len(the_step.inp)
            kwarg = the_step.special_kwargs[i]
            if not kwarg:
                continue
            full_kwargs[kwarg] = os.path.join(out_dir_ext_tuple[0], dataset + '.' + out_dir_ext_tuple[1])

        if the_step.process_state_functions_kwarg:
            full_kwargs[the_step.process_state_functions_kwarg] = state_management.get_process_state_functions(self.pipeline_map, self.state_dict, self.lock, dataset, step_index)

        return full_kwargs


    def _start_multiprocessing_step_dataset(self, step_index:int, dataset:str)->None:
        """ Starts the multiprocessing process for any step and dataset. Does not check anything. Adds a .p-log. Also adds to the resource utilization.
        """
        
        the_step:Step = self.pipeline_map[step_index]
        
        full_kwargs:dict[str, Any] = self._build_kwargs(step_index, dataset)
        
        # Let's wrap the function to add the lock when done:
        target_func:Callable = self._p_launch_func(the_step.function, the_step.on_return)
        p = multiprocessing.Process(target=target_func, args=(step_index, dataset), kwargs=full_kwargs)

        file_io.create_p_file(self.pipeline_map, step_index, dataset, 'p-log')
        p.start()

        self._update_resource_utilization(step_index, starting_task=True)
        
        self.log(f"rmPP: Started {the_step.function.__name__} for dataset {dataset}.")
        self._running_processes.append((step_index, dataset))


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
            self._stream_logs_to_console()

            try:
                if self.use_vertical:
                    task = self._find_legal_task_vertical()
                else:
                    task = self._find_legal_task()
            except ResourceExhaustionException:
                time.sleep(self.resource_timeout / 1000)
                continue

            if not task:
                # Are we done?
                if self._are_all_steps_done():
                    stop_reason = "rmPP: All datasets have been completed for the final step."
                    continue

                # do we have stragglers to clean/restart?
                else: 
                    task = self._clean_and_get_valid_straggler_task()
                    if not task:
                        continue

            self._start_multiprocessing_step_dataset(task[0], task[1])
            time.sleep(self.new_instance_timeout / 1000)

        print(f"rmPP: stop_reason: {stop_reason}")


