from multiprocessing.managers import DictProxy
from .rmPL_Types import ProcessStateFunctions, Step, process_state


def set_state_dict_cooldowns(state_dict:DictProxy, lock, dataset:str, step_index:int, resource:str, cooldown_ms: int)->None:
    with lock:
        this_dataset:dict[int, process_state] = state_dict[dataset]
        this_dataset[step_index].resource_cooldowns[resource] = cooldown_ms
        state_dict[dataset] = this_dataset
    return


def set_state_dict_progress(state_dict, lock, dataset:str, step_index:int, progress:int)->None:
    """ Sets the progress of a dataset of a Step with the state_dict DictProxy properly.

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


def get_process_state_functions(pipeline_map:list[Step], state_dict:DictProxy, lock, dataset:str, step_index:int)->ProcessStateFunctions:
    def set_progress_func(progress:int)->None:
        """
        Sets the progress of the current process.

        Args:
            progress:int from 0 to 100. Progress will automatically be set to 100 upon completion, so you need not do that to mark as complete.
        """
        set_state_dict_progress(state_dict, lock, dataset, step_index, progress)
        return
    def set_resource_cooldown_func(resource:str, cooldown_ms:int):
        """
        Reports resource cooldowns (eg, rate limits) to the main process.

        Args:
            resource:str, the same resource as defined in the resource_limits, etc. 
            cooldown_ms: int, the amount of time, in milliseconds, that we should wait starting now, before starting another Step that uses this resource.
        """
        set_state_dict_cooldowns(state_dict, lock, dataset, step_index, resource, cooldown_ms)
        return

    # now get the resource names.
    resource_names:list[str] = list(pipeline_map[step_index].resource_penalties.keys())
    resource_names:list[str] = [resource for resource in resource_names if resource.lower() != 'overall']

    return ProcessStateFunctions(set_progress_func=set_progress_func, set_resource_cooldown_func=set_resource_cooldown_func, rate_limit_resource_names=resource_names)


