from dataclasses import dataclass, field
from typing import Any, Callable, Union, Sequence
from multiprocessing.managers import DictProxy




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
class Step():
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
    on_return:Callable|None=None

_ListStep = Sequence[Step]
_NestOne  = Sequence[Union[Step, _ListStep]]
NestedSteps  = Sequence[Union[Step, _NestOne]]

@dataclass
class OnReturnInfoStruct():
    pipeline_map: list[Step]
    dataset:str
    step_index:int
    state_dict:DictProxy
    lock:Any

@dataclass(frozen=True)
class ParallelOptions():
    """
    Args:
        pipeline_map: A list of Steps. You can nest them, if that is easier, with a limit of 3 layers of total depth. (No nesting is one layer).
        resource_limits: dict[str, int] A dictionary keyed by resource types and valued by the limits on the resources.
        resource_timeout: int=1000, The time in milliseconds, after a resource exhaustion, before we can attempt to try again.
        new_instance_timeout:int=20, The time in milliseconds before any attempt to start a new Step. 
        resource_cooldowns:dict[str, resource_cooldown]={}, optional. Keyed by resources. Valued by resource_cooldown's. Only need to touch resource_cooldown.fixed_ms. 
        redundant_process_limit:float=0.0: The portion of the resource limits that each redundant process can use at a time.
        restrict_datasets:tuple[str,str]|None=None: Only run pipeline on datasets between restrict_datasets[0] and restrict_datasets[1], inclusive. Relies on string sorting.
        clear_existing:bool=False. If true, all outputs for active datasets (all by default or those in restrict_datasets), will be cleared at the beginning of the program. 
    """
    pipeline_map: NestedSteps
    resource_limits:dict[str, int]
    resource_timeout_ms:int=1000
    new_instance_timeout_ms:int=20
    resource_cooldowns:dict[str,resource_cooldown]=field(default_factory=dict)
    use_vertical:bool=True
    clear_orphan_p_log:bool=True
    redundant_process_limit:float=0.0
    restrict_datasets:tuple[str,str]|None=None
    clear_existing:bool=False


