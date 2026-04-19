"""
This file serves to define the router functions and other subBlock specific applications. For OptionalBlock, for example,
we need to generate a router fun
"""

from .rmPL_Types import Block, CheckRetryBlock, OnReturnInfoStruct, OptionalBlock, Step, RouterType, RecursionBlock
from typing import Any
import shutil
from . import file_io
from . import state_management



def _get_all_step_ids_with_subBlock_id(pipeline_map:list[Step], block: Block, subBlock_type:type, subBlock_ID:int)->list[int]:
    """Returns a sorted list of step indices of steps in pipeline_map who have the right subBlock_ID"""
    this_block:list[int] = []
    for i, step in enumerate(pipeline_map):
        ids = step._subBlockIds.get(subBlock_type)
        if ids and subBlock_ID in ids:
            this_block.append(i)
    return sorted(this_block)


def generate_optional_block_router(optional_block: OptionalBlock, optional_block_ID:int)->RouterType:
    """
    This is called when processing and converting optional_blocks. Returns a router function for
    this specific optional_block, that should be attached to the first step in the optional_block.
    """
    def router(return_val: Any, ORIS: OnReturnInfoStruct):
        """ return_val==True means we continue with this OB, so we do nothing.
            if not True, then we need to skip everything inside. They already respect
            first-step prerequisite at this point.
        """

        if return_val==True:
            return
        
        # Now we need to skip every step, minus the last step, that is part of this
        # optional_block. We'll first need to get every step in this optional_block.

        this_block:list[int] = _get_all_step_ids_with_subBlock_id(ORIS.pipeline_map, optional_block, OptionalBlock, optional_block_ID)

        final_step_index = max(this_block)
        
        for intermediate_step_index in this_block:
            if intermediate_step_index == final_step_index:
                # we'll do this later.
                continue
            # set the progress to 100 to make the p-lock
            state_management.set_state_dict_progress(ORIS.state_dict, ORIS.lock, ORIS.dataset, intermediate_step_index, 100)

        # And for the final step, we will copy over the input.
        for index, dir_ext_tuple in enumerate(optional_block.output_DET):
            input_DET = optional_block.input_DET[index]

            if dir_ext_tuple not in ORIS.pipeline_map[final_step_index].out:
                raise ValueError("OptionalBlock.output_DET contains DET not in the final step's out.")
            
            file_to_set:str = file_io.get_dataset_filename(dir_ext_tuple, ORIS.dataset)
            file_to_copy_from:str = file_io.get_dataset_filename(input_DET, ORIS.dataset)         
            shutil.copy(file_to_copy_from, file_to_set)
            state_management.set_state_dict_progress(ORIS.state_dict, ORIS.lock, ORIS.dataset, final_step_index, 100)
        return
    return router


def generate_check_retry_block_router(check_retry_block: CheckRetryBlock, check_retry_block_ID:int)->RouterType:
    """ Returns a router. If the return value is True, then all the steps to be retried (either all, by default,
        or just those specified if check_retry_block.step_ids_to_retry is set.
    """
    def router(return_val:Any, ORIS: OnReturnInfoStruct)->None:
        if return_val is not True:
            return

        print(f"rmPL: Retrying dataset {ORIS.dataset}.")

        # Get the steps to undo.
        if check_retry_block.step_ids_to_undo is not None:
            step_ids_to_undo:list[str] = check_retry_block.step_ids_to_undo
        else:
            # We undo all steps except for the final one.
            # Find out all the steps in this one.
            this_block:list[int] = _get_all_step_ids_with_subBlock_id(ORIS.pipeline_map, check_retry_block, CheckRetryBlock, check_retry_block_ID)
            step_ids_to_undo:list[str] = [ORIS.pipeline_map[step_index].step_id for step_index in this_block if step_index != ORIS.step_index]

        file_io.undo_steps(ORIS.dataset, step_ids_to_undo, ORIS.pipeline_map)

        # Then we will set the progress of this final router step to -100 to free resources and delete g-log without marking done with a p-lock.
        state_management.set_state_dict_progress(ORIS.state_dict, ORIS.lock, ORIS.dataset, ORIS.step_index, -100)
    return router






