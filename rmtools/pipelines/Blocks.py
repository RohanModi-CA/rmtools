from __future__ import annotations
from typing import Callable, Any
from .rmPL_Types import Block, CheckRetryBlock, OptionalBlock, Step
from dataclasses import dataclass, replace
from . import CustomBlocks
import random

####### ROUTERS FOR THE SUBCLASSES OF BLOCKS ########

def _process_deepest_optionalBlock(optional_block:OptionalBlock, optional_block_ID:int)->Block:
    """ For optionalBlocks, we just need to give them a unique optionalBlock_ID.
    """
    assign_block_ids(optional_block, OptionalBlock, optional_block_ID)


    # We also need to force them to respect the first-step prerequisite.
    _force_first_step_prerequisite_onto_block(optional_block)



    # And we want the first block to have the router for this block.
    if optional_block.steps: # I guess we don't want to break empty optionalBlocks, though that is undefined behaviour.
        first_step:Step = _get_nth_flattened_step(optional_block, 0)
        first_step.on_return.append(CustomBlocks.generate_optional_block_router(optional_block, optional_block_ID))

        # And we need to ensure that the 'short-circuit' skip file is prerequisite.
        current_inps:int = len(first_step.inp)
        for dir_ext_tuple in optional_block.input_DET:
            first_step.inp.append(dir_ext_tuple)
            # We can put them all at that index and it just pushes the others down.
            first_step.special_kwargs.insert(current_inps, "")


    out:Block = Block(steps=optional_block.steps)
    return out


def _process_check_reply_block(check_retry_block:CheckRetryBlock, check_retry_block_ID:int)->Block:
    """ These just need their IDs and to assign the final one their router. 
    """

    assign_block_ids(check_retry_block, CheckRetryBlock, check_retry_block_ID)
    
    if check_retry_block.steps:
        last_step:Step = _get_nth_flattened_step(check_retry_block, -1)
        last_step.on_return.append(CustomBlocks.generate_check_retry_block_router(check_retry_block, check_retry_block_ID))
    out:Block = Block(steps=check_retry_block.steps)
    return out




####### ENSURE THEY ARE IN HANDLER #############

SUB_BLOCK_HANDLERS:dict[type, Callable[[Any, int],Block]] = {
        OptionalBlock: _process_deepest_optionalBlock,
        CheckRetryBlock: _process_check_reply_block
}


################################################

@dataclass
class _mutableInt():
    """ used just as a recursion counter, basically """ 
    val:int=1


def assign_block_ids(subBlock:Block, subBlock_type:type, subBlock_ID:int)->None:
    def assign_step_block_id(step:Step)->None:
        if subBlock_type not in step._subBlockIds.keys():
            step._subBlockIds[subBlock_type]=[]
        step._subBlockIds[subBlock_type].append(subBlock_ID)
        return
    _apply_func_on_all_steps_within(subBlock, assign_step_block_id)
    return


def _init_step_ids(pipeline_map: Block)->None:
    """ Goes through the steps and assigns each a random step_id=str(flattened_index) if it does not already have one. Avoids collisions. """
    flattened:list[Step] = _complete_flatten_block(pipeline_map)

    for step in flattened:
        existing_ids:list[str] = [step.step_id for step in flattened]
        while not step.step_id:
            random_id = str(random.randint(0, 2**30))
            if random_id not in existing_ids:
                step.step_id = random_id
    return


def _complete_flatten_block(block: Block)->list[Step]:
    flattened:list[Step] = []
    for val in block.steps:
        if isinstance(val, Step):
            flattened.append(val)
        if isinstance(val, Block):
            flattened += _complete_flatten_block(val)
    return flattened


def _get_nth_flattened_step(block: Block, n:int)->Step:
    flattened:list[Step] = _complete_flatten_block(block)
    nth_step:Step = flattened[n]
    return nth_step


def _apply_func_on_all_steps_within(block: Block, func:Callable[[Step], Any])->None:
    """ Will iterate through Block, and on each step within it or its nested blocks,
        it will apply func(step). 
    """
    for step in _complete_flatten_block(block):
        func(step)
    return

def _is_block_without_nested_subBlock(block: Block, subBlock_to_avoid:type)->bool:
    """This does not check that block is not a subBlock_to_avoid.
    """
    for step in block.steps:
        if isinstance(step, subBlock_to_avoid):
            return False
        if isinstance(step, Block):
            if not _is_block_without_nested_subBlock(step, subBlock_to_avoid):
                return False
    return True


def _process_single_subBlock(subBlock:Block, subBlock_type:type, block_ID:int)->Block:
    """ A router for each subblock type to call their specific unnester handling function."""
    handler: Callable[[Block, int], Block] | None = SUB_BLOCK_HANDLERS.get(subBlock_type)
    if not handler:
        raise ValueError("_process_single_subBlock: unsupported subBlock_type")
    return handler(subBlock,block_ID)



def _process_and_convert_subBlock(pipeline_map:Block, subBlock_type:type)->Block:
    """ Calls the _process_single_subBlock function on each nested subBlock of subBlock_type. Starts at the deepest ones.
        Returns a Block with the same properties as pipeline_map but all instances of subBlock have been converted to
        plain Blocks.
    """
    def _recursive_call_process_convert(pipeline_map:Block, subBlock_type:type, deepest_block_count:_mutableInt|None=None)->Block:
        """ This may be O(n^2) instead of O(N) but I think it is much more readable this way and the fact of the matter is that
            these operations are quick and the pipeline_maps people give are not going to be long enough for this to be a major
            bottleneck.
        """
        if not deepest_block_count:
            deepest_block_count = _mutableInt(val=1)

        result_list: list[Step | Block] = []
        
        for val in pipeline_map.steps:
            if isinstance(val, Step):
                result_list.append(val)
            if isinstance(val, Block):
                if _is_block_without_nested_subBlock(val, subBlock_type):
                    # val could still be itself a subBlock_type. If it's not, good:
                    if not isinstance(val, subBlock_type):
                        result_list.append(val)
                    else:
                        # This is a subBlock that does not contain any subBlocks. Thus it is a deepest subBlock. Process it to turn it into a regular Block.
                        result_list.append(_process_single_subBlock(val, subBlock_type, deepest_block_count.val))
                        deepest_block_count.val += 1
                else:
                    # If there are nested subBlocks within this, go in, and try again.
                    # Easiest case if this is not a subBlock:
                    if not isinstance(val, subBlock_type):
                        result_list.append(_recursive_call_process_convert(pipeline_map=val, subBlock_type=subBlock_type, deepest_block_count=deepest_block_count))
                    else:
                        # If it is, then we need to process both its children, then it.
                        # First process the children:
                        val = _recursive_call_process_convert(val, subBlock_type=subBlock_type, deepest_block_count=deepest_block_count)
                        
                        # Then, it needs to be processesed itself.
                        val = _process_single_subBlock(val, subBlock_type=subBlock_type, block_ID=deepest_block_count.val)
                        deepest_block_count.val += 1

                        # And all this entire thing back:
                        result_list.append(val)

        # We want to copy all fields except the steps which we've done work on.
        out: Block = replace(pipeline_map, steps=result_list)

        return out

    converted_and_processed: Block = _recursive_call_process_convert(pipeline_map, subBlock_type)
    return converted_and_processed
                





def _force_first_step_prerequisite_onto_block(block: Block)->None:
    """ For optional_blocks, and perhaps for others, the first step
        is special, and we want to ensure that no other steps can
        proceed without it. Thus we will manually make the first
        step a prerequisite, but we will ensure it does not get passed
        to the function. The user may have done this themselves, but
        prereqs that are not passed to the function are ignored so
        that should not cause issues.
    """
    flattened = _complete_flatten_block(block)
    
    for dir_ext_tuple in flattened[0].out:
        # add this as a prerequisite to inp of the functions with no kwarg.
        for index, step in enumerate(flattened):
            if index==0:
                continue

            insertion_index = len(step.inp)
            step.inp.append(dir_ext_tuple)

            # Now we need to insert this "" at the corresponding place; after all the inps.
            step.special_kwargs.insert(insertion_index, "")
    return


"""
The main event in terms of Blocks is actually just the unwrapping. So let's just get a general flattener.
"""

def _final_flatten(pipeline_map:Block)->list[Step]:
    """Flattens a Block whose only components are steps or *plain* blocks. 
       Will error if pipeline_map contains a subBlock. Returns the flattened
       list[Step]
    """
    
    # First let's just go through and check that there are no subClasses.
    for subBlock_type in SUB_BLOCK_HANDLERS:
        if not _is_block_without_nested_subBlock(pipeline_map, subBlock_type):
            raise ValueError("_final_flatten: cannot have subBlocks at the flattening stage.")

    # Now, let's flatten, which we'll do recursively.
    def _recursive_flatten(block: Block)->list[Step]:
        out: list[Step] = []
        
        for val in block.steps:
            if isinstance(val, Step):
                out.append(val)
            elif isinstance(val, Block):
                out += _recursive_flatten(val)
            else:
                raise Exception("_final_flatten: Should be unreachable. Logic Error.")
        return out

    out = _recursive_flatten(pipeline_map)
    return out

def _full_process(pipeline_map:Block)->Block:
    """ Calls the process_and_convert on each subBlock_type in SUB_BLOCK_HANDLERS to create
        a Block that contains only Blocks without subBlocks.
    """

    cleaned_Block: Block = pipeline_map

    for subBlock_type in SUB_BLOCK_HANDLERS:
        cleaned_Block: Block = _process_and_convert_subBlock(cleaned_Block, subBlock_type)

    return cleaned_Block


def process_and_flatten_input_pipeline_map(pipeline_map:Block)->list[Step]:
    """ Takes an input pipeline_map from the user, containing nested Blocks and subBlocks,
        processes them according to the rules for each subBlock type, and then flattens into
        a list of steps for execution.
    """

    #TODO: Fix, right now need this hacky solution:
    PM = Block(steps=[pipeline_map])

    # Assign every step a step_id if needed.
    _init_step_ids(PM)

    # First step: get rid of the subBlocks.
    cleaned_Block: Block = _full_process(PM)

    # Now just flatten this
    flattened: list[Step] = _final_flatten(cleaned_Block)

    return flattened


