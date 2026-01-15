from typing import Callable, Any
from rmPL_Types import Block, OptionalBlock, Step
from dataclasses import dataclass, replace
import rmPL

@dataclass
class _mutableInt():
    val:int=1

def _complete_flatten_block(block: Block)->list[Step]:
    flattened:list[Step] = []
    for val in block.steps:
        if isinstance(val, Step):
            flattened.append(val)
        if isinstance(val, Block):
            flattened += _complete_flatten_block(val)
    return flattened



def _apply_func_on_all_steps_within(block: Block, func:Callable[[Step], Any])->None:
    """ Will iterate through Block, and on each step within it or its nested blocks,
        it will apply func(step). 
    """
    for step in _complete_flatten_block(block):
        func(step)
    return

def is_block_without_nested_subBlock(block: Block, subBlock_to_avoid:type)->bool:
    """This does not check that block is not a subBlock_to_avoid.
    """
    for step in block.steps:
        if isinstance(step, subBlock_to_avoid):
            return False
        if isinstance(step, Block):
            if not is_block_without_nested_subBlock(step, subBlock_to_avoid):
                return False
    return True


def process_deepest_optionalBlock(optional_block:OptionalBlock, optional_block_ID:int)->Block:
    """ For optionalBlocks, we just need to give them a unique optionalBlock_ID.
    """
    def assign_optional_block_ID(step:Step)->None:
        step.optional_block_IDs.append(optional_block_ID)
        return

    _apply_func_on_all_steps_within(optional_block, assign_optional_block_ID)
    out:Block = Block(steps=optional_block.steps)
    return out




def process_single_subBlock(subBlock:Block, subBlock_type:type, block_ID:int)->Block:
    """ A router for each subblock type to call their specific unnester handling function."""

    if subBlock_type == OptionalBlock:
        if not isinstance(subBlock, OptionalBlock):
            raise ValueError("subBlock should be an instance of subBlock_type.")
        return process_deepest_optionalBlock(subBlock, block_ID)
    else:
        raise ValueError("process_single_subBlock: unsupported subBlock_type")



def process_and_convert_subBlock(pipeline_map:Block, subBlock_type:type)->Block:
    """ Calls the process_single_subBlock function on each nested subBlock of subBlock_type. Starts at the deepest ones.
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
                if is_block_without_nested_subBlock(val, subBlock_type):
                    # val could still be itself a subBlock_type. If it's not, good:
                    if not isinstance(val, subBlock_type):
                        result_list.append(val)
                    else:
                        # This is a subBlock that does not contain any subBlocks. Thus it is a deepest subBlock. Process it to turn it into a regular Block.
                        result_list.append(process_single_subBlock(val, subBlock_type, deepest_block_count.val))
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
                        val = process_single_subBlock(val, subBlock_type=subBlock_type, block_ID=deepest_block_count.val)
                        deepest_block_count.val += 1

                        # And all this entire thing back:
                        result_list.append(val)

        # We want to copy all fields except the steps which we've done work on.
        out: Block = replace(pipeline_map, steps=result_list)

        return out

    converted_and_processed: Block = _recursive_call_process_convert(pipeline_map, subBlock_type)
    return converted_and_processed
                



 
