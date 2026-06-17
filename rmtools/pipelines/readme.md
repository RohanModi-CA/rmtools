Some notes on the functionality of the program.
---

The idea of this library is to make it simple to run several steps on a group of datasets independantly.
We adopt an approach where everything is stored in files. I've taken other approaches in the past:
trying to do everything in memory, or other methods, but in the end this is the simplest and most easy
to have progress save, to manually inspect the execution, and visualize everything.

In essense, this is just a runner of steps. Thus, we'll eventually run our 'pipeline_map', which is a list of steps.
Thus to get things going, one defines their list of steps.


---

Now, one thing that one often wants is a 'router': a step where a program looks at what has been done so far,
and then decides where to go from there. For example, you might have an LLM called at the end of your
pipeline to determine whether your final output is 'correct'. If the LLM decides it is not, you may want to
restart the pipeline for this dataset, either from the beginning or from somewhere else.

For these sort of purposes, in general, there is an `on_return:Callable=None` parameter when creating a `step`. Usually, the return value of
the `func` callable is discarded, and the library just marks the step as done, and moves on. If `on_return` is
set, however, we will call `on_return` after `func` is done. `on_return` must take exactly two positional arguments.
After `func` is done, its return value will be saved, and passed as the first argument to the `on_return`. The second
argument will be passed the `OnReturnInfoStruct`. 

If all you want is a basic verificiation router, we make available `Parallel.gen_verification_router()->Callable`. The function generated
by this method will check to see whether the prior `Func` returns `False`. If it returns `False`, then it will undo the steps 
passed to it. 

```python3
from rmtools import pipelines as rmPL

router:Callable = Parallel.gen_verification_router(['StepToDeleteID_1', 'StepToDeleteID_2')

my_step = rmPL.Step(
    inp=[('prev_step_dir/', 'txt')],
    out=[('this_step_dir/', 'txt')],
    func=func,
    special_kwargs=["inp_dir", "out_dir"],
    on_return=router,
    step_id='VerificationStep'
)
```

Now, `OnReturnInfoStruct` contains information for routers. It has: `pipeline_map:list[Step], dataset:str, step_index:int, state_dict:dict, lock:Lock`. 


Note: by default, steps will set their progress to 100 when done, creating a g-lock, preventing them from being worked on. This behaviour is usually unwanted for routers. Setting progress to -100 in an on_return function will remove g-log, free resources, but not mark the file as done. gen_verification_router does this automatically if return_val == False.

---py

Nesting Steps in a Pipeline Map
---


Nesting lists of Steps within a Pipeline map triggers custom behaviour.
This creates a `block` of Steps. `Blocks` are meant to serve as optional step pipelines. 


---

OptionalBlock
---

A common thing one wants to do is define a task, or series of tasks, that happen sometimes, and not other times. OptionalBlock was created for this.
An OptionalBlock, like all other Blocks, contains a .steps item, which is list[Step|Block]. For example, we might define one like this:

```python3
OB = OptionalBlock(steps=
[
    S1: Step,
    S2: Step,
    B1: Block,
    OB1: OptionalBlock,
    S3: Step
]
)
```

The first step in an OptionalBlock is special: It is the step that determines whether or not the other steps within the OptionalBlock will be run.
If this first step returns True, the following Steps and Blocks will run. If it does not return True, then it will skip them. 

To do this, all intermediate steps will be marked as completed through `p-lock`s, but no output files will actually be made. It is strongly recommended to
never use those files as inputs to steps outside of the OptionalBlock. Now, the final step is again special. This final step is sort of the 'output' of the
OptionalBlock. It should depend on the outputs of the intermediate steps. 

Should we have skipped the intermediate steps, the program will automatically mark this final step as completed through a `p-lock`. In this case, though,
we will create output files for this step. This way, other steps can either take advantage of whatever work was produced through the optional block, or
just continue with whatever was there before. Thus, we must set `OptionalBlock.input_DET` to be a list of dir-ext-tuples that existed before the start
of the OptionalBlock. These will be put into the `OptionalBlock.output_DET` output of the final step. The program will error if `finalStep.out` contains
dir_ext_tuples not within `OptionalBlock.output_DET`.


Note: if the first/last step is a Block, then this refers to the first/last step of that Block. I don't recommend doing that, since it becomes complicated mentally,
but the program will handle it just fine.


CheckRetryBlock
---

A very common process is: do task 1, do task 2, then check tasks 1 and 2, and restart them if unsatisfactory. This is the role of `CheckRetryBlock`. 

This SubBlock has one special step: the final one. If the final step returns True, we will delete the outputs of all the steps within the Block, and
start over. 

If you do not want this behaviour, and you only want to reset certain steps, you can set `CheckRetryBlock.step_ids_to_undo`. By default, it
is `None`, but if it is a `list[str]`, then only those steps will be reset. Careful with that, because you might cause an infinite loop. Do not include
the step_id of the final step in `CheckRetryBlock.step_ids_to_undo`. 


RecursionBlock
---

Sometimes, you'd like to recursively iterate on your data. For example, perhaps you put an image in an upscaler. Then, you check, but you determine it is not upscaled enough, 
so you want to upscale the upscaled image, again, and again. For this, we have RecursionBlocks. 








