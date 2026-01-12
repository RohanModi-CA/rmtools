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

