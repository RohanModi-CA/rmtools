import os
from .rmPL_Types import Step

def get_dataset_filename(dir_ext_tuple:tuple[str,str], dataset:str)->str:
    directory = dir_ext_tuple[0]
    extension = dir_ext_tuple[1]
    return os.path.join(directory, f'{dataset}.{extension}')


def create_p_file(pipeline_map:list[Step], step_index, dataset, p_ext:str, delete:bool=False)->None:
    """ p_ext is either 'p-lock' or 'p-log'. if delete it deletes the p-file.
    """
    the_step: Step = pipeline_map[step_index]
    for dir_ext_tuple in the_step.out:
        pipeline_dir = os.path.join(dir_ext_tuple[0], '.pipeline')
        os.makedirs(pipeline_dir, exist_ok=True)
        log_filename = os.path.join(pipeline_dir, f'{dataset}.{dir_ext_tuple[1]}.{p_ext}')
        if not delete:
            with open(log_filename, 'a'):
                pass
        else:
            if os.path.exists(log_filename):
                os.remove(log_filename)


def undo_steps(dataset:str, step_ids: list[str], pipeline_map: list[Step])->None:
    """
    For a given dataset, this function will delete the p-logs and p-locks of the steps in the list. 
    """

    steps_to_delete: list[Step|None] = [step_i if step_i.step_id in step_ids else None for step_i in pipeline_map]

    for step_index, _ in enumerate(steps_to_delete):
        create_p_file(pipeline_map, step_index, dataset, 'p-lock', delete=True)
        create_p_file(pipeline_map, step_index, dataset, 'p-log', delete=True)
    return


