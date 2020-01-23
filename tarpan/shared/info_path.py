"""Finding a path where analysis files are saved"""

from dataclasses import dataclass
import inspect
import os


@dataclass
class InfoPath:
    # Used in `sub_dir_name` to indicate that sub-directory should
    # not be created, and the file is placed in parent directory instead.
    DO_NOT_CREATE: str = "do not create"

    # Path information for creating summaries
    path: str = None  # Full path to parent of info dir. Auto if None.
    dir_name: str = "model_info"  # Name of the info directory.
    sub_dir_name: str = None  # Name of the subdirectory. Auto if None.
    stack_depth: int = 3  # Used fo automatic `path` and `sub_dir_name`
    base_name: str = None  # Base name of the summary file or plot
    extension: str = None  # Extension of the summary or plot
    dpi: int = 300  # DPI setting using for plots


def get_info_path(info_path=InfoPath()):
    """
    Get full path to the plot or summary file.

    Parameters
    ----------
    info_path : InfoPath
        Path information for creating summaries.


    Returns
    --------
    str
        Path to an analysis file.
    """

    info_path = InfoPath(**info_path.__dict__)
    frame = inspect.stack()[info_path.stack_depth]
    module = inspect.getmodule(frame[0])
    codefile = module.__file__

    if info_path.path is None:
        info_path.path = os.path.dirname(codefile)
    else:
        info_path.path = os.path.expanduser(info_path.path)

    full_path = os.path.join(info_path.path, info_path.dir_name)

    if info_path.sub_dir_name != info_path.DO_NOT_CREATE:
        if info_path.sub_dir_name is None:
            info_path.sub_dir_name = os.path.basename(codefile).rsplit('.', 1)[0]

        full_path = os.path.join(full_path, info_path.sub_dir_name)

    os.makedirs(full_path, exist_ok=True)
    filename = f'{info_path.base_name}.{info_path.extension}'
    return os.path.join(full_path, filename)
