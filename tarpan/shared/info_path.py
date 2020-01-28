"""Finding a path where analysis files are saved"""

from dataclasses import dataclass
import inspect
import os


@dataclass
class InfoPath:
    """
    Path information for creating summaries
    """

    # Used in `sub_dir_name` to indicate that sub-directory should
    # not be created, and the file is placed in parent directory instead.
    DO_NOT_CREATE: str = "do not create"

    """
    Name of the top directory in which a sub-subdirectory `sub_dir_name`
    containing plots is placed.
    """
    dir_name: str = "model_info"

    """
    Name of the subdirectory. If None, determined automatically
    based on the name of the python script file from which a
    Tarpan's function is called.
    """
    sub_dir_name: str = None

    base_name: str = None  # Base name of the summary file or plot
    extension: str = None  # Extension of the summary or plot

    """
    DPI setting used for plots. Determines the quality of the plots
    """
    dpi: int = 300

    """
    Path to the python type from which a Tarpan function was called.
    This `codefile_path` variable is determined automatically.
    This path is then used to automatically create subdirectory
    `sub_dir_name` where plots are saved.
    """
    codefile_path: str = None

    """
    Full path to parent of info dir. Auto if None. If None, determined
    automatically based on the parent directory of the python script file
    from which a Tarpan's function is called.
    """
    path: str = None

    def set_codefile(self):
        """
        Sets the path to python file that called a Tarpan function.

        Returns
        --------
        InfoPath
            The path object with codefile_path set.
        """

        if self.codefile_path is not None:
            # The path is already set
            return

        tree_depth = 2
        frame = inspect.stack()[tree_depth]
        module = inspect.getmodule(frame[0])
        self.codefile_path = module.__file__


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

    info_path.set_codefile()
    full_path = get_info_dir(info_path)
    filename = f'{info_path.base_name}.{info_path.extension}'
    return os.path.join(full_path, filename)


def get_info_dir(info_path=InfoPath()):
    """
    Get full path to the directory where plots and summaries are placed

    Parameters
    ----------
    info_path : InfoPath
        Path information for creating summaries.


    Returns
    --------
    str
        Path to an analysis file.
    """

    info_path.set_codefile()

    if info_path.path is None:
        info_path.path = os.path.dirname(info_path.codefile_path)
    else:
        info_path.path = os.path.expanduser(info_path.path)

    full_path = os.path.join(info_path.path, info_path.dir_name)

    if info_path.sub_dir_name != info_path.DO_NOT_CREATE:
        if info_path.sub_dir_name is None:
            info_path.sub_dir_name = os.path.basename(info_path.codefile_path).rsplit('.', 1)[0]

        full_path = os.path.join(full_path, info_path.sub_dir_name)

    os.makedirs(full_path, exist_ok=True)

    return full_path
