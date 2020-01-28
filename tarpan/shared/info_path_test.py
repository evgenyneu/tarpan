import os
from tarpan.shared.info_path import InfoPath, get_info_path, get_info_dir


def test_set_codefile():
    info_path = InfoPath()

    set_codefile_sun_function(info_path)

    assert os.path.basename(info_path.codefile_path) == "info_path_test.py"


def set_codefile_sun_function(info_path):
    info_path.set_codefile()


def test_get_info_path():
    info_path = InfoPath()
    info_path.base_name = 'my_basename'
    info_path.extension = 'test_extension'
    result = get_info_path(info_path)

    assert 'tarpan/shared/model_info/info_path_test/my_basename\
.test_extension' in result


def test_get_info_dir():
    info_path = InfoPath()
    result = get_info_dir(info_path)

    assert 'tarpan/shared/model_info/info_path_test' in result
