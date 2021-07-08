import os

def is_safe_path(basedir, path):
    """
        Returns true if a path is safe
            A safe path is a path in which the base dir is a
            common prefix between both itself and the path.
            /data == /data/test
            /data != /test
    """
    return os.path.commonprefix((os.path.realpath(path), basedir)) == basedir


def safe_join(s1, s2):
    """
        Returns a join if paths are safe
    """
    if is_safe_path(s1, s2):
        return os.path.join(s1, s2)
    else:
        raise NotADirectoryError("Bad path.")