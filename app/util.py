import os

def is_safe_path(basedir, path):
    """
        chceks if path is safe
    """
    return os.path.commonprefix((os.path.realpath(path), basedir)) == basedir


def safe_join(s1, s2):
    """
        Returns a join if safe
    """
    if is_safe_path(s1, s2):
        return os.path.join(s1, s2)
    else:
        raise NotADirectoryError("Bad path.")