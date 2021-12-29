import pickle
import json
import os
import fnmatch


# https://stackoverflow.com/a/58942584/8656360
def pickle_write(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def pickle_read(filename):
    with open(filename, "rb") as f:
        out = pickle.load(f)
    return out


# save data to json file
def json_store(data, filename):
    with open(filename, 'w') as fw:
        # 将字典转化为字符串
        # json_str = json.dumps(data)
        # fw.write(json_str)
        # 上面两句等同于下面这句
        json.dump(data,fw)


# load json data from file
def json_load(filename):
    with open(filename,'r') as f:
        return json.load(f)


def ffname(path, pat):
    """Return a list containing the names of the files in the directory matches PATTERN.

    'path' can be specified as either str, bytes, or a path-like object.  If path is bytes,
        the filenames returned will also be bytes; in all other circumstances
        the filenames returned will be str.
    If path is None, uses the path='.'.
    'Patterns' are Unix shell style:
        *       matches everything
        ?       matches any single character
        [seq]   matches any character in seq
        [!seq]  matches any char not in seq
    Ref: https://stackoverflow.com/questions/33743816/how-to-find-a-filename-that-contains-a-given-string
    """
    return [filename
            for filename in os.listdir(path)
            if fnmatch.fnmatch(filename, pat)]
