import pickle
import json

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
        data = json.load(f)
        return data