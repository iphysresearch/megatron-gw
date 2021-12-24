import msgpack_numpy as m
m.patch()               # Important line to monkey-patch for numpy support!

import redis
import numpy as np
from tqdm import tqdm

connection_pool = redis.ConnectionPool(host='localhost', port=6379, db=0, decode_responses=False)
r = redis.Redis(connection_pool=connection_pool)

def toRedis(value,name):
    """Store given Numpy array 'value' in Redis under key 'name'"""
    r.set(name,m.packb(value))
    return

def fromRedis(r,n):
    """Retrieve Numpy array from Redis key 'n'"""
    # Retrieve and unpack the data
    try:
        return m.unpackb(r.get(n))
    except TypeError:
        print('No this value')

def set_get_Redis(r, value, name):
    try:
        return m.unpackb(r.set(name, m.packb(value), get=True))    
    except TypeError:
        return None

def is_exist_Redis(name):
    return r.exists(name) 

def clear_Redis():
    for key in tqdm(r.keys('*')):
        r.delete(key)
    r.flushall()
    r.flushdb()