from time import time

def _timeit(func, indent):
    name = func.__name__
    def _func(*args, **kwargs):
        print(indent + ' Function <%s> starts. '%name)
        start = time()
        res = func(*args, **kwargs)
        print(indent + ' Function <%s> ends! Time cost = %f sec. '%(name, time()-start))
        return res
    return _func

def timeit(indent=""):
    if isinstance(indent, str):
        return lambda func: _timeit(func, indent)
    else:
        return _timeit(indent, "")