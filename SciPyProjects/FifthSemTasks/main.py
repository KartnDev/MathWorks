import time



import cupy, numpy


from numba import jit, prange
import numpy as np



def cpu(a, b):
    start_time = time.time()
    b.reshape(1, len(b))
    res = np.multiply(a, b)
    print(f"Execution: {time.time() - start_time}")
    return res

def gpu(a, b):
    start_time = time.time()
    res =
    print(f"Execution: {time.time() - start_time}")
    return res

N = 15000

a = numpy.random.randn(N, N) * 100
b = numpy.random.randn(N) * 100
cpu_res = cpu(a, b)

a = cupy.asarray(a)
b = cupy.asarray(b)
gpu_res = gpu(a, b)

print(cpu_res)
print(gpu_res)

