import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule

f = open("cuda_kernel.cu", 'r')
sm = pycuda.compiler.SourceModule(f.read(), options=['-lineinfo'])

def decipher(num_rounds, input_data, key):
    func = sm.get_function("decipher")
    """XTEA implementation in python, decryption.

    Modified version from Simon Biewald (http://varbin.github.io/xtea/)

    Arguments:
    num_rounds -- the number of iterations in the algorithm, 32 is reccomended
    input_data -- the input data to use, 32 bits of the first 2 elements are used
    key -- 128-bit key to use

    returns -- a numpy array containing the deciphered data"""
    v0 = input_data[0]
    v1 = input_data[1]
    delta = 0x9e3779b9L
    mask = 0xffffffffL
    sum = (delta*num_rounds) & mask

    # number of thread
    n_thread = len(num_rounds)
    func(drv.InOut(v0), drv.InOut(v1), drv.In(sum), block=(n_thread,1,1))

    return np.array([v0, v1], dtype=np.uint32)


def multiply_them():
    # get function pointer
    func = sm.get_function("multiply_them")

    a = np.random.randn(400).astype(np.float32)
    b = np.random.randn(400).astype(np.float32)
    dest = np.zeros_like(a)

    func(drv.Out(dest), drv.In(a), drv.In(b),block=(400,1,1))

    print dest-a*b

