import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
import precode
from pycuda.compiler import SourceModule

f = open("cuda_kernel.cu", 'r')
sm = pycuda.compiler.SourceModule(f.read(), options=['-lineinfo'])

# decrypt_bytes function which will call the cuda_kernel
def decrypt_bytes(bytes_in, key):
    # get the function from the cuda_kernel
    func = sm.get_function("decipher")

    # each thread will have a cur_guess
    iv = np.array([1,2], dtype=np.uint32)
    ha = np.fromstring(hashlib.md5(key).digest(), np.uint32)

    output = np.empty_like(bytes_in)
    prev_decrypt = iv

    # assign a number < 1024 for the number of threads for each block
    num_thread = 512

    # number of blocks is the total length divided for the number of threads per block and + 1 in case of rest
    num_block = (len(bytes_in) / num_thread) + 1


    # try cuda_kernel with passing of parameters
    func(drvInOut(bytes_in),block=(num_thread,1,1), grid=(num_block,1,1))

    print bytes_in[0]

"""
    i = 0
    length = len(bytes_in)

    #TODO: Remove the while loops and call the cuda_kernel
    while(i < length - 1):
        output[i:i+2] = np.bitwise_xor(decipher(32, bytes_in[i:i+2], ha), prev_decrypt)

        prev_decrypt = bytes_in[i:i+2]
        i += 2
    return output
"""