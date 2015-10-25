import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import hashlib
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
    num_threads = 256

    # number of blocks is the total length divided for the number of threads per block and + 1 in case of rest
    num_blocks = (len(bytes_in) / num_threads) + 1

    # call the function decipher and generate the output [v1,v2] nump array
    decipher_output = np.empty_like(bytes_in)

    func(np.int32(32), drv.In(bytes_in), drv.In(ha), drv.InOut(decipher_output), block=(num_threads,1,1), grid=(num_blocks,1,1))

    # now in decipher_output there are all the v0 and v1 couple

    i = 0;
    while(i < length - 1):
        output[i:i+2] = np.bitwise_xor(decipher_output[i:i+2], prev_decrypt)
        prev_decrypt = bytes_in[i:i+2]
        i += 2
    return output

