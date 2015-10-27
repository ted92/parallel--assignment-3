import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import hashlib
import numpy as np
import numpy.linalg as la
import precode
from pycuda.compiler import SourceModule

# decrypt_bytes function which will call the cuda_kernel
def decrypt_bytes(bytes_in, key):
    f = open("cuda_kernel.cu", 'r')
    sm = pycuda.compiler.SourceModule(f.read(), options=['-lineinfo'])

    # get the function from the cuda_kernel
    func = sm.get_function("decipher")

    # each thread will have a cur_guess
    iv = np.array([1,2], dtype=np.uint32)
    ha = np.fromstring(hashlib.md5(key).digest(), np.uint32)

    output = np.empty_like(bytes_in)
    prev_decrypt = iv
    length = len(bytes_in)

    # assign a number < 1024 for the number of threads for each block
    num_threads = 256

    # number of blocks is the total length divided by 2, divided for the number of threads per block and + 1 in case of rest
    # /2 because in decipher each part of the message is processed in pair
    num_blocks = ((length / 2) / num_threads) + 1

    # call the function decipher and generate the output [v1,v2] nump array
    decipher_output = np.empty(length, dtype=np.uint32)

    func(np.int32(32), drv.In(bytes_in), drv.In(ha), drv.InOut(decipher_output), np.int32(length), block=(num_threads,1,1), grid=(num_blocks,1,1))

    # now in decipher_output there are all the v0 and v1 couple
    i = 0;
    while(i < length - 1):
        output[i:i+2] = np.bitwise_xor(decipher_output[i:i+2], prev_decrypt)
        prev_decrypt = bytes_in[i:i+2]
        i += 2
    return output

def reconstruct_secret(secret):
    f = open("cuda_kernel_reconstruct.cu", 'r')
    sm = pycuda.compiler.SourceModule(f.read(), options=['-lineinfo'])

    func = sm.get_function("reconstruc_secret")

    # result type: numpy.ndarray. each element is numpy.uint8
    result = np.empty_like(secret.astype(np.uint8))
    #mask type: long
    mask = 0xffL

    num_threads = 256
    num_blocks = (len(secret) / num_threads) + 1

    length_secret = len(secret)
    length_result = len(result)
    # call the reconstruct function in the kernel
    func(drv.InOut(secret), drv.InOut(result), np.int32(length_secret), np.int32(length_result), block=(num_threads,1,1), grid=(num_blocks,1,1))

    return result

