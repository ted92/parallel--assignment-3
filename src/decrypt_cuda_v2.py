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
def decrypt_bytes(bytes_in, key_list):
    # get the function from the cuda_kernel
    func = sm.get_function("decipher")

    length = len(bytes_in)
    # output = []
    # decipher_output = []
    iv = np.array([1,2], dtype=np.uint32)

    # create a list of  ha and initialize output
    for i in range(len(key_list)):
        # create a long list with the same key for len(bytes_in)/2 so it could be possible to pop from ha for every thread call
        for j in range(len(bytes_in)/2):
            ha[j+i] = np.fromstring(hashlib.md5(key_list[i]).digest(), np.uint32)

        # output.append(np.empty_like(bytes_in))

        # in each position of decipher output there will be an array with the decipher message
        # decipher_output.append(np.empty(lenght, dtype=np.uint32))

    output = np.empty(length*len(key_list), dtype=np.uint32)
    decipher_output = np.empty(length*len(key_list), dtype=np.uint32)
    prev_decrypt = iv

    # assign a number < 1024 for the number of threads for each block
    num_threads = 1024

    # each block will analyze one key
    num_blocks = ((length / 2) / num_threads) + 1 + len(key_list)

    func(np.int32(32), drv.In(bytes_in), drv.In(ha.popleft()), drv.InOut(decipher_output), np.int32(length), block=(num_threads,1,1), grid=(num_blocks,1,1))

    #TODO: MANAGE OUTPUT
    # now in decipher_output there are all the v0 and v1 couple

    j = 0
    i = 0
    while(j < (len(key_list))):
        output[i:i+2] = np.bitwise_xor(decipher_output[i:i+2], prev_decrypt)
        prev_decrypt = bytes_in[i:i+2]
        i += 2
        if (i == length-2):
            i = 0
            ++j
            prev_decrypt = iv

    return output