import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import hashlib
import collections
import numpy as np
import numpy.linalg as la
import precode
from pycuda.compiler import SourceModule

f = open("cuda_kernel.cu", 'r')
sm = pycuda.compiler.SourceModule(f.read(), options=['-lineinfo'])

# decrypt_bytes function which will call the cuda_kernel
def decrypt_bytes(bytes_in, key_list):

    # key_list -- list of NUM_KEY keys

    # get the function from the cuda_kernel
    func = sm.get_function("decipher")

    length = len(bytes_in)
    # output = []
    # decipher_output = []
    iv = np.array([1,2], dtype=np.uint32)
    length_ha = len(key_list) * len(bytes_in)/2
    ha = np.empty(length_ha, dtype=np.ndarray)

    i = 0
    j = 0
    # create a list of  ha and initialize output
    while(j < (len(key_list))):
        # create a long list with the same key for len(bytes_in)/2 so it could be possible to pop from ha for every thread call
        ha[i] = np.fromstring(hashlib.md5(key_list[j]).digest(), np.uint32)
        i = i + 1

        if(i % (len(bytes_in)/2) == 0):
            j = j + 1;

        # output.append(np.empty_like(bytes_in))

        # in each position of decipher output there will be an array with the decipher message
        # decipher_output.append(np.empty(lenght, dtype=np.uint32))

    # output -- containing information of every piece of message for all the keys
    output = np.empty(length*len(key_list), dtype=np.uint32)
    decipher_output = np.empty(length*len(key_list), dtype=np.uint32)
    prev_decrypt = iv

    # assign a number < 1024 for the number of threads for each block
    num_threads = 1024

    # each block will analyze one key
    num_blocks = ((length / 2) / num_threads) + 1 + len(key_list)

    # convert the array ha in a popable list of element
    ha_list = collections.deque(ha)
    func(np.int32(32), drv.In(bytes_in), drv.In(ha_list.popleft()), drv.InOut(decipher_output), np.int32(length), block=(num_threads,1,1), grid=(num_blocks,1,1))


    # now in decipher_output there are all the v0 and v1 couple

    # counter for each key and for the index of bytes_in
    x = 0
    i = 0
    while(i < length*len(key_list)):
        output[i:i+2] = np.bitwise_xor(decipher_output[i:i+2], prev_decrypt)
        prev_decrypt = bytes_in[x:x+2]
        i += 2
        x += 2
        if ( x == length):
            x = 0
            prev_decrypt = iv

    return output