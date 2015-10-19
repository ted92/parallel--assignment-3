import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.compiler
import numpy as np

# Generate source module
f = open("cuda_kernel.cu", 'r')

sm = pycuda.compiler.SourceModule(f.read(), options=['-lineinfo'])

def guess_password(max_length, in_data, known_part):
    func = sm.get_function("guess_password")