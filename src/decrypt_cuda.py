import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule

f = open("try_cuda.cu", 'r')
sm = pycuda.compiler.SourceModule(f.read(), options=['-lineinfo'])

def multiply_them():
    # get function pointer
    func = sm.get_function("multiply_them")

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)
    dest = numpy.zeros_like(a)

    func(drv.Out(dest), drv.In(a), drv.In(b),block=(400,1,1))

    print dest-a*b

