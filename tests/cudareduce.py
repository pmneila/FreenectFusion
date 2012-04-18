
import string
import numpy as np
from pycuda.compiler import SourceModule

add_vectors_source = """

__global__ void add_vectors($T* out, $T* in, size_t size)
{
    __shared__ $T sdata[$BLOCK_SIZE*$D];
    
    unsigned int thid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int gridSize = blockDim.x*gridDim.x;
    
    $T* current_sdata = &sdata[$D*thid];
    for(int j=0; j<$D; ++j)
        current_sdata[j] = 0;
    for(; i < size; i += gridSize)
        for(int j=0; j<$D; ++j)
            current_sdata[j] += in[$D*i + j];
    __syncthreads();
    
    for(int s=blockDim.x>>1; s>0; s>>=1)
    {
        if(thid < s)
            for(int j=0; j<$D; ++j)
                current_sdata[j] += sdata[(thid + s)*$D + j];
        __syncthreads();
    }
    
    if(thid == 0)
    {
        $T* current_out = &out[$D*blockIdx.x];
        for(int j=0; j<$D; ++j)
            current_out[j] = current_sdata[j];
    }
}
"""

BLOCK_SIZE = 128

def memoize(f):
    """A simple memoize function."""
    cache= {}
    def memf(*x):
        if x not in cache:
            cache[x] = f(*x)
        return cache[x]
    return memf

@memoize
def _get_add_vectors(typestr, dims):
    source = string.Template(add_vectors_source)
    source = source.substitute(T=typestr, D=dims, BLOCK_SIZE=BLOCK_SIZE)
    module = SourceModule(source)
    return module.get_function("add_vectors")

def add_vectors(v_gpu, size, dims):
    
    add_vectors = _get_add_vectors("float", dims)
    step = BLOCK_SIZE*32
    i, oldi = (size+step-1)//step, size
    while oldi>1:
        add_vectors(v_gpu, v_gpu, np.int32(oldi), block=(BLOCK_SIZE,1,1), grid=(i,1))
        i, oldi = (i+step-1)/step, i
    return v_gpu

if __name__ == '__main__':
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    
    a = gpuarray.zeros((640*480, 8), dtype=np.float32) + 1
    add_vectors(a, 640*480, 8)
    print a.get()[0]
