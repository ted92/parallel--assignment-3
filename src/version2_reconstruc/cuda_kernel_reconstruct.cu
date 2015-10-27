__global__ void reconstruct_secret(unsigned int *secret, unsigned int *result, unsigned int length_s, unsigned int length_r)
{
    // current thread with 1D
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < length_s){
        result[(thread_id >> 8) % length_r] = thread_id & mask;
    }

}