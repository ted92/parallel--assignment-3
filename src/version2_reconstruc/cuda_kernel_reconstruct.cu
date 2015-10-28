__global__ void reconstruct_secret(unsigned int *result, unsigned int *secrets, unsigned int length_s, unsigned int length_one_secret)
{
    // current thread with 1D
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long mask = 0xffL;
    if(thread_id < length_s){
        unsigned int size = sizeof(secret);
        unsigned int i = 0;
        for(i = 0; i < length_one_secret; i++){
            result[(i >> 8) % sizeof(result)] = i & mask;
        }
    }
}