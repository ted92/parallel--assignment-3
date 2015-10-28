__global__ void reconstruct_secret(unsigned int *result, unsigned int *secrets, unsigned int length_s)
{
    // current thread with 1D
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long mask = 0xffL;
    if(thread_id < length_s){
        //take only the [thread_id] secret in secrets list
        unsigned int * secret = secrets[thread_id];
        unsigned int size = sizeof(secret);
        unsigned int i = 0;
        for(i = 0; i < size; i++){
            result[(element >> 8) % len(result)] = element & mask;
        }
    }
}