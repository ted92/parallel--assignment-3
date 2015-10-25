__global__ void decipher(unsigned int num_rounds, unsigned int *input_data, unsigned int *key, unsigned int *output, unsigned int length)
{
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < length - 1){
        // current thread with 1D

        unsigned int v0 = input_data[thread_id];
        unsigned int v1 = input_data[thread_id+1];
        unsigned long delta = 0x9e3779b9L;
        unsigned long mask = 0xffffffffL;
        unsigned long sum = (delta*num_rounds) & mask;

        unsigned int i;
        for (i = 0; i < num_rounds; i++){
            v1 = (v1 - (((v0<<4 ^ v0>>5) + v0) ^ (sum + key[sum>>11 & 3]))) & mask;
            sum = (sum - delta) & mask;
            v0 = (v0 - (((v1<<4 ^ v1>>5) + v1) ^ (sum + key[sum & 3]))) & mask;
        }

        output[thread_id] = v0;
        output[thread_id+1] = v1;
    }

}