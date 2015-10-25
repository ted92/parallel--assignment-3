__global__ void decipher(unsigned int *input_data)
{
    // declare shared variables inside the block
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x


    input_data[0] = thread_id;
    return input_data;

}