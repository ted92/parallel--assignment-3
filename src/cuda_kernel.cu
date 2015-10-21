__global__ void decipher(unsigned int *input_data, int sum, unsigned int *key, int delta, int mask)
{
    // declare shared variables inside the block
    __shared__ unsigned int v0_s;
    __shared__ unsigned int v1_s;
    __shared__ int sum_s;

    unsigned int v0 = input_data[0];
    unsigned int v1 = input_data[1];
    // the number of the tread is not relevant

    v1_s = (v1 - (((v0<<4 ^ v0>>5) + v0) ^ (sum + key[sum>>11 & 3]))) & mask;
    sum_s = (sum - delta) & mask;
    v0_s = (v0 - (((v1<<4 ^ v1>>5) + v1) ^ (sum + key[sum & 3]))) & mask;

    input_data[0] = v0;
    input_data[1] = v1;

}