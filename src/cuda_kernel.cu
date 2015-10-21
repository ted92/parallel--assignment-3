__global__ void decipher(unsigned int v0, unsigned int v1, int sum)
{
    // declare shared variables inside the block
    __shared__ unsigned int v0 = v0;
    __shared__ unsigned int v1 = v1;
    __shared__ int sum = sum;

    // the number of the tread is not relevant

    v1 = (v1 - (((v0<<4 ^ v0>>5) + v0) ^ (sum + key[sum>>11 & 3]))) & mask;
    sum = (sum - delta) & mask;
    v0 = (v0 - (((v1<<4 ^ v1>>5) + v1) ^ (sum + key[sum & 3]))) & mask;

}