inline __device__ void
gpuAdd(float *A, float *B, float *C)
{
    *C = *A + *B;
}
