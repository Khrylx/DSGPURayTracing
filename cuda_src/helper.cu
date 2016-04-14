#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#define Pi 3.1415926

__device__ inline float
power(float X,float Y)
{
    return pow(X,Y);
}

__device__ inline float
dist3D(const float *X, const float *Y)
{
    return sqrt((X[0]-Y[0])*(X[0]-Y[0])+(X[1]-Y[1])*(X[1]-Y[1])+(X[2]-Y[2])*(X[2]-Y[2]));
}

__device__ inline float
norm3D(const float *X)
{
    return sqrt(X[0]*X[0]+X[1]*X[1]+X[2]*X[2]);
}

__device__ inline void
normalize3D(float *X)
{
    double norm = sqrt(X[0]*X[0]+X[1]*X[1]+X[2]*X[2]);
    X[0] /= norm;
    X[1] /= norm;
    X[2] /= norm;
}

__device__ inline void
initVector3D(float x, float y, float z, float* S)
{
    S[0] = x;
    S[1] = y;
    S[2] = z;
}

__device__ inline void
readVector3D(float* src, float* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}

__device__ inline void
negVector3D(const float *X,float* S)
{
    S[0] = -X[0];
    S[1] = -X[1];
    S[2] = -X[2];
}

__device__ inline void
addVector3D(const float *X,float* S)
{
    S[0] += X[0];
    S[1] += X[1];
    S[2] += X[2];
}

__device__ inline void
addVector3D(const float *X, const float *Y, float *S) {
    S[0] = X[0] + Y[0];
    S[1] = X[1] + Y[1];
    S[2] = X[2] + Y[2];
}

__device__ inline void
subVector3D(const float *X, const float *Y, float *S) {
    S[0] = X[0] - Y[0];
    S[1] = X[1] - Y[1];
    S[2] = X[2] - Y[2];
}

__device__ inline void
VectorMulVectorT3D(const float *X,float* S)
{
    int i;
    int j;
    for (int k=0;k<9;k++)
    {
        i=k/3;
        j=k%3;
        S[k]+=X[i]*X[j];
    }
}

__device__ inline float
VectorDot3D(const float *X,const float *Y)
{
    return X[0]*Y[0]+X[1]*Y[1]+X[2]*Y[2];
}

__device__ inline void
VectorCross3D(const float *u, const float *v, float *s) {
    s[0] = u[1] * v[2] - u[2] * v[1];
    s[1] = u[2] * v[0] - u[0] * v[2];
    s[2] = u[0] * v[1] - u[1] * v[0];
}

inline __device__
float det3D(const float* X)
{
    return X[0]*X[4]*X[8]+X[3]*X[7]*X[2]+X[6]*X[1]*X[5]
    -X[2]*X[4]*X[6]-X[5]*X[7]*X[0]-X[8]*X[1]*X[3];
}

inline __device__
float trace3D(const float* X)
{
    return X[0]+X[4]+X[8];
}

inline __device__
void inverse3D(const float* X,float* Y)
{
    float a=det3D(X);
    Y[0]=(X[4]*X[8]-X[5]*X[7])/a;
    Y[1]=(X[2]*X[7]-X[1]*X[8])/a;
    Y[2]=(X[1]*X[5]-X[2]*X[4])/a;
    Y[3]=(X[5]*X[6]-X[3]*X[8])/a;
    Y[4]=(X[0]*X[8]-X[2]*X[6])/a;
    Y[5]=(X[2]*X[3]-X[0]*X[5])/a;
    Y[6]=(X[3]*X[7]-X[4]*X[6])/a;
    Y[7]=(X[1]*X[6]-X[0]*X[7])/a;
    Y[8]=(X[0]*X[4]-X[1]*X[3])/a;


}

inline __device__
void VectorTMulMatrix3D(float* X,float* S,float* Y)
{
    for (int i=0;i<3;i++)
    {
        Y[i]=X[0]*S[i]+X[1]*S[i+3]+X[2]*S[i+6];
    }

}

inline __device__
void MatrixMulVector3D(float* S,float* X,float* Y)
{
    for (int i=0;i<3;i++)
    {
        Y[i]=X[0]*S[3*i]+X[1]*S[3*i+1]+X[2]*S[3*i+2];
    }

}

inline __device__
void MatrixMulMatrix3D(float* X,float* Y,float* Z)
{
    int i,k;
    for (k = 0; k < 9; k++)
    {
        i=k/3;
        Z[k]=X[3*i]*Y[i]+X[3*i+1]*Y[i+3]+X[3*i+2]*Y[i+6];
    }
}

inline __device__
void MatrixAddMatrix3D(float* X,float* Y,float* Z)
{
    int k;
    for (k = 0; k < 9; k++)
    {
        Z[k]=X[k]+Y[k];
    }
}

inline __device__
void MatrixScale3D(float* X,float a)
{
    int k;
    for (k = 0; k < 9; k++)
    {
        X[k]=X[k]*a;
    }
}



inline __device__ void
gpuAdd(float *A, float *B, float *C)
{
    *C = *A + *B;
}
