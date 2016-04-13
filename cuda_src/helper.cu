#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#define Pi 3.1415926

extern "C" __device__ inline float
power(float X,float Y)
{
    return pow(X,Y);
}

extern "C" __device__ inline float
dist3D(const float *X, const float *Y)
{
    return sqrt((X[0]-Y[0])*(X[0]-Y[0])+(X[1]-Y[1])*(X[1]-Y[1])+(X[2]-Y[2])*(X[2]-Y[2]));
}

extern "C" __device__ inline float
norm3D(const float *X)
{
    return sqrt(X[0]*X[0]+X[1]*X[1]+X[2]*X[2]);
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
localization(float r,float h)
{
    float tmp=4*r/h;
    return exp(-tmp*tmp);
}

__device__ inline float
VectorDot3D(const float *X,const float *Y)
{
    return X[0]*Y[0]+X[1]*Y[1]+X[2]*Y[2];
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
void VectorLMulMatrix3D(float* X,float* S,float* Y)
{
    for (int i=0;i<3;i++)
    {
        Y[i]=X[0]*S[i]+X[1]*S[i+3]+X[2]*S[i+6];
    }
    
}

inline __device__
void VectorRMulMatrix3D(float* X,float* S,float* Y)
{
    for (int i=0;i<3;i++)
    {
        Y[i]=X[0]*S[3*i]+X[1]*S[3*i+1]+X[2]*S[3*i+2];
    }
    
}

inline __device__
void MatrixMulMatrix3D(float* X,float* Y,float* Z)
{
    int i,j,k;
    for (k = 0; k < 9; k++)
    {
        i=k/3;
        j=k%3;
        Z[k]=X[3*i]*Y[i]+X[3*i+1]*Y[i+3]+X[3*i+2]*Y[i+6];
    }
}

inline __device__
void MatrixAddMatrix3D(float* X,float* Y,float* Z)
{
    int i,j,k;
    for (k = 0; k < 9; k++)
    {
        Z[k]=X[k]+Y[k];
    }
}

inline __device__
void MatrixScale3D(float* X,float a)
{
    int i,j,k;
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
