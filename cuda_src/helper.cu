#include <stdio.h>
#include <math.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#define Pi 3.1415926

__device__ float2 gridSampler(curandState *s) {
    float2 rt;
    rt.x = curand_uniform(s);
    rt.y = curand_uniform(s);
    return rt;
}

__device__ float3 UniformHemisphereSampler(curandState *s) {
    float2 tmp = gridSampler(s);
    double r1 = tmp.x;
    double r2 = tmp.y;

    double sin_theta = sqrt(1 - r1 * r1);
    double phi = 2 * PI * r2;

    float3 rt;
    rt.x = sin_theta * cos(phi);
    rt.y = sin_theta * sin(phi);
    rt.z = r1;
    return rt;
}

__device__ float3 CosineWeightedHemisphereSampler(float *pdf, curandState *s) {
    float2 tmp = gridSampler(s);
    float r1 = tmp.x;
    float r2 = tmp.y;
    float theta = acos(1 - 2 * r1) / 2;
    float phi = 2 * PI * r2;
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    *pdf = cos_theta / PI;

    float3 rt;
    rt.x = sin_theta*cos(phi);
    rt.y = sin_theta*sin(phi);
    rt.z = cos_theta;
    return rt;
}

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
scaleVector3D(float* X, float a)
{
    X[0] *= a;
    X[1] *= a;
    X[2] *= a;
}

__device__ inline void
scaleVector3D(float* X, float a, float *S)
{
    S[0] = a * X[0];
    S[1] = a * X[1];
    S[2] = a * X[2];
}

// S = X + a * Y
__device__ inline void
addScaledVector3D(float* X, float* Y, float a, float* S)
{
    S[0] = X[0] + Y[0] * a;
    S[1] = X[1] + Y[1] * a;
    S[2] = X[2] + Y[2] * a;
}

__device__ inline void
readVector3D(float x, float y, float z, float *dst) {
    dst[0] = x;
    dst[1] = y;
    dst[2] = z;
}

__device__ inline void
readVector3D(const float* src, float *dst) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
}

__device__ inline void
readVector3D(float3 src, float *dst) {
    dst[0] = src.x;
    dst[1] = src.y;
    dst[2] = src.z;
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

inline __device__
void MatrixTranspose3D(float* X,float* S)
{
    for(int k = 0; k < 9; k++)
    {
        S[(k % 3) * 3 + k / 3] = X[k];
    }
}

inline __device__
void make_coord_space(const float* n, float* o2w) {

    float z[3];
    float h[3];
    float x[3];
    float y[3];

    readVector3D(n, z);
    readVector3D(z, h);

    if (fabs(h[0]) <= fabs(h[1]) && fabs(h[0]) <= fabs(h[2])) h[0] = 1.0;
    else if (fabs(h[1]) <= fabs(h[0]) && fabs(h[1]) <= fabs(h[2])) h[1] = 1.0;
    else h[2] = 1.0;

    normalize3D(z);
    VectorCross3D(h, z, y);
    normalize3D(y);
    VectorCross3D(z, y, x);
    normalize3D(x);

    o2w[0] = x[0]; o2w[1] = y[0]; o2w[2] = z[0];
    o2w[3] = x[1]; o2w[4] = y[1]; o2w[5] = z[1];
    o2w[6] = x[2]; o2w[7] = y[2]; o2w[8] = z[2];
}

inline __device__ void
gpuAdd(float *A, float *B, float *C)
{
    *C = *A + *B;
}

__global__ void
vectorAdd(float *A, float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        gpuAdd(A + i, B + i, C + i);
        //C[i] = A[i] + B[i];
    }
}


__global__ void
printInfo()
{
    GPUBSDF* bsdfs = const_bsdfs;
    GPUCamera camera = const_camera;

    for (int i = 0; i < const_params.lightNum; i++)
    {
        printf("ligth: %d\n", const_lights[i].type);
    }

    for (int i = 0; i < 8; i++) {
        if (bsdfs[i].type == 0) {
            printf("0: %lf %lf %lf\n", bsdfs[i].albedo[0], bsdfs[i].albedo[1], bsdfs[i].albedo[2] );
        }
        else if (bsdfs[i].type == 1) {
            printf("1: %lf %lf %lf\n", bsdfs[i].reflectance[0], bsdfs[i].reflectance[1], bsdfs[i].reflectance[2] );
        }
        else if (bsdfs[i].type == 2) {
            //cout << "2" << endl;
        }
        else if (bsdfs[i].type == 3) {
            printf("3: %lf %lf %lf\n", bsdfs[i].reflectance[0], bsdfs[i].reflectance[1], bsdfs[i].reflectance[2] );
            printf("3: %lf %lf %lf\n", bsdfs[i].transmittance[0], bsdfs[i].transmittance[1], bsdfs[i].transmittance[2] );
        }
        else {
            printf("4: %lf %lf %lf\n", bsdfs[i].albedo[0], bsdfs[i].albedo[1], bsdfs[i].albedo[2] );
        }
    }


    printf("%lf %lf %lf\n", camera.pos[0], camera.pos[1], camera.pos[2] );


    float* positions = const_params.positions;
    float* normals = const_params.normals;

    printf("+++++++++++++++++++++++\n");
    for (int i = 0; i < const_params.primNum; i++) {
        printf("%d %d %d\n\n",const_params.types[i] ,const_params.bsdfIndexes[i], const_bsdfs[const_params.bsdfIndexes[i]].type);

        printf("%lf %lf %lf\n", positions[9 * i], positions[9 * i + 1], positions[9 * i + 2] );
        printf("%lf %lf %lf\n", positions[9 * i + 3], positions[9 * i + 4], positions[9 * i + 5] );
        printf("%lf %lf %lf\n", positions[9 * i + 6], positions[9 * i + 7], positions[9 * i + 8] );
        printf("=======================\n");
        printf("%lf %lf %lf\n", normals[9 * i], normals[9 * i + 1], normals[9 * i + 2] );
        printf("%lf %lf %lf\n", normals[9 * i + 3], normals[9 * i + 4], normals[9 * i + 5] );
        printf("%lf %lf %lf\n", normals[9 * i + 6], normals[9 * i + 7], normals[9 * i + 8] );
        printf("+++++++++++++++++++++++\n\n");
    }

}
