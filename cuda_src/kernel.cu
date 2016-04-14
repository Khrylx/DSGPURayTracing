#include <stdio.h>

#include "helper.cu"
#include "setup.h"

#define MAX_NUM_LIGHT 20
#define MAX_NUM_BSDF 20

__constant__  GPUCamera const_camera;
__constant__  GPUBSDF const_bsdfs[MAX_NUM_BSDF];
__constant__  GPULight const_lights[MAX_NUM_LIGHT];
__constant__  Parameters const_params;


__device__ void
generateRay(GPURay* ray, float x, float y)
{
    float sp[3];
    sp[0] = -(x-0.5) * const_camera.widthDivDist;
    sp[1] = -(y-0.5) * const_camera.heightDivDist;
    sp[2] = 1;
    float dir[3];
    dir[0] = -sp[0];
    dir[1] = -sp[1];
    dir[2] = -sp[2];
    float world_sp[3];
    MatrixMulVector3D(const_camera.c2w, sp, world_sp);
    
}

__device__ float3
tracePixel(int x, int y)
{
    float3 s;
    
    int w = const_params.screenW;
    int h = const_params.screenH;
    
    float px = x / (float)w;
    float py = y / (float)h;
    
    GPURay ray;
    generateRay(&ray, px, py);
}


__global__ void
traceScene()
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index >= const_params.screenW * const_params.screenH) {
        return;
    }
    
    int x = index % const_params.screenW;
    int y = index / const_params.screenW;
    
    tracePixel(x, y);
    
    const_params.frameBuffer[3 * index] = 1.0;
    const_params.frameBuffer[3 * index + 1] = 0.5;
    const_params.frameBuffer[3 * index + 2] = 0.5;
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





