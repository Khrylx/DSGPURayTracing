#include <stdio.h>
#include <curand_kernel.h>
#include "setup.h"

#define MAX_NUM_LIGHT 20
#define MAX_NUM_BSDF 20

#define INF_FLOAT 1e20

__constant__  GPUCamera const_camera;
__constant__  GPUBSDF const_bsdfs[MAX_NUM_BSDF];
__constant__  GPULight const_lights[MAX_NUM_LIGHT];
__constant__  Parameters const_params;

#include "helper.cu"
#include "light.cu"
#include "intersect.cu"


__device__ void
generateRay(GPURay* ray, float x, float y)
{
    ray->depth = 0;
    ray->min_t = 0;
    ray->max_t = INF_FLOAT;

    float sp[3];
    float dir[3];

    readVector3D(-(x-0.5) * const_camera.widthDivDist, -(y-0.5) * const_camera.heightDivDist, 1, sp);
    negVector3D(sp, dir);
    MatrixMulVector3D(const_camera.c2w, sp, ray->o);
    addVector3D(const_camera.pos, ray->o);
    MatrixMulVector3D(const_camera.c2w, dir, ray->d);
    normalize3D(ray->d);
}

__device__ float3
traceRay(curandState* s, GPURay* ray)
{
    float3 L_out = make_float3(0.0, 0.0, 0.0);

    GPUIntersection isect;
    isect.t = INF_FLOAT;

    bool isIntersect = false;
    for(int i = 0; i < const_params.primNum; i++)
    {
        isIntersect = intersect(i, *ray, &isect) || isIntersect;
    }

    if(!isIntersect)
        return L_out;

    GPUBSDF& bsdf = const_bsdfs[isect.bsdfIndex];

    switch(bsdf.type)
    {
        case 0: case 4: L_out = make_float3(bsdf.albedo[0], bsdf.albedo[1], bsdf.albedo[2]); break;
        case 1: L_out = make_float3(0.0, 1.0, 0.0); break;
        case 3: L_out = make_float3(0.0, 0.0, 1.0); break;
        default: break;
    }

    float hit_p[3];
    addScaledVector3D(ray->o, ray->d, isect.t, hit_p);

    float o2w[9];
    float w2o[9];
    make_coord_space(isect.n, o2w);
    MatrixTranspose3D(o2w, w2o);

    float tmpVec[3];
    float w_out[3];
    subVector3D(ray->o, hit_p, tmpVec);
    MatrixMulVector3D(w2o, tmpVec, w_out);
    normalize3D(w_out);


    return L_out;

}

__device__ float3
tracePixel(curandState* s, int x, int y, bool verbose)
{
    float3 spec = make_float3(0.0, 0.0, 0.0);

    int w = const_params.screenW;
    int h = const_params.screenH;
    int ns_aa = const_params.ns_aa;

    for (int i = 0; i < ns_aa; i++)
    {
        float2 r = gridSampler(s);
        float px = (x + r.x) / (float)w;
        float py = (y + r.y) / (float)h;

        GPURay ray;
        generateRay(&ray, px, py);

        float3 tmpSpec = traceRay(s, &ray);
        spec.x += tmpSpec.x;
        spec.y += tmpSpec.y;
        spec.z += tmpSpec.z;

    }


    return make_float3(spec.x / ns_aa, spec.y / ns_aa, spec.z / ns_aa);
}


__global__ void
traceScene(int xStart, int yStart, int width, int height)
{
    int tIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int x = xStart + tIndex % width;
    int y = yStart + tIndex / width;
    int index = x + y * const_params.screenW;

    if (tIndex >= width * height || index > const_params.screenW * const_params.screenH) {
        return;
    }


    curandState s;
    curand_init((unsigned int)index, 0, 0, &s);

    float3 spec = tracePixel(&s, x, y, x == 500 && y == 300);

    const_params.frameBuffer[3 * index] = spec.x;
    const_params.frameBuffer[3 * index + 1] = spec.y;
    const_params.frameBuffer[3 * index + 2] = spec.z;

    // initialize random sampler state
    // need to pass to further functions


}
