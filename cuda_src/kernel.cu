#include <stdio.h>
#include <curand_kernel.h>
#include "setup.h"

#define MAX_NUM_LIGHT 20
#define MAX_NUM_BSDF 20

#define RUSSIAN_ROULETTE

#define INF_FLOAT 1e20
#define ESP_N 5e-3
#define EPS_K 1e-4

#define BLOCK_DIM 64

__constant__  GPUCamera const_camera;
__constant__  GPUBSDF const_bsdfs[MAX_NUM_BSDF];
__constant__  GPULight const_lights[MAX_NUM_LIGHT];
__constant__  Parameters const_params;

#include "helper.cu"
#include "light.cu"
#include "intersect.cu"
#include "bsdf.cu"


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
traceRay(curandState* s, GPURay* ray, bool includeLe, bool verbose)
{
    float3 L_out = make_float3(0.0, 0.0, 0.0);

    GPUIntersection isect;
    isect.t = INF_FLOAT;

    bool isIntersect = BVH_intersect(*ray, &isect);

    // bool isIntersect = false;
    // for(int i = 0; i < const_params.primNum; i++)
    // {
    //     isIntersect = intersect(i, *ray, &isect) || isIntersect;
    // }

    if(!isIntersect)
        return L_out;

    GPUBSDF& bsdf = const_bsdfs[isect.bsdfIndex];
    if(includeLe && bsdf.type == 4)
    {
        L_out = make_float3(bsdf.albedo[0], bsdf.albedo[1], bsdf.albedo[2]);
    }

    // GPUBSDF& bsdf = const_bsdfs[isect.bsdfIndex];
    //
    // switch(bsdf.type)
    // {
    //     case 0: case 4: L_out = make_float3(bsdf.albedo[0], bsdf.albedo[1], bsdf.albedo[2]); break;
    //     case 1: L_out = make_float3(0.0, 1.0, 0.0); break;
    //     case 3: L_out = make_float3(0.0, 0.0, 1.0); break;
    //     default: break;
    // }

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

    float dir_to_light[3];
    float dist_to_light;
    float pdf;
    float w_in[3];

    for(int i = 0; i < const_params.lightNum; i++)
    {
        float3 L = make_float3(0, 0, 0);
        int num_light_samples = const_lights[i].type == 0 ? 1 : const_params.ns_area_light;
        float scale = 1.0 / num_light_samples;

        for (int j=0; j<num_light_samples; j++) {

            float3 light_L = sample_L(i, hit_p, dir_to_light, &dist_to_light, &pdf, s);

            float eps = const_lights[i].type == 0 ? EPS_N : 0;

            GPURay sR;
            addScaledVector3D(hit_p, isect.n, eps, sR.o);
            addScaledVector3D(sR.o, dir_to_light, EPS_K, sR.o);
            readVector3D(dir_to_light, sR.d);
            sR.min_t = 0;
            sR.max_t = dist_to_light * 0.99;
            // if (verbose)
            //     printf("Before intersection.\n");

            // isIntersect = false;
            // for(int i = 0; i < const_params.primNum; i++)
            // {
            //     if(intersect(i, sR)){
            //         isIntersect = true;
            //         break;
            //     }
            // }
            // if (verbose)
            //     printf("After intersection.\n");

            isIntersect = BVH_intersect(sR);
            if(isIntersect) continue;


            MatrixMulVector3D(w2o, dir_to_light, w_in);
            normalize3D(w_in);

            float coeff = fmaxf(0.0, w_in[2]) / pdf;

            float3 f = BSDF_f(isect.bsdfIndex, w_out, w_in);

            L.x += coeff * light_L.x * f.x;
            L.y += coeff * light_L.y * f.y;
            L.z += coeff * light_L.z * f.z;
        }
        L_out.x += L.x * scale;
        L_out.y += L.y * scale;
        L_out.z += L.z * scale;
    }

    if(ray->depth >= const_params.max_ray_depth){
        return L_out;
    }

    float3 f = BSDF_sample_f(isect.bsdfIndex, w_out, w_in, &pdf, s);

#ifdef RUSSIAN_ROULETTE
    float terminateProbability = fmaxf(1 - illum(f), 0.f);
    if (curand_uniform(s) < terminateProbability){
        return L_out;
    }
#endif

    float cos_theta = fabs(w_in[2]);
    float v[3];
    MatrixMulVector3D(o2w, w_in, v);
    normalize3D(v);

    GPURay newR;
    addScaledVector3D(hit_p, v, EPS_K, newR.o);
    readVector3D(v, newR.d);
    newR.depth = ray->depth + 1;
    newR.min_t = 0;
    newR.max_t = INF_FLOAT;

#ifdef RUSSIAN_ROULETTE
    float coeff = cos_theta / pdf / (1 - terminateProbability);
#else
    float coeff = cos_theta / pdf;
#endif

    float3 indirL = traceRay(s, &newR, bsdf.type != 0 && bsdf.type != 4, verbose);

    L_out.x += coeff * indirL.x * f.x;
    L_out.y += coeff * indirL.y * f.y;
    L_out.z += coeff * indirL.z * f.z;

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

        float3 tmpSpec = traceRay(s, &ray, true, x == 500 && y == 300);
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

__device__ int globalPoolNextRay = 0;




__device__ float3
tracePixelPT(curandState* s, int x, int y, bool verbose)
{

    int w = const_params.screenW;
    int h = const_params.screenH;
    int ns_aa = const_params.ns_aa;

    float2 r = gridSampler(s);
    float px = (x + r.x) / (float)w;
    float py = (y + r.y) / (float)h;

    GPURay ray;
    generateRay(&ray, px, py);

    float3 spec = traceRay(s, &ray, true, verbose);

    return make_float3(spec.x / ns_aa, spec.y / ns_aa, spec.z / ns_aa);
}


__global__ void
traceScenePT(int xStart, int yStart, int width, int height)
{
    
    int globalPoolRayCount = height * width * const_params.ns_aa;
    __shared__ float3 spec[BLOCK_DIM];
    __shared__ int bIndex[BLOCK_DIM];
    __shared__ volatile int localPoolNextRay;

    while(true){

        if(threadIdx.x == 0){
            localPoolNextRay = atomicAdd(&globalPoolNextRay, BLOCK_DIM);
        }
        __syncthreads();

        int myRayIndex = localPoolNextRay + threadIdx.x;
        if (myRayIndex >= globalPoolRayCount)
            return;


        int index = myRayIndex / const_params.ns_aa;
        int x = index % width + xStart;
        int y = index / height + yStart;

        bIndex[threadIdx.x] = y * const_params.screenW + x;

        curandState s;
        curand_init((unsigned int)myRayIndex * (xStart * TILE_DIM + yStart), 0, 0, &s);

        spec[threadIdx.x] = tracePixelPT(&s, x, y, false);

        __syncthreads();

        if(threadIdx.x == 0 || (threadIdx.x > 0 && bIndex[threadIdx.x - 1] != bIndex[threadIdx.x])){
            
            for (int i = threadIdx.x + 1; i < BLOCK_DIM; ++i)
            {
                if (bIndex[i - 1] != bIndex[i]) break;
                    
                spec[threadIdx.x].x += spec[i].x;
                spec[threadIdx.x].y += spec[i].y;
                spec[threadIdx.x].z += spec[i].z;
            }



            atomicAdd(&const_params.frameBuffer[3 * bIndex[threadIdx.x]], spec[threadIdx.x].x);
            atomicAdd(&const_params.frameBuffer[3 * bIndex[threadIdx.x] + 1], spec[threadIdx.x].y);
            atomicAdd(&const_params.frameBuffer[3 * bIndex[threadIdx.x] + 2], spec[threadIdx.x].z);
        }

        
    }


}
