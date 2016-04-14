#include <stdio.h>
#include <curand_kernel.h>

#include "helper.cu"
#include "setup.h"

#define MAX_NUM_LIGHT 20
#define MAX_NUM_BSDF 20

#define INF_FLOAT 1e20

__constant__  GPUCamera const_camera;
__constant__  GPUBSDF const_bsdfs[MAX_NUM_BSDF];
__constant__  GPULight const_lights[MAX_NUM_LIGHT];
__constant__  Parameters const_params;

__device__ float2 gridSampler(curandState *s) {
    float2 rt;
    rt.x = curand_uniform(s);
    rt.y = curand_uniform(s);
    return rt;
}

__device__ float3 uniformHemisphereSampler(curandState *s) {
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

__device__ void
generateRay(GPURay* ray, float x, float y)
{
    ray->depth = 0;
    ray->min_t = 0;
    ray->max_t = 1e10;

    float sp[3];
    float dir[3];

    initVector3D(-(x-0.5) * const_camera.widthDivDist, -(y-0.5) * const_camera.heightDivDist, 1, sp);
    negVector3D(sp, dir);
    MatrixMulVector3D(const_camera.c2w, sp, ray->o);
    addVector3D(const_camera.pos, ray->o);
    MatrixMulVector3D(const_camera.c2w, dir, ray->d);
    normalize3D(ray->d);
}

__device__ float3
tracePixel(curandState* s, int x, int y, bool verbose)
{
    float3 spec = make_float3(1.0, 0.0, 0.0);

    int w = const_params.screenW;
    int h = const_params.screenH;


    for (int i = 0; i < 5; i++)
    {
        float2 r = gridSampler(s);
        float px = (x + r.x) / (float)w;
        float py = (y + r.y) / (float)h;

        GPURay ray;
        generateRay(&ray, px, py);

        // if(verbose)
        // {
        //     printf("%f %f\n", r.x, r.y);
        //     printf("%f %f %f\n", ray.o[0], ray.o[1], ray.o[2]);
        //     printf("%f %f %f\n", ray.d[0], ray.d[1], ray.d[2]);
        // }
    }


    return spec;
}


__global__ void
traceScene()
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= const_params.screenW * const_params.screenH) {
        return;
    }

    curandState s;
    curand_init((unsigned int)index, 0, 0, &s);

    int x = index % const_params.screenW;
    int y = index / const_params.screenW;

    tracePixel(&s, x, y, x == 500 && y == 300);

    const_params.frameBuffer[3 * index] = 1.0;
    const_params.frameBuffer[3 * index + 1] = 0.5;
    const_params.frameBuffer[3 * index + 2] = 0.5;

    // initialize random sampler state
    // need to pass to further functions


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

// primitive and normals are shift pointers to the primitive and normal we selected
__device__ bool triangleIntersect(int primIndex, GPURay& r) {
    
    float* primitive = const_params.positions + 9 * primIndex;

    float* v1 = primitive;
    float* v2 = primitive + 3;
    float* v3 = primitive + 6;

    float e1[3], e2[3], s[3];
    subVector3D(v2, v1, e1);
    subVector3D(v3, v1, e2);
    subVector3D(r.o, v1, s);

    float tmp[3];
    VectorCross3D(e1, r.d, tmp);
    double f = VectorDot3D(tmp, e2);
    if (f == 0) {
        return false;
    }

    VectorCross3D(s, r.d, tmp);
    double u = VectorDot3D(tmp, e2) / f;
    VectorCross3D(e1, r.d, tmp);
    double v = VectorDot3D(tmp, s) / f;
    VectorCross3D(e1, s, tmp);
    double t = - VectorDot3D(tmp, e2) / f;

    if (u >= 0 && v >= 0 && u+v <= 1 && t > r.min_t && t < r.max_t) {
        return true;
    }

    return false;
}

// primitive and normals are shift pointers to the primitive and normal we selected
__device__ bool triangleIntersect(int primIndex, GPURay& r, GPUIntersection *isect) {
    
    float* primitive = const_params.positions + 9 * primIndex;
    float* normals = const_params.normals + 9 * primIndex;

    float* v1 = primitive;
    float* v2 = primitive + 3;
    float* v3 = primitive + 6;

    float e1[3], e2[3], s[3];
    subVector3D(v2, v1, e1);
    subVector3D(v3, v1, e2);
    subVector3D(r.o, v1, s);

    float tmp[3];
    VectorCross3D(e1, r.d, tmp);
    double f = VectorDot3D(tmp, e2);
    if (f == 0) {
        return false;
    }

    VectorCross3D(s, r.d, tmp);
    double u = VectorDot3D(tmp, e2) / f;
    VectorCross3D(e1, r.d, tmp);
    double v = VectorDot3D(tmp, s) / f;
    VectorCross3D(e1, s, tmp);
    double t = - VectorDot3D(tmp, e2) / f;

    if (!(u >= 0 && v >= 0 && u+v <= 1 && t > r.min_t && t < r.max_t && t < isect->t)) {
        return false;
    }

    r.max_t = t;

    isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->t = t;
    isect->pIndex = primIndex;

    float *n1 = normals;
    float *n2 = normals + 3;
    float *n3 = normals + 6;

    float n[3];
    for (int i = 0; i < 3; ++i)
    {
        n[i] = (1 - u - v) * n1[i] + u * n2[i] + v * n3[i];
    }
    if (VectorDot3D(r.d, n) > 0)
    {
        negVector3D(n, n);
    }
    readVector3D(n, isect->n);

    return true;
}

__device__ bool sphereTest(int primIndex, GPURay& ray, double& t1, double& t2) {
    float* primitive = const_params.positions + 9 * primIndex;
    float* o = primitive;
    float r = primitive[3];
    float r2 = r * r;

    float m[3];
    subVector3D(o, ray.o, m);
    double b = VectorDot3D(m, ray.d);
    double c = VectorDot3D(m, m) - r2;
    double delta = b * b - c;
    if (delta < 0) {
        return false;
    }

    t1 = b - sqrt(delta);
    t2 = b + sqrt(delta);

    if (t1 >= ray.max_t || t2 <= ray.min_t) {
        return false;
    }

    return true;
}

__device__ bool sphereIntersect(int primIndex, GPURay& r) {
    double tmp;
    return sphereTest(primIndex, r, tmp, tmp);
}

__device__ bool sphereIntersect(int primIndex, GPURay& r, GPUIntersection *isect) {
    double t1;
    double t2;
    bool res = sphereTest(primIndex, r, t1, t2);
    if (!res) {
        return false;
    }
    isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->pIndex = primIndex;

    float* primitive = const_params.positions + 9 * primIndex;
    float* o = primitive;
    double t = t1;
    if (t1 <= r.min_t) {
        t = t2;
    }
    float n[3];
    float tmp[3];
    for (int i = 0; i < 3; ++i)
    {
        tmp[i] = r.d[i] * t;
    }
    addVector3D(r.o, tmp);
    subVector3D(tmp, o, n);
    normalize3D(n);
    readVector3D(n, isect->n);
    isect->t = t;
    r.max_t = t;

    return true;
}

__device__ bool intersect(int primIndex, GPURay& r) {
    if (const_params.types[primIndex] == 0) {
        // sphere
        return sphereIntersect(primIndex, r);
    } else {
        // triangle
        return triangleIntersect(primIndex, r);
    }
}

__device__ bool intersect(int primIndex, GPURay& r, GPUIntersection *isect) {
    if (const_params.types[primIndex] == 0) {
        // sphere
        return sphereIntersect(primIndex, r, isect);
    } else {
        // triangle
        return triangleIntersect(primIndex, r, isect);
    }
}

__device__ float3 DirectionalLight_sample_L(GPULight *light, float *p, float *wi, float *distToLight, float *pdf) {
    readVector3D(light->dirToLight, wi);
    *distToLight = INF_FLOAT;
    *pdf = 1.0;

    float3 spec;
    spec.x = light->radiance[0];
    spec.y = light->radiance[1];
    spec.z = light->radiance[2];
    return spec;
}

__device__ float3 InfiniteHemisphereLight_sample_L(GPULight *light, float *p, float *wi, float *distToLight, float *pdf, curandState *s) {
    float3 dirTmp = uniformHemisphereSampler(s);
    float dir[3];
    readVector3D(dirTmp, dir);

    MatrixMulVector3D(light->sampleToWorld, dir, wi);
    *distToLight = INF_FLOAT;
    *pdf = 1.0 / (2.0 * M_PI);

    float3 spec;
    spec.x = light->radiance[0];
    spec.y = light->radiance[1];
    spec.z = light->radiance[2];
    return spec;
}

__device__ float3 PointLight_sample_L(GPULight *light, float *p, float *wi, float *distToLight, float *pdf, curandState *s) {
    float d[3];
    float d_unit[3];
    subVector3D(light->position, p, d);
    readVector3D(d, d_unit);
    normalize3D(d_unit);
    readVector3D(d_unit, wi);
    *distToLight = norm3D(d);
    *pdf = 1.0;

    float3 spec;
    spec.x = light->radiance[0];
    spec.y = light->radiance[1];
    spec.z = light->radiance[2];
    return spec;
}

__device__ float3 sample_L(int lightIndex, float *p, float *wi, float *distToLight, float *pdf, curandState *s) {
    GPULight *light = const_lights + lightIndex;
    float3 spec;
    switch(light->type) {
        case 0:
            spec = DirectionalLight_sample_L(light, p, wi, distToLight, pdf);
            break;
        case 1:
            spec = InfiniteHemisphereLight_sample_L(light, p, wi, distToLight, pdf, s);
            break;
        case 2:
            spec = PointLight_sample_L(light, p, wi, distToLight, pdf, s);
        break;
        case 3:
        break;
        default:
        break;
    }
    return spec;
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
