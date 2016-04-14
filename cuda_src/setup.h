#pragma once

#include "../src/pathtracer.h"
#include "../src/static_scene/sphere.h"
#include "../src/static_scene/triangle.h"
#include "../src/static_scene/light.h"

using namespace CMU462;
using namespace StaticScene;

struct GPUCamera{
    float widthDivDist;
    float heightDivDist;
    float c2w[9];
    float pos[3];
};

struct GPULight
{
    float radiance[3];
    float dirToLight[3];
    float position[3];
    float direction[3];
    float dim_x[3];
    float dim_y[3];
    float area;
    int type; // 0 - Directional Light, 1 - InfiniteHemisphereLight, 2 - PointLight, 3 - AreaLight
};

// Use structures for better data locality
//struct GPUPrimitives{
//    int* types;
//    int* bsdfs;
//    float* positions;
//    float* normals;
//};

struct GPUBSDF{
    int type;
    float albedo[3];
    float transmittance[3];
    float reflectance[3];
    float ior;
};

struct Parameters
{
    int screenW;
    int screenH;
    int max_ray_depth; ///< maximum allowed ray depth (applies to all rays)
    int ns_aa;         ///< number of camera rays in one pixel (along one axis)
    int ns_area_light; ///< number samples per area light source
    int lightNum;
    int primNum;
    
    int* types;
    int* bsdfIndexes;
    float* positions;
    float* normals;
    
    float* frameBuffer;
};

struct GPURay
{
    int depth;  ///< depth of the Ray
    
    float o[3];  ///< origin
    float d[3];  ///< direction
    float min_t; ///< treat the ray as a segment (ray "begin" at max_t)
    float max_t; ///< treat the ray as a segment (ray "ends" at max_t)
};

struct GPUIntersection {
    
    float t;    ///< time of intersection
    
    int pIndex;
    
    float n[3];  ///< normal at point of intersection
    
    int bsdfIndex; ///< BSDF of the surface at point of intersection
    
    // More to follow.
};

class CUDAPathTracer{
    
    int screenW, screenH;
    int primNum;
    
    int* gpu_types;    // size: N.    *** 1 for triangle, 0 for sphere
    int* gpu_bsdfIndexes; // size: N.  ***  index for bsdf
    float* gpu_positions; // size: 9 * N.  *** for triangle, 9 floats representing all 3 vertices;
                          // for sphere, first 3 floats represent origin, 4th float represent radius
    float* gpu_normals;  // size: 9 * N.  *** normals for triangle
    float* frameBuffer;
    
public:
    CUDAPathTracer(PathTracer* _pathTracer);
    ~CUDAPathTracer();

    void loadCamera();

    void loadPrimitives();
    
    void loadLights();

    void loadParameters();
    
    void createFrameBuffer();

    void toGPULight(SceneLight *light, GPULight *gpuLight);
    
    void init();
    
    void startRayTracing();
    
    void updateHostSampleBuffer();
    
private:
    PathTracer* pathTracer;
    
};