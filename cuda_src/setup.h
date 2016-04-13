#include "../src/pathtracer.h"
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
    int type;
};

struct Parameters
{
    size_t screenW;
    size_t screenH;
    size_t max_ray_depth; ///< maximum allowed ray depth (applies to all rays)
    size_t ns_aa;         ///< number of camera rays in one pixel (along one axis)
    size_t ns_area_light; ///< number samples per area light source
    size_t lightNum;
};

class CUDAPathTracer{
    
    GPUCamera* camera;
    GPULight* gpu_lights;
    Parameters *parms;
    
public:
    CUDAPathTracer(PathTracer* _pathTracer);
    ~CUDAPathTracer();

    void loadCamera();

    void loadLights();

    void loadParameters();

    void toGPULight(SceneLight *light, GPULight *gpuLight);
    
    void init();
    
    
private:
    PathTracer* pathTracer;
    
};