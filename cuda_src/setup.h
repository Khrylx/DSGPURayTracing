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
};

class CUDAPathTracer{
    
    GPUCamera* camera;
    
public:
    CUDAPathTracer(PathTracer* _pathTracer);
    ~CUDAPathTracer();

    void loadCamera();

    void loadLights();

    void toGPULight(SceneLight *light, GPULight *gpuLight);
    
    void init();
    
    
private:
    PathTracer* pathTracer;
    
};