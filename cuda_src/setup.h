#include "../src/pathtracer.h"
#include "../src/static_scene/sphere.h"
#include "../src/static_scene/triangle.h"

using namespace CMU462;
using namespace StaticScene;

struct GPUCamera{
    float widthDivDist;
    float heightDivDist;
    float c2w[9];
    float pos[3];
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


class CUDAPathTracer{
    
    int* gpu_types;
    int* gpu_bsdfs;
    float* gpu_positions;
    float* gpu_normals;
    
    // just for release memory in GPUPrimitives
    
public:
    CUDAPathTracer(PathTracer* _pathTracer);
    ~CUDAPathTracer();

    void loadCamera();
    void loadPrimitives();
    
    
    void init();
    
    
private:
    PathTracer* pathTracer;
    
};