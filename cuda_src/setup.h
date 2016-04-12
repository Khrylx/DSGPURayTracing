#include "../src/pathtracer.h"

using namespace CMU462;

struct GPUCamera{
    float widthDivDist;
    float heightDivDist;
    float c2w[9];
    float pos[3];
};



class CUDAPathTracer{
    
    GPUCamera* camera;
    
public:
    CUDAPathTracer(PathTracer* _pathTracer);
    ~CUDAPathTracer();

    void loadCamera();
    
    
    void init();
    
    
private:
    PathTracer* pathTracer;
    
};