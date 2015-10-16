#include "sampler.h"

namespace CMU462 {

// Uniform Sampler2D Implementation //

Vector2D UniformGridSampler2D::get_sample() const {

  // TODO:
  // Implement uniform 2D grid sampler

    
    
    
  return Vector2D(rand()/(double)RAND_MAX,rand()/(double)RAND_MAX);

}

// Uniform Hemisphere Sampler3D Implementation //

Vector3D UniformHemisphereSampler3D::get_sample() const {

  // TODO:
  // Implement uniform 3D hemisphere sampler
    double r1 = rand()/(double)RAND_MAX;
    double r2 = rand()/(double)RAND_MAX;
    double s = sqrt(1-r1*r1);
    double theta = 2*PI*r2;
    
  return -Vector3D(s*cos(theta), s*sin(theta), r1);

}


} // namespace CMU462
