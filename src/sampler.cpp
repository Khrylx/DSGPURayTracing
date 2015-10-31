#include "sampler.h"

namespace CMU462 {

// Uniform Sampler2D Implementation //

Vector2D UniformGridSampler2D::get_sample() const {

  // TODO:
  // Implement uniform 2D grid sampler

    
    
    return Vector2D(std::rand()/(double)RAND_MAX,std::rand()/(double)RAND_MAX);

}

// Uniform Hemisphere Sampler3D Implementation //

Vector3D UniformHemisphereSampler3D::get_sample() const {

  // TODO:
  // Implement uniform 3D hemisphere sampler
    double r1 = std::rand()/(double)RAND_MAX;
    double r2 = std::rand()/(double)RAND_MAX;
    double sin_theta = sqrt(1-r1*r1);
    double phi = 2*PI*r2;
    
  return Vector3D(sin_theta*cos(phi), sin_theta*sin(phi), r1);

}
    
    
    Vector3D CosineWeightedHemisphereSampler3D::get_sample() const{
        double r1 = std::rand()/(double)RAND_MAX;
        double r2 = std::rand()/(double)RAND_MAX;
        double theta = acos(1 - 2*r1) / 2;
        double phi = 2*PI*r2;
        double sin_theta = sin(theta);
        
        return Vector3D(sin_theta*cos(phi), sin_theta*sin(phi), cos(theta));
    }

    Vector3D CosineWeightedHemisphereSampler3D::get_sample(float* pdf) const {
        double r1 = std::rand()/(double)RAND_MAX;
        double r2 = std::rand()/(double)RAND_MAX;
        double theta = acos(1 - 2*r1) / 2;
        double phi = 2*PI*r2;
        double sin_theta = sin(theta);
        double cos_theta = cos(theta);
        *pdf = cos_theta / PI;
        
        return Vector3D(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
        
    }
} // namespace CMU462
