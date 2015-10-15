#include "sampler.h"

namespace CMU462 {

// Uniform Sampler2D Implementation //

Vector2D UniformGridSampler2D::get_sample() const {

  // TODO:
  // Implement uniform 2D grid sampler

  return Vector2D(0.5,0.5);

}

// Uniform Hemisphere Sampler3D Implementation //

Vector3D UniformHemisphereSampler3D::get_sample() const {

  // TODO:
  // Implement uniform 3D hemisphere sampler

  return Vector3D(0, 0, 1);

}


} // namespace CMU462
