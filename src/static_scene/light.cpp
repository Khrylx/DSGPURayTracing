#include "light.h"

#include <iostream>

#include "../sampler.h"

namespace CMU462 { namespace StaticScene {

DirectionalLight::DirectionalLight(const Spectrum& rad,
                                   const Vector3D& lightDir)
    : radiance(rad) {
  dirToLight = -lightDir.unit();
    dirToLight.normalize();
}

Spectrum DirectionalLight::sample_L(const Vector3D& p, Vector3D* wi,
                                    double* distToLight, double* pdf) const {
  *wi = dirToLight;
  *distToLight = INF_D;
  *pdf = 1.0;
  return radiance;
}

InfiniteHemisphereLight::InfiniteHemisphereLight(const Spectrum& rad)
    : radiance(rad) {
  sampleToWorld[0] = Vector3D(1, 0, 0);
  sampleToWorld[1] = Vector3D(0, 0, -1);
  sampleToWorld[2] = Vector3D(0, 1, 0);
}

Spectrum InfiniteHemisphereLight::sample_L(const Vector3D& p, Vector3D* wi,
                                           double* distToLight,
                                           double* pdf) const {
  Vector3D dir = sampler.get_sample();

  *wi = sampleToWorld* dir;
  *distToLight = INF_D;
  *pdf = 1.0 / (2.0 * M_PI);
  return radiance;
}

} // namespace StaticScene
} // namespace CMU462
