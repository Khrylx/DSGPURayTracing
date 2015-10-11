#ifndef CMU462_STATICSCENE_LIGHT_H
#define CMU462_STATICSCENE_LIGHT_H

#include "CMU462/vector3d.h"
#include "CMU462/matrix3x3.h"
#include "CMU462/spectrum.h"
#include "../sampler.h"

namespace CMU462 { namespace StaticScene {

class Light {
 public:
  virtual Spectrum sample_L(const Vector3D& p, Vector3D* wi,
                            double* distToLight, double* pdf) const = 0;
  virtual bool is_delta_light() const = 0;
};

class DirectionalLight : public Light {
 public:
  DirectionalLight(const Spectrum& rad, const Vector3D& lightDir);
  Spectrum sample_L(const Vector3D& p, Vector3D* wi, double* distToLight,
                    double* pdf) const;
  bool is_delta_light() const { return true; }

 private:
  Spectrum radiance;
  Vector3D dirToLight;
};

class InfiniteHemisphereLight : public Light {
 public:
  InfiniteHemisphereLight(const Spectrum& rad);
  Spectrum sample_L(const Vector3D& p, Vector3D* wi, double* distToLight,
                    double* pdf) const;
  bool is_delta_light() const { return false; }

 private:
  Spectrum radiance;
  UniformHemisphereSampler3D sampler;
  Matrix3x3 sampleToWorld;
};

} // namespace StaticScene
} // namespace CMU462

#endif  // CMU462_STATICSCENE_BRDF_H
