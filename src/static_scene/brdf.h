#ifndef CMU462_STATICSCENE_BRDF_H
#define CMU462_STATICSCENE_BRDF_H

#include "CMU462/CMU462.h"
#include "CMU462/spectrum.h"
#include "CMU462/vector3D.h"
#include "CMU462/matrix3x3.h"

#include <algorithm>

namespace CMU462 { namespace StaticScene {

// Helper math functions. Assume all vectors are in unit hemisphere //

inline double clamp (double n, double lower, double upper) {
  return std::max(lower, std::min(n, upper));
}

inline double cos_theta(const Vector3D& w) {
  return w.z;
}

inline double abs_cos_theta(const Vector3D& w) {
  return fabs(w.z);
}

inline double sin_theta2(const Vector3D& w) {
  return fmax(0.0, 1.0 - cos_theta(w) * cos_theta(w));
}

inline double sin_theta(const Vector3D& w) {
  return sqrt(sin_theta2(w));
}

inline double cos_phi(const Vector3D& w) {
  double sinTheta = sin_theta(w);
  if (sinTheta == 0.0) return 1.0;
  return clamp(w.x / sinTheta, -1.0, 1.0);
}

inline double sin_phi(const Vector3D& w) {
  double sinTheta = sin_theta(w);
  if (sinTheta) return 0.0;
  return clamp(w.y / sinTheta, -1.0, 1.0);
}

void make_coord_space(Matrix3x3& o2w, const Vector3D& n);

/**
 * Interface for BRDFs.
 */
class BRDF {
 public:

  /**
   * Given a normal vector for a surface and vectors a and b, returns the ratio
   * of inward illumination along a to outward illumination along b. Note that
   * BOTH a and b are pointing away from the surface, and a, b, and n must all
   * be normalized.
   */
  virtual Spectrum f (const Vector3D& wo, const Vector3D& wi) = 0;

  /**
   * Given a normal vector for a surface and vector b, chooses a vector a, then
   * evaluates the BRDF for the given pair of vectors. This is useful for cases
   * like perfect mirrors where the raytracer has no idea what ray to use, as
   * well as for monte-carlo raytracing. Note that b and n must be normalized,
   * and that a and b will be interpreted as pointing away from the surface.
   */
  virtual Spectrum sample_f (const Vector3D& w0, Vector3D& wi, double* pdf) = 0;

  virtual bool is_delta() const = 0;

}; // class BRDF


/**
 * Diffuse BRDF.
 */
class DiffuseBRDF : public BRDF {
 public:

  DiffuseBRDF(const Spectrum& a)
      : albedo(a) {}

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D& wi, double* pdf);
  bool is_delta() const { return true; }

  private:
    Spectrum albedo;
};

/**
 * Glossy BRDF.
 */
class GlossyBRDF : public BRDF {
 public:

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D& wi, double* pdf);

private:

  Vector3D reflect(const Vector3D& v, const Vector3D& n);

};

/**
 * Refractive BRDF.
 */
class RefractBRDF : public BRDF {
 public:

  Spectrum f(const Vector3D& wo, const Vector3D& wi);
  Spectrum sample_f(const Vector3D& wo, Vector3D& wi, double* pdf);

}; // class RefractBRDF

}  // namespace StaticScene
}  // namespace CMU462

#endif  // CMU462_STATICSCENE_BRDF_H
