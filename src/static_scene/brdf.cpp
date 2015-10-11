#include "brdf.h"

#include <iostream>

namespace CMU462 {
namespace StaticScene {

void make_coord_space(Matrix3x3& o2w, const Vector3D& n) {

    Vector3D z = Vector3D(n.x, n.y, n.z);
    Vector3D h = z;
    if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z)) h.x = 1.0;
    else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z)) h.y = 1.0;
    else h.z = 1.0;

    z.normalize();
    Vector3D y = cross(h, z);
    y.normalize();
    Vector3D x = cross(z, y);
    x.normalize();

    o2w[0] = x;
    o2w[1] = y;
    o2w[2] = z;
}

// Diffuse BRDF //

Spectrum DiffuseBRDF::f(const Vector3D& wo, const Vector3D& wi) {
  return albedo * (1.0 / PI);
}

Spectrum DiffuseBRDF::sample_f(const Vector3D& wo, Vector3D& owi, double* pdf) {
  return Spectrum();
}


// Glossy BRDF //

Spectrum GlossyBRDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum GlossyBRDF::sample_f(const Vector3D& wo, Vector3D& owi, double* pdf) {
  return Spectrum();
}


// Refraction BRDF //

Spectrum RefractBRDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum RefractBRDF::sample_f(const Vector3D& wo, Vector3D& owi, double* pdf) {
  return Spectrum();
}



} // namespace StaticScene
} // namespace CMU462
