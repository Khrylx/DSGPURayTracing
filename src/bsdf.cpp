#include "bsdf.h"

#include <iostream>
#include <algorithm>
#include <utility>

using std::min;
using std::max;
using std::swap;

namespace CMU462 {

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

// Diffuse BSDF //

Spectrum DiffuseBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return albedo * (1.0 / PI);
}

Spectrum DiffuseBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
    // Generate random directions
    
    UniformHemisphereSampler3D sampler;
    *wi = sampler.get_sample();
    *pdf = 0.5/PI;
    
  return albedo * (1.0 / PI);
}

// Mirror BSDF //

Spectrum MirrorBSDF::f(const Vector3D& wo, const Vector3D& wi) {
    double eps = 1e-3;
    
    if (fabs(wo[2] - wi[2]) < eps && fabs(wo[0] + wi[0]) < eps && fabs(wo[1] + wi[1]) < eps ) {
        return reflectance * (1/std::max(wi[2],1e-8));
    }
    
    return Spectrum();
}

Spectrum MirrorBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {

  // TODO: 
  // Implement MirrorBSDF

    reflect(wo, wi);
    *pdf = 1;
    
    return reflectance * (1/std::max(wo[2],1e-8));
}

// Glossy BSDF //

/*
Spectrum GlossyBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum GlossyBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *pdf = 1.0f;
  return reflect(wo, wi, reflectance);
}
*/

// Refraction BSDF //

Spectrum RefractionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum RefractionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {

  // TODO: 
  // Implement RefractionBSDF 

    *pdf = 1;
    
    // Get the initial refract direction.
    bool res = refract(wo, wi, ior);
    if (!res) {
        return Spectrum();
    }
    
    double ni = ior;
    double no = 1;
    if (wo[2] < 0) {
        swap(ni,no);
    }

    double ratio = no/ni;
    return transmittance * ratio*ratio * (1/std::max(std::fabs((*wi)[2]),1e-8));
}

// Glass BSDF //

Spectrum GlassBSDF::f(const Vector3D& wo, const Vector3D& wi) {
    
    return Spectrum();
}

Spectrum GlassBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {

  // TODO: 
  // Compute Fresnel coefficient and either reflect or refract based on it.
    
    *pdf = 1;

    // Get the initial refract direction.
    bool res = refract(wo, wi, ior);
    if (!res) {
        return transmittance * (1/std::max(std::fabs((*wi)[2]),1e-8));
    }
    
    
    // Calculating Fr
    double ni = ior;
    double no = 1;
    double cos_i = std::fabs((*wi)[2]);
    double cos_o = std::fabs(wo[2]);
    if (wo[2] < 0) {
        swap(ni,no);
    }
    
    double r1 = (no*cos_i - ni*cos_o)/(no*cos_i + ni*cos_o);
    double r2 = (ni*cos_i - no*cos_o)/(ni*cos_i + no*cos_o);
    double Fr = 0.5*(r1*r1 + r2*r2);

    if (rand() / (double)RAND_MAX <= Fr) {    // If we choose reflection
        reflect(wo, wi);
        // Here we don't need to multiply Fr because we already using randomized strategy to achieve it.
        return reflectance * (1/std::max(std::fabs((*wi)[2]),1e-8));
    }
    else{                       // If we choose refraction
        double ratio = no/ni;
        // Here we don't need to multiply (1-Fr) because we already using randomized strategy to achieve it.
        return transmittance * ratio*ratio * (1/std::max(std::fabs((*wi)[2]),1e-8));
    }
    
}

void BSDF::reflect(const Vector3D& wo, Vector3D* wi) {

  // TODO:
  // Implement reflection of wo about normal (0,0,1) and store result in wi.
    *wi = Vector3D(-wo[0],-wo[1],wo[2]);
}

bool BSDF::refract(const Vector3D& wo, Vector3D* wi, float ior) {

  // TODO:
  // Use Snell's Law to refract wo surface and store result ray in wi.
  // Return false if refraction does not occur due to total internal reflection
  // and true otherwise. When dot(wo,n) is positive, then wo corresponds to a 
  // ray entering the surface through vacuum.  

    int sign = 1;
    float ratio = ior;
    if (wo[2] > 0) {
        sign = -1;
        ratio = 1/ratio;
    }
    
    float cos2_wi = 1 - ratio*ratio*(1 - wo[2]*wo[2]);
    if (cos2_wi < 0) {
        *wi = Vector3D(-wo[0],-wo[1],wo[2]);
        return false;
    }

    *wi = Vector3D(-wo[0]*ratio,-wo[1]*ratio,sign * sqrt(cos2_wi)).unit();
    
    return true;
}

// Emission BSDF //

Spectrum EmissionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return radiance * (1.0 / PI);
}

Spectrum EmissionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *wi  = sampler.get_sample(pdf);
  return radiance * (1.0 / PI);
}

} // namespace CMU462
