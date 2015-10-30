#include "environment_light.h"
#include <iostream>

namespace CMU462 { namespace StaticScene {

EnvironmentLight::EnvironmentLight(const HDRImageBuffer* envMap)
    : envMap(envMap) {
  // TODO: initialize things here as needed

}

Vector3D getUniformSphereSample3D()
{
    //Matrix3x3 sampleToWorld;
    //sampleToWorld[0] = Vector3D(1,  0,  0);
    //sampleToWorld[1] = Vector3D(0,  0, -1);
    //sampleToWorld[2] = Vector3D(0,  1,  0);
    
    double r1 = rand()/(double)RAND_MAX;
    double r2 = rand()/(double)RAND_MAX;
    double cos_theta = 1 - 2*r1;
    double sin_theta = sqrt(1-cos_theta*cos_theta);
    double phi = 2*PI*r2;
    
    //return sampleToWorld * Vector3D(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
    return Vector3D(sin_theta*cos(phi), cos_theta, -sin_theta*sin(phi));
}
    
Spectrum EnvironmentLight::sample_L(const Vector3D& p, Vector3D* wi,
                                    float* distToLight,
                                    float* pdf) const {
  // TODO: Implement
    
    *wi = getUniformSphereSample3D();
    *distToLight = INF_D;
    *pdf = 1.0 / (4.0 * M_PI);
    
    Ray r(p,*wi);
    return sample_dir(r);
}

Spectrum EnvironmentLight::sample_dir(const Ray& r) const {
  // TODO: Implement
    
    double theta = acos(r.d[1]);
    double sin_theta = sqrt(1 - r.d[1]*r.d[1]);
    double phi = acos(r.d[0] / sin_theta);
    if (r.d[2] > 0) {
        phi = 2*PI - phi;
    }
    double u = phi / (2*PI);
    double v = theta / PI;

    int w = envMap->w;
    int h = envMap->h;

    
    float tu = u * w - 0.5;
    float tv = v * h - 0.5;
    
    int su = (int)tu;
    int sv = (int)tv;
    
    float a,b;
    int px1, px2, py1, py2;
    if (tu < 0) {
        a = tu+1;
        px1 = w-1;
        px2 = 0;
    }
    else if (tu >= w-1){
        a = tu-w+1;
        px1 = w-1;
        px2 = 0;
    }
    else{
        a = tu-su;
        px1 = su;
        px2 = su+1;
    }
    
    if (tv < 0) {
        b = tv+1;
        py1 = h-1;
        py2 = 0;
    }
    else if (tv >= h-1){
        b = tv-h+1;
        py1 = h-1;
        py2 = 0;
    }
    else{
        b = tv-sv;
        py1 = sv;
        py2 = sv+1;
    }
    
    Spectrum z11 = envMap->data[px1 + w*py1];
    Spectrum z21 = envMap->data[px2 + w*py1];
    Spectrum z12 = envMap->data[px1 + w*py2];
    Spectrum z22 = envMap->data[px2 + w*py2];
    
    Spectrum zy1 = z11 * (1-a) + z21 * a;
    Spectrum zy2 = z12 * (1-a) + z22 * a;
    
    return zy1 * (1-b) + zy2 * b;
    
}

} // namespace StaticScene
} // namespace CMU462
