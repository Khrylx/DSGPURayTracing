#include "environment_light.h"
#include <iostream>

namespace CMU462 { namespace StaticScene {

EnvironmentLight::EnvironmentLight(const HDRImageBuffer* envMap) : envMap(envMap)
{
  // TODO: initialize things here as needed
    w = envMap->w;
    h = envMap->h;
    
    pThetaPhi.resize(h,std::vector<float>(w));
    pTheta.resize(h,0);
    pPhiGivenTheta.resize(h,std::vector<float>(w));
    
    // Cal
    float C = 0;
    for (int y = 0; y < h; y++) {
        float theta = (y + 0.5) / h * PI;
        float sin_theta = sin(theta);

        for (int x = 0; x < w; x++) {
            pThetaPhi[y][x] = envMap->data[x + w*y].illum() * sin_theta;
            C += pThetaPhi[y][x];
        }
    }
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            pThetaPhi[y][x] /= C;
            pTheta[y] += pThetaPhi[y][x];
        }
        
        if (pTheta[y] != 0) {
            for (int x = 0; x < w; x++) {
                pPhiGivenTheta[y][x] = pThetaPhi[y][x] / pTheta[y];
            }
        }
    }
    
    for (int y = 0; y < h; y++) {
        if (y > 0) {
            pTheta[y] += pTheta[y-1];
        }
        for (int x = 0; x < w; x++) {
            if (x > 0) {
                pPhiGivenTheta[y][x] += pPhiGivenTheta[y][x-1];
            }
        }
    }

}

Vector3D getUniformSphereSample3D()
{
    //Matrix3x3 sampleToWorld;
    //sampleToWorld[0] = Vector3D(1,  0,  0);
    //sampleToWorld[1] = Vector3D(0,  0, -1);
    //sampleToWorld[2] = Vector3D(0,  1,  0);
    
    double r1 = std::rand()/(double)RAND_MAX;
    double r2 = std::rand()/(double)RAND_MAX;
    double cos_theta = 1 - 2*r1;
    double sin_theta = sqrt(1-cos_theta*cos_theta);
    double phi = 2*PI*r2;
    
    //return sampleToWorld * Vector3D(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
    return Vector3D(-sin_theta*sin(phi), cos_theta, sin_theta*cos(phi));
}
    
void EnvironmentLight::importanceSampling(Vector3D *wi, float *pdf) const
{
    float r1 = std::rand()/(float)RAND_MAX;
    float r2 = std::rand()/(float)RAND_MAX;
    float theta;
    float phi;
    
    float x,y;
    int t,q;
    
    // Calculate theta
    
    // Normalize r1 because pTheta.back() may not be 1 for precision reason.
    r1 *= pTheta.back();
    // Find the first element geq the random float.
    std::vector<float>::const_iterator itr = std::lower_bound(pTheta.begin(), pTheta.end(), r1);
    // calculate index
    t = itr - pTheta.begin();
    // handle case t = 0
    float prev = t > 0? *(itr-1) : 0;
    // calculate theta by interpolation
    y = t + (r1 - prev)/(*itr - prev);
    theta = std::min(y / h, 1.f) * PI;
    
    // Normalize r2
    r2 *= pPhiGivenTheta[t].back();
    // Calculate phi, almost the same as theta
    itr = std::lower_bound(pPhiGivenTheta[t].begin(), pPhiGivenTheta[t].end(), r2);
    q = itr - pPhiGivenTheta[t].begin();
    prev = q > 0? *(itr-1) : 0;
    x = q + (r2 - prev)/(*itr - prev);
    phi = std::min(x / w, 1.f) * 2*PI;
    
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    
    // calculate pixelwise pdf
    *pdf = pThetaPhi[t][q];
    
    // calculate ray-wise pdf , a pixel represent a solid angle subtending sin_theta * d_theta * d_phi area
    *pdf /= (sin_theta * (2*PI / w) * (PI / h));
    
    *wi = Vector3D(-sin_theta*sin(phi), cos_theta, sin_theta*cos(phi));
    
}

Spectrum EnvironmentLight::sample_L(const Vector3D& p, Vector3D* wi,
                                    float* distToLight,
                                    float* pdf) const {
  // TODO: Implement
    
    // Uniform Sampling
    //*wi = getUniformSphereSample3D();
    //*pdf = 1.0 / (4.0 * M_PI);
    
    //Importance Sampling
    importanceSampling(wi,pdf);
    
    *distToLight = INF_D;
    
    Ray r(p,*wi);
    return sample_dir(r);
}

Spectrum EnvironmentLight::sample_dir(const Ray& r) const {
  // TODO: Implement
    
    double theta = acos(r.d[1]);
    double sin_theta = sqrt(1 - r.d[1]*r.d[1]);
    double phi = sin_theta == 0 ? PI : acos(clamp(r.d[2] / sin_theta,-1.0,1.0));
    if (r.d[0] > 0) {
        phi = 2*PI - phi;
    }
    double u = phi / (2*PI);
    double v = theta / PI;
    
    float tu = u * w - 0.5;
    float tv = v * h - 0.5;
    
    int su = (int)tu;
    int sv = (int)tv;
    
    float a,b;
    int px1, px2, py1, py2;
    
    // Handling all corner cases by wrapping around
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
    
    //std::cout << r.d << std::endl;
    //std::cout << u <<"," << v << "," << px1 << "," << px2 << "," << py1 << "," << py2 << "," << std::endl;
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
