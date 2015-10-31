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

Vector3D getUniformSphereSample3D(float& theta, float& phi)
{
    //Matrix3x3 sampleToWorld;
    //sampleToWorld[0] = Vector3D(1,  0,  0);
    //sampleToWorld[1] = Vector3D(0,  0, -1);
    //sampleToWorld[2] = Vector3D(0,  1,  0);
    
    float r1 = std::rand()/(float)RAND_MAX;
    float r2 = std::rand()/(float)RAND_MAX;
    float cos_theta = 1 - 2*r1;
    float sin_theta = sqrt(1-cos_theta*cos_theta);
    theta = acos(1 - 2*r1);
    phi = 2*PI*r2;
    
    //return sampleToWorld * Vector3D(sin_theta*cos(phi), sin_theta*sin(phi), cos_theta);
    return Vector3D(sin_theta*cos(phi), cos_theta, -sin_theta*sin(phi));
}
    
void EnvironmentLight::importanceSampling(Vector3D *wi, float *pdf, float& theta, float& phi) const
{
    float r1 = std::rand()/(float)RAND_MAX;
    float r2 = std::rand()/(float)RAND_MAX;
    
    float x,y;
    int t,q;
    float a;
    float prev;
    
    // Calculate theta
    // Find the first element geq the random float
    std::vector<float>::const_iterator itr = std::lower_bound(pTheta.begin(), pTheta.end(), r1);
    if (itr == pTheta.end()) {
        t = h-1;
        y = h;
        theta = PI;
    }
    else{
        // calculate index
        t = itr - pTheta.begin();
        // handle case t = 0
        prev = t > 0? *(itr-1) : 0;
        // calculate theta by interpolation
        y = t + (r1 - prev)/(*itr - prev);
        theta = y / h * PI;
        // choose t to be the nearest integer theta
        if (round(y) > t && t < h-1) {
            t++;
        }
    }
    
    // Calculate phi, almost the same as theta
    itr = std::lower_bound(pPhiGivenTheta[t].begin(), pPhiGivenTheta[t].end(), r2);
    if (itr == pPhiGivenTheta[t].end()) {
        q = w-1;
        x = w;
        a = 1;
        phi = 2 * PI;
    }
    else{
        q = itr - pPhiGivenTheta[t].begin();
        prev = q > 0? *(itr-1) : 0;
        a = (r2 - prev)/(*itr - prev);
        x = q + a;
        phi = x / w * 2*PI;

    }
    
    prev = q > 0? pThetaPhi[t][q-1] : 0;
    
    float sin_theta = sin(theta);
    float cos_theta = cos(theta);
    
    // calculate pixelwise pdf
    *pdf = (a * prev + (1-a)*pThetaPhi[t][q]);
    
    // calculate ray-wise pdf , a pixel represent a solid angle subtending sin_theta * d_theta * d_phi area
    *pdf /= (sin_theta * (2*PI / w) * (PI / h));
    
    *wi = Vector3D(sin_theta*cos(phi), cos_theta, -sin_theta*sin(phi));
    
}

Spectrum EnvironmentLight::sample_L(const Vector3D& p, Vector3D* wi,
                                    float* distToLight,
                                    float* pdf) const {
  // TODO: Implement
    float theta,phi;
    // Uniform Sampling
    //*wi = getUniformSphereSample3D(theta,phi);
    //*pdf = 1.0 / (4.0 * M_PI);
    
    // Importance Sampling
    
    
    importanceSampling(wi,pdf,theta,phi);
    
    *distToLight = INF_D;
    
    return sample_dir(theta,phi);
}
    
    
// I change the parameter here, because these will save some unnecessary calculation.
Spectrum EnvironmentLight::sample_dir(float theta, float phi) const {
  // TODO: Implement
    
//    float theta = acos(r.d[1]);
//    float sin_theta = sin(theta);
//    float phi = sin_theta == 0 ? PI : acos(r.d[0] / sin_theta);
//    if (r.d[2] > 0) {
//        phi = 2*PI - phi;
//    }
    
    float u = phi / (2*PI);
    float v = theta / PI;
    
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
