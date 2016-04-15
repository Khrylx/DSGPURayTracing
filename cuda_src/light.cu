__device__ float3 uniformHemisphereSampler(curandState *s) {
    float2 tmp = gridSampler(s);
    double r1 = tmp.x;
    double r2 = tmp.y;

    double sin_theta = sqrt(1 - r1 * r1);
    double phi = 2 * PI * r2;

    float3 rt;
    rt.x = sin_theta * cos(phi);
    rt.y = sin_theta * sin(phi);
    rt.z = r1;
    return rt;
}

__device__ float3 DirectionalLight_sample_L(GPULight *light, float *p, float *wi, float *distToLight, float *pdf) {
    readVector3D(light->dirToLight, wi);
    *distToLight = INF_FLOAT;
    *pdf = 1.0;

    float3 spec;
    spec.x = light->radiance[0];
    spec.y = light->radiance[1];
    spec.z = light->radiance[2];
    return spec;
}

__device__ float3 InfiniteHemisphereLight_sample_L(GPULight *light, float *p, float *wi, float *distToLight, float *pdf, curandState *s) {
    float3 dirTmp = uniformHemisphereSampler(s);
    float dir[3];
    readVector3D(dirTmp, dir);

    MatrixMulVector3D(light->sampleToWorld, dir, wi);
    *distToLight = INF_FLOAT;
    *pdf = 1.0 / (2.0 * M_PI);

    float3 spec;
    spec.x = light->radiance[0];
    spec.y = light->radiance[1];
    spec.z = light->radiance[2];
    return spec;
}

__device__ float3 PointLight_sample_L(GPULight *light, float *p, float *wi, float *distToLight, float *pdf, curandState *s) {
    float d[3];
    float d_unit[3];
    subVector3D(light->position, p, d);
    readVector3D(d, d_unit);
    normalize3D(d_unit);
    readVector3D(d_unit, wi);
    *distToLight = norm3D(d);
    *pdf = 1.0;

    float3 spec;
    spec.x = light->radiance[0];
    spec.y = light->radiance[1];
    spec.z = light->radiance[2];
    return spec;
}

__device__ float3 AreaLight_sample_L(GPULight *light, float *p, float *wi, float *distToLight, float *pdf, curandState *s) {
    float2 sample = gridSampler(s);
    float d[3];
    for (int i = 0; i < 3; ++i) {
        d[i] = light->position[i] + sample.x * light->dim_x[i] + sample.y * light->dim_y[i] - p[i];
    }
    float cosTheta = VectorDot3D(d, light->direction);
    float sqDist = VectorDot3D(d, d); // norm square
    float dist = sqrt(sqDist);
    for (int i = 0; i < 3; ++i) {
        wi[i] = d[i] / dist;
    }
    *distToLight = dist;
    *pdf = sqDist / (light->area * fabs(cosTheta));

    float3 spec;
    if (cosTheta < 0) {
        spec.x = light->radiance[0];
        spec.y = light->radiance[1];
        spec.z = light->radiance[2];
    }
    return spec;
}

__device__ float3 sample_L(int lightIndex, float *p, float *wi, float *distToLight, float *pdf, curandState *s) {
    GPULight *light = const_lights + lightIndex;
    float3 spec;
    switch(light->type) {
        case 0:
            spec = DirectionalLight_sample_L(light, p, wi, distToLight, pdf);
            break;
        case 1:
            spec = InfiniteHemisphereLight_sample_L(light, p, wi, distToLight, pdf, s);
            break;
        case 2:
            spec = PointLight_sample_L(light, p, wi, distToLight, pdf, s);
        break;
        case 3:
            spec = AreaLight_sample_L(light, p, wi, distToLight, pdf, s);
        break;
        default:
        break;
    }
    return spec;
}
