__device__ float3 BSDF_f(int bsdfIndex, float* wo, float* wi)
{
    GPUBSDF* bsdf = const_bsdfs + bsdfIndex;

    if (bsdf->type == 0){
        return make_float3(bsdf->albedo[0] / PI, bsdf->albedo[1] / PI, bsdf->albedo[2] / PI);
    }

    return make_float3(0.0, 0.0, 0.0);
}

__device__ float3 DiffuseBSDF_sample_f(GPUBSDF *bsdf, const float *wo, float *wi, float *pdf, curandState *s) {
	float3 tmp = CosineWeightedHemisphereSampler(pdf, s);
	readVector3D(tmp, wi);
	return make_float3(bsdf->albedo[0] / PI, bsdf->albedo[1] / PI, bsdf->albedo[2] / PI);
}

__device__ void reflect(const float *wo, float *wi) {
	wi[0] = -wo[0];
	wi[1] = -wo[1];
	wi[2] = wo[2];
}

__device__ float3 MirrorBSDF_sample_f(GPUBSDF *bsdf, const float *wo, float *wi, float *pdf) {
	reflect(wo, wi);
	*pdf = 1.0;
	return make_float3(bsdf->reflectance[0] / fmaxf(wo[2], 1e-8), bsdf->reflectance[1] / fmaxf(wo[2], 1e-8), bsdf->reflectance[2] / fmaxf(wo[2], 1e-8));
}

__device__ float3 EmissionBSDF_sample_f(GPUBSDF *bsdf, const float *wo, float *wi, float *pdf, curandState *s) {
	float3 tmp = CosineWeightedHemisphereSampler(pdf, s);
	readVector3D(tmp, wi);
	return make_float3(0.0, 0.0, 0.0);
}

__device__ bool refract(const float *wo, float *wi, float ior) {
	int sign = 1;
	float ratio = ior;
	if (wo[2] > 0) {
		sign = -1;
		ratio = 1 / ratio;
	}

	float cos2_wi = 1 - ratio * ratio * (1 - wo[2] * wo[2]);
	if (cos2_wi < 0) {
        // *wi = Vector3D(-wo[0],-wo[1],wo[2]);
        wi[0] = -wo[0];
        wi[1] = -wo[1];
        wi[2] = wo[2];
        return false;
    }

    wi[0] = -wo[0] * ratio;
    wi[1] = -wo[1] * ratio;
    wi[2] = sign * sqrt(cos2_wi);
    normalize3D(wi);
    return true;
}