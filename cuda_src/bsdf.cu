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

__device__ float3 RefractionBSDF_sample_f(GPUBSDF *bsdf, const float *wo, float *wi, float *pdf) {
	*pdf = 1;
	bool res = refract(wo, wi, bsdf->ior);
	if (!res) {
		return make_float3(0.0, 0.0, 0.0);
	}

	float ni = bsdf->ior;
	float no = 1;
	if (wo[2] < 0) {
		float tmp = ni;
		ni = no;
		no = tmp;
	}
	float ratio = no / ni;
	float coef = ratio * ratio / fmaxf(fabs(wi[2]), 1e-8);
	return make_float3(bsdf->transmittance[0] * coef, bsdf->transmittance[1] * coef, bsdf->transmittance[2] * coef);
}

__device__ float3 GlassBSDF_sample_f(GPUBSDF *bsdf, const float *wo, float *wi, float *pdf, curandState *s) {
	*pdf = 1;

	bool res = refract(wo, wi, bsdf->ior);
	if (!res) {
		float coef = 1.0 / fmaxf(fabs(wi[2]), 1e-8);
		return make_float3(bsdf->transmittance[0] * coef, bsdf->transmittance[1] * coef, bsdf->transmittance[2] * coef);
	}

	float ni = bsdf->ior;
	float no = 1;
	float cos_i = fabs(wi[2]);
	float cos_o = fabs(wo[2]);
	if (wo[2] < 0) {
		float tmp = ni;
		ni = no;
		no = tmp;
	}

	float r1 = (no*cos_i - ni*cos_o)/(no*cos_i + ni*cos_o);
    float r2 = (ni*cos_i - no*cos_o)/(ni*cos_i + no*cos_o);
    float Fr = 0.5*(r1*r1 + r2*r2);

    if (curand_uniform(s) <= Fr) {
    	reflect(wo, wi);
    	float coef = 1.0 / fmaxf(fabs(wi[2]), 1e-8);
    	return make_float3(bsdf->reflectance[0] * coef, bsdf->reflectance[1] * coef, bsdf->reflectance[2] * coef);
    } else {
    	float ratio = no / ni;
    	float coef = ratio * ratio / fmaxf(fabs(wi[2]), 1e-8);
    	return make_float3(bsdf->transmittance[0] * coef, bsdf->transmittance[1] * coef, bsdf->transmittance[2] * coef);
    }
}