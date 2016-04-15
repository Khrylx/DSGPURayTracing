__device__ float3 BSDF_f(int bsdfIndex, float* wo, float* wi)
{
    GPUBSDF* bsdf = const_bsdfs + bsdfIndex;

    if (bsdf->type == 0){
        return make_float3(bsdf->albedo[0] / PI, bsdf->albedo[1] / PI, bsdf->albedo[2] / PI);
    }

    return make_float3(0.0, 0.0, 0.0);
}
