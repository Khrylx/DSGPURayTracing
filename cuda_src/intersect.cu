
__device__ inline bool bboxIntersect(const GPUBBox *bbox, GPURay& r, float* divR, float* invR, float& t0) {

    float x0 = bbox->min[0] * invR[0] - divR[0];
    float x1 = bbox->max[0] * invR[0] - divR[0];
    float y0 = bbox->min[1] * invR[1] - divR[1];
    float y1 = bbox->max[1] * invR[1] - divR[1];
    float z0 = bbox->min[2] * invR[2] - divR[2];
    float z1 = bbox->max[2] * invR[2] - divR[2];

    t0 = fmin_fmax(x0,x1,fmin_fmax(y0,y1,
                fmin_fmax(z0,z1,0)));
    float t1 = fmax_fmin(x0,x1,fmax_fmin(y0,y1,
                fmax_fmin(z0,z1,r.max_t)));

    return t0 <= t1;
}

// Woop triangle intersection test
__device__ inline bool triangleIntersectWoop(int primIndex, GPURay& r) {

    float4* woopPositions = const_params.woopPositions + 3 * primIndex;
    float4 v0 = woopPositions[0];
    float4 v1 = woopPositions[1];

    float Oz = v0.w - r.o[0]*v0.x - r.o[1]*v0.y - r.o[2]*v0.z;
    float invDz = 1.0f / (r.d[0]*v0.x + r.d[1]*v0.y + r.d[2]*v0.z);
    float t = Oz * invDz;

    if (t <= r.min_t)
        return false;

    float Ox = v1.w + r.o[0]*v1.x + r.o[1]*v1.y + r.o[2]*v1.z;
    float Dx = r.d[0]*v1.x + r.d[1]*v1.y + r.d[2]*v1.z;
    float u = Ox + t*Dx;

    if (u < 0.0f) return false;

    float4 v2 = woopPositions[2];
    float Oy = v2.w + r.o[0]*v2.x + r.o[1]*v2.y + r.o[2]*v2.z;
    float Dy = r.d[0]*v2.x + r.d[1]*v2.y + r.d[2]*v2.z;
    float v = Oy + t*Dy;

    if(v < 0.0f || u + v > 1.0f)
        return false;

    return true;
}


__device__ inline bool triangleIntersectWoop(int primIndex, GPURay& r, GPUIntersection *isect) {


    float* normals = const_params.normals + 9 * primIndex;
    float4* woopPositions = const_params.woopPositions + 3 * primIndex;
    float4 v0 = woopPositions[0];
    float4 v1 = woopPositions[1];

    float Oz = v0.w - r.o[0]*v0.x - r.o[1]*v0.y - r.o[2]*v0.z;
    float invDz = 1.0f / (r.d[0]*v0.x + r.d[1]*v0.y + r.d[2]*v0.z);
    float t = Oz * invDz;

    if (t <= r.min_t || t >= isect->t)
        return false;

    float Ox = v1.w + r.o[0]*v1.x + r.o[1]*v1.y + r.o[2]*v1.z;
    float Dx = r.d[0]*v1.x + r.d[1]*v1.y + r.d[2]*v1.z;
    float u = Ox + t*Dx;

    if (u < 0.0f) return false;

    float4 v2 = woopPositions[2];
    float Oy = v2.w + r.o[0]*v2.x + r.o[1]*v2.y + r.o[2]*v2.z;
    float Dy = r.d[0]*v2.x + r.d[1]*v2.y + r.d[2]*v2.z;
    float v = Oy + t*Dy;

    if(v < 0.0f || u + v > 1.0f)
        return false;

    isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->t = t;
    isect->pIndex = primIndex;

    float *n1 = normals;
    float *n2 = normals + 3;
    float *n3 = normals + 6;

    for (int i = 0; i < 3; ++i)
    {
        isect->n[i] = (1 - u - v) * n1[i] + u * n2[i] + v * n3[i];
    }
    if (VectorDot3D(r.d, isect->n) > 0)
    {
        negVector3D(isect->n, isect->n);
    }

    return true;
}



// primitive and normals are shift pointers to the primitive and normal we selected
__device__ inline bool triangleIntersect(int primIndex, GPURay& r) {

    float* primitive = const_params.positions + 9 * primIndex;

    float* v1 = primitive;
    float* e1 = primitive + 3;
    float* e2 = primitive + 6;

    //float e1[3], e2[3];
    float pvec[3], qvec[3];
    //subVector3D(v2, v1, e1);
    //subVector3D(v3, v1, e2);
    VectorCross3D(r.d, e2, pvec);

    float det = VectorDot3D(e1, pvec);
    if (det == 0) {
        return false;
    }

    float invDet = 1 / det;

    float tvec[3];
    subVector3D(r.o, v1, tvec);

    float u = VectorDot3D(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    VectorCross3D(tvec, e1, qvec);
    float v = VectorDot3D(r.d, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    
    float t = VectorDot3D(e2, qvec) * invDet;
    if (t <= r.min_t || t >= r.max_t) {
        return false;
    }

    return true;
}

// primitive and normals are shift pointers to the primitive and normal we selected
__device__ inline bool triangleIntersect(int primIndex, GPURay& r, GPUIntersection *isect) {

    float* primitive = const_params.positions + 9 * primIndex;
    float* normals = const_params.normals + 9 * primIndex;

    float* v1 = primitive;
    float* e1 = primitive + 3;
    float* e2 = primitive + 6;

    //float e1[3], e2[3];
    float pvec[3], qvec[3];
    //subVector3D(v2, v1, e1);
    //subVector3D(v3, v1, e2);
    VectorCross3D(r.d, e2, pvec);

    float det = VectorDot3D(e1, pvec);
    if (det == 0) {
        return false;
    }

    float invDet = 1 / det;

    float tvec[3];
    subVector3D(r.o, v1, tvec);

    float u = VectorDot3D(tvec, pvec) * invDet;
    if (u < 0 || u > 1) return false;

    VectorCross3D(tvec, e1, qvec);
    float v = VectorDot3D(r.d, qvec) * invDet;
    if (v < 0 || u + v > 1) return false;

    float t = VectorDot3D(e2, qvec) * invDet;    
    if (t <= r.min_t || t >= r.max_t || t >= isect->t) {
        return false;
    }

    r.max_t = t;

    isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->t = t;
    isect->pIndex = primIndex;

    float *n1 = normals;
    float *n2 = normals + 3;
    float *n3 = normals + 6;

    for (int i = 0; i < 3; ++i)
    {
        isect->n[i] = (1 - u - v) * n1[i] + u * n2[i] + v * n3[i];
    }
    if (VectorDot3D(r.d, isect->n) > 0)
    {
        negVector3D(isect->n, isect->n);
    }

    return true;
}

__device__ inline bool sphereTest(int primIndex, GPURay& ray, float& t1, float& t2) {
    float* primitive = const_params.positions + 9 * primIndex;
    float* o = primitive;
    float r = primitive[3];
    float r2 = r * r;

    float m[3];
    subVector3D(o, ray.o, m);
    float b = VectorDot3D(m, ray.d);
    float c = VectorDot3D(m, m) - r2;
    float delta = b * b - c;
    if (delta < 0) {
        return false;
    }

    t1 = b - sqrt(delta);
    t2 = b + sqrt(delta);

    if (t1 >= ray.max_t || t2 <= ray.min_t) {
        return false;
    }

    return true;
}

__device__ inline bool sphereIntersect(int primIndex, GPURay& r) {
    float tmp;
    return sphereTest(primIndex, r, tmp, tmp);
}

__device__ inline bool sphereIntersect(int primIndex, GPURay& r, GPUIntersection *isect) {
    float t1;
    float t2;
    bool res = sphereTest(primIndex, r, t1, t2);
    if (!res) {
        return false;
    }
    isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->pIndex = primIndex;

    float* primitive = const_params.positions + 9 * primIndex;
    float* o = primitive;
    float t = t1;
    if (t1 <= r.min_t) {
        t = t2;
    }
    float n[3];
    float tmp[3];
    for (int i = 0; i < 3; ++i)
    {
        tmp[i] = r.d[i] * t;
    }
    addVector3D(r.o, tmp);
    subVector3D(tmp, o, n);
    normalize3D(n);
    readVector3D(n, isect->n);
    isect->t = t;
    r.max_t = t;

    return true;
}

__device__ inline bool intersect(int primIndex, GPURay& r) {
    if (const_params.types[primIndex] == 0) {
        // sphere
        return sphereIntersect(primIndex, r);
    } else {
        // triangle
        return triangleIntersectWoop(primIndex, r);
    }
}

__device__ inline bool intersect(int primIndex, GPURay& r, GPUIntersection *isect) {
    if (const_params.types[primIndex] == 0) {
        // sphere
        return sphereIntersect(primIndex, r, isect);
    } else {
        // triangle
        return triangleIntersectWoop(primIndex, r, isect);
    }
}
