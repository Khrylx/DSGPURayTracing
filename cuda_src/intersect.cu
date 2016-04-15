__device__ inline bool bboxIntersect(GPUBBox *bbox, GPURay& r, double& t0, double& t1) {
    for (int i = 0; i < 3; ++i) {
        if (r.d[i] != 0.0) {
            double tx1 = (bbox->min[i] - r.o[i]) / r.d[i];
            double tx2 = (bbox->max[i] - r.o[i]) / r.d[i];

            t0 = fmaxf(t0, fminf(tx1, tx2));
            t1 = fminf(t1, fmaxf(tx1, tx2));
        }
    }

    return t0 <= t1;
}

// primitive and normals are shift pointers to the primitive and normal we selected
__device__ inline bool triangleIntersect(int primIndex, GPURay& r) {

    float* primitive = const_params.positions + 9 * primIndex;

    float* v1 = primitive;
    float* v2 = primitive + 3;
    float* v3 = primitive + 6;

    float e1[3], e2[3], s[3];
    subVector3D(v2, v1, e1);
    subVector3D(v3, v1, e2);
    subVector3D(r.o, v1, s);

    float tmp[3];
    VectorCross3D(e1, r.d, tmp);
    double f = VectorDot3D(tmp, e2);
    if (f == 0) {
        return false;
    }

    VectorCross3D(s, r.d, tmp);
    double u = VectorDot3D(tmp, e2) / f;
    VectorCross3D(e1, r.d, tmp);
    double v = VectorDot3D(tmp, s) / f;
    VectorCross3D(e1, s, tmp);
    double t = - VectorDot3D(tmp, e2) / f;

    if (u >= 0 && v >= 0 && u+v <= 1 && t > r.min_t && t < r.max_t) {
        return true;
    }

    return false;
}

// primitive and normals are shift pointers to the primitive and normal we selected
__device__ inline bool triangleIntersect(int primIndex, GPURay& r, GPUIntersection *isect) {

    float* primitive = const_params.positions + 9 * primIndex;
    float* normals = const_params.normals + 9 * primIndex;

    float* v1 = primitive;
    float* v2 = primitive + 3;
    float* v3 = primitive + 6;

    float e1[3], e2[3], s[3];
    subVector3D(v2, v1, e1);
    subVector3D(v3, v1, e2);
    subVector3D(r.o, v1, s);

    float tmp[3];
    VectorCross3D(e1, r.d, tmp);
    double f = VectorDot3D(tmp, e2);
    if (f == 0) {
        return false;
    }

    VectorCross3D(s, r.d, tmp);
    double u = VectorDot3D(tmp, e2) / f;
    VectorCross3D(e1, r.d, tmp);
    double v = VectorDot3D(tmp, s) / f;
    VectorCross3D(e1, s, tmp);
    double t = - VectorDot3D(tmp, e2) / f;

    if (!(u >= 0 && v >= 0 && u+v <= 1 && t > r.min_t && t < r.max_t && t < isect->t)) {
        return false;
    }

    r.max_t = t;

    isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->t = t;
    isect->pIndex = primIndex;

    float *n1 = normals;
    float *n2 = normals + 3;
    float *n3 = normals + 6;

    float n[3];
    for (int i = 0; i < 3; ++i)
    {
        n[i] = (1 - u - v) * n1[i] + u * n2[i] + v * n3[i];
    }
    if (VectorDot3D(r.d, n) > 0)
    {
        negVector3D(n, n);
    }
    readVector3D(n, isect->n);

    return true;
}

__device__ inline bool sphereTest(int primIndex, GPURay& ray, double& t1, double& t2) {
    float* primitive = const_params.positions + 9 * primIndex;
    float* o = primitive;
    float r = primitive[3];
    float r2 = r * r;

    float m[3];
    subVector3D(o, ray.o, m);
    double b = VectorDot3D(m, ray.d);
    double c = VectorDot3D(m, m) - r2;
    double delta = b * b - c;
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
    double tmp;
    return sphereTest(primIndex, r, tmp, tmp);
}

__device__ inline bool sphereIntersect(int primIndex, GPURay& r, GPUIntersection *isect) {
    double t1;
    double t2;
    bool res = sphereTest(primIndex, r, t1, t2);
    if (!res) {
        return false;
    }
    isect->bsdfIndex = const_params.bsdfIndexes[primIndex];
    isect->pIndex = primIndex;

    float* primitive = const_params.positions + 9 * primIndex;
    float* o = primitive;
    double t = t1;
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
        return triangleIntersect(primIndex, r);
    }
}

__device__ inline bool intersect(int primIndex, GPURay& r, GPUIntersection *isect) {
    if (const_params.types[primIndex] == 0) {
        // sphere
        return sphereIntersect(primIndex, r, isect);
    } else {
        // triangle
        return triangleIntersect(primIndex, r, isect);
    }
}
