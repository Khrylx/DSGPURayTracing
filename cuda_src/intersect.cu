//#define SPECULATIVE


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


    float3* normals = const_params.normals + 3 * primIndex;
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

    float3 n1 = normals[0];
    float3 n2 = normals[1];
    float3 n3 = normals[2];

    isect->n[0] = (1 - u - v) * n1.x + u * n2.x + v * n3.x;
    isect->n[1] = (1 - u - v) * n1.y + u * n2.y + v * n3.y;
    isect->n[2] = (1 - u - v) * n1.z + u * n2.z + v * n3.z;

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
    float3* normals = const_params.normals + 3 * primIndex;

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

    float3 n1 = normals[0];
    float3 n2 = normals[1];
    float3 n3 = normals[2];

    isect->n[0] = (1 - u - v) * n1.x + u * n2.x + v * n3.x;
    isect->n[1] = (1 - u - v) * n1.y + u * n2.y + v * n3.y;
    isect->n[2] = (1 - u - v) * n1.z + u * n2.z + v * n3.z;
    
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

// intersect with BVHNode
// __device__ bool node_intersect(const GPUBVHNode *node, GPURay &ray, GPUIntersection *i) {
//     if (node->left == NULL && node->right == NULL) {
//         bool isIntersect = false;
//         for (int j = 0; j < node->range; j++) {
//             int primIndex = const_params.BVHPrimMap[node->start + j];
//             bool res = intersect(primIndex, ray, i);
//             // bool res = false;
//             isIntersect = isIntersect || res;
//         }
//         return isIntersect;
//     } else if (node->left == NULL) {
//         return node_intersect(node->right, ray, i);
//     } else if (node->right == NULL) {
//         return node_intersect(node->left, ray, i);
//     } else {
//         float tminl = -INF_FLOAT;
//         float tminr = -INF_FLOAT;

//         GPURay nray = ray;
//         float eps[3] = {EPS_K, EPS_K, EPS_K};
//         addVector3D(eps, nray.d);
//         normalize3D(nray.d);

//         bool hitl = bboxIntersect(&(node->left->bbox), nray, tminl);
//         bool hitr = bboxIntersect(&(node->right->bbox), nray, tminr);

//         if (hitl && hitr) {
//             GPUBVHNode* first = (tminl <= tminr) ? node->left : node->right;
//             GPUBVHNode* second = (tminl <= tminr) ? node->right : node->left;

//             hitl = node_intersect(first, ray, i);
//             if (!hitl || i->t > fmaxf(tminl, tminr)) {
//                 hitr = node_intersect(second, ray, i);
//             }
//             return hitl || hitr;
//         } else if (hitl) {
//             return node_intersect(node->left, ray, i);
//         } else if (hitr) {
//             return node_intersect(node->right, ray, i);
//         }

//         return false;

//     }

// }

__device__ bool node_intersect_iter(GPUBVHNode *node, GPURay &ray, GPUIntersection *i) {
    
    GPUBVHNode* stack[64];
    GPUBVHNode** stackPtr = stack;
    *stackPtr++ = NULL;

    bool isIntersect = false;

    GPURay nray = ray;
    float eps[3] = {EPS_K, EPS_K, EPS_K};
    addVector3D(eps, nray.d);
    float invR[3], divR[3];
    invR[0] = 1 / nray.d[0];
    invR[1] = 1 / nray.d[1];
    invR[2] = 1 / nray.d[2];
    divR[0] = nray.o[0] * invR[0];
    divR[1] = nray.o[1] * invR[1];
    divR[2] = nray.o[2] * invR[2];

    float tminl, tminr;
    GPUBVHNode* node2;
#ifdef SPECULATIVE
    GPUBVHNode* postponedNode = NULL;
#endif

    while(1){

#ifdef SPECULATIVE
        bool searching = true;
#endif
        while(node){

            if (node->left == NULL && node->right == NULL) {

#ifdef SPECULATIVE
                if(searching){
                    searching = false;
                    postponedNode = node;
                    node = *--stackPtr;
                }   
#else
                break;
#endif
            }
            else {

                bool hitl = bboxIntersect(&(node->left->bbox), nray, divR, invR, tminl);
                bool hitr = bboxIntersect(&(node->right->bbox), nray, divR, invR, tminr);

                node2 = node->right;
                node = node->left;
                if(!hitr) node2 = NULL;
                if(!hitl){
                    node = node2;
                    node2 = NULL;
                }

                if (node == NULL){
                    node = *--stackPtr;
                }
                else if (node2){
                    if(tminr < tminl){
                        GPUBVHNode* tmp = node;
                        node = node2;
                        node2 = tmp;
                    }
                    *stackPtr++ = node2;
                }
            }
#ifdef SPECULATIVE
            if (!__any(searching))
               break;
#endif
        } 

#ifdef SPECULATIVE       
        if (node == NULL && postponedNode == NULL) {
            return isIntersect;
        }

        while(postponedNode->left == NULL && postponedNode->right == NULL)
        {
            for (int j = 0; j < postponedNode->range; j++) {
                int primIndex = const_params.BVHPrimMap[postponedNode->start + j];
                bool res = intersect(primIndex, ray, i);
                // bool res = false;
                isIntersect = isIntersect || res;
            }

            if (node == NULL || node->left || node->right)  break;

            postponedNode = node;
            node = *--stackPtr;
        }
        postponedNode = NULL;

#else
        if (node == NULL) {
            return isIntersect;
        }
        for (int j = 0; j < node->range; j++) {
                int primIndex = const_params.BVHPrimMap[node->start + j];
                bool res = intersect(primIndex, ray, i);
                // bool res = false;
                isIntersect = isIntersect || res;
        }
        node = *--stackPtr;
#endif
    }
    

}


// __device__ bool node_intersect(GPUBVHNode *node, GPURay &ray) {
//     if (node == NULL) {
//         return false;
//     }
//     float t0;

//     if (!bboxIntersect(&(node->bbox), ray, t0)) {
//         return false;
//     }

//     if (node->left == NULL && node->right == NULL) {
//         // node is leaf
//         for (int i = 0; i < node->range; i++) {
//             int primIndex = const_params.BVHPrimMap[node->start + i];
//             if (intersect(primIndex, ray)) {
//                 return true;
//             }
//         }
//         return false;
//     } else {
//         return node_intersect(node->left, ray) || node_intersect(node->right, ray);
//     }
// }

__device__ bool node_intersect_iter(GPUBVHNode *node, GPURay &ray) {
    GPUBVHNode* stack[64];
    GPUBVHNode** stackPtr = stack;
    *stackPtr++ = NULL;

    GPURay nray = ray;
    float eps[3] = {EPS_K, EPS_K, EPS_K};
    addVector3D(eps, nray.d);
    float invR[3], divR[3];
    invR[0] = 1 / nray.d[0];
    invR[1] = 1 / nray.d[1];
    invR[2] = 1 / nray.d[2];
    divR[0] = nray.o[0] * invR[0];
    divR[1] = nray.o[1] * invR[1];
    divR[2] = nray.o[2] * invR[2];

    float tminl, tminr;
    GPUBVHNode* node2;
#ifdef SPECULATIVE
    GPUBVHNode* postponedNode = NULL;
#endif

    while(1){

#ifdef SPECULATIVE
        bool searching = true;
#endif
        while(node){

            if (node->left == NULL && node->right == NULL) {

#ifdef SPECULATIVE
                if(searching){
                    searching = false;
                    postponedNode = node;
                    node = *--stackPtr;
                }   
#else
                break;
#endif
            }
            else {

                bool hitl = bboxIntersect(&(node->left->bbox), nray, divR, invR, tminl);
                bool hitr = bboxIntersect(&(node->right->bbox), nray, divR, invR, tminr);

                node2 = node->right;
                node = node->left;
                if(!hitr) node2 = NULL;
                if(!hitl){
                    node = node2;
                    node2 = NULL;
                }

                if (node == NULL){
                    node = *--stackPtr;
                }
                else if (node2){
                    if(tminr < tminl){
                        GPUBVHNode* tmp = node;
                        node = node2;
                        node2 = tmp;
                    }
                    *stackPtr++ = node2;
                }
            }
#ifdef SPECULATIVE
            if (!__any(searching))
               break;
#endif
        }

#ifdef SPECULATIVE
        if (node == NULL && postponedNode == NULL) {
            return false;
        }

        while(postponedNode)
        {
            for (int j = 0; j < postponedNode->range; j++) {
            int primIndex = const_params.BVHPrimMap[postponedNode->start + j];
            if(intersect(primIndex, ray))
                return true;
            }
            
            if (node == NULL || node->left || node->right)  break;

            postponedNode = node;
            node = *--stackPtr;
        }
        postponedNode = NULL;
#else
        if (node == NULL) {
            return false;
        }

        for (int j = 0; j < node->range; j++) {
            int primIndex = const_params.BVHPrimMap[node->start + j];
            if(intersect(primIndex, ray))
                return true;
        }
        node = *--stackPtr;
#endif

                
    }
}

__device__ inline bool BVH_intersect(GPURay &ray, GPUIntersection *isect) {
    return node_intersect_iter(const_params.BVHRoot, ray, isect);
}

__device__ inline bool BVH_intersect(GPURay &ray) {
    return node_intersect_iter(const_params.BVHRoot, ray);
}