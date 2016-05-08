//#define SPECULATIVE

__device__ bool BVH_traversal(GPUBVHNode *node, GPURay &ray, GPUIntersection *i) {
    
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

__device__ bool BVH_traversal(GPUBVHNode *node, GPURay &ray) {
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



__device__ inline bool BVH_intersect(GPURay &ray, GPUIntersection *isect) {
    return BVH_traversal(const_params.BVHRoot, ray, isect);
}

__device__ inline bool BVH_intersect(GPURay &ray) {
    return BVH_traversal(const_params.BVHRoot, ray);
}
