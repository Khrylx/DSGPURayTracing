#include "bvh.h"

#include "CMU462/CMU462.h"
#include "static_scene/triangle.h"

#include <iostream>
#include <stack>

using namespace std;

namespace CMU462 { namespace StaticScene {

struct Bucket{
    BBox bb;
    int prim_count;
    Bucket():prim_count(0) {}
};

    int maxLeaf;
// buildBVH helper
void buildBVH(vector<Primitive *>& primitives, BVHNode* node, int bucketNum, size_t max_leaf_size)
{
    
    BBox lbb,rbb;
    int lRange,rRange;
    
    // Keep less local variables to save stack space.
    {
        double minC[3] = {INF_D, INF_D, INF_D};
        int minBIndex[3];
        
        for (int k = 0; k < 3; k++) {
            double ub = node->bb.max[k];
            double lb = node->bb.min[k];
            if (ub == lb) {
                continue;
            }
            double interval = (ub-lb) / bucketNum;
            vector<Bucket> Buckets(bucketNum);
            vector<Bucket> rBuckets(bucketNum);
            
            // calculate buckets
            for (int i = 0; i < node->range; i++) {
                Primitive* P = primitives[node->start + i];
                BBox pbb = P->get_bbox();
                double c = (pbb.min[k] + pbb.max[k])*0.5;
                int bIndex = (c - lb) / interval;
                //cout << bIndex <<endl;
                Buckets[bIndex].bb.expand(pbb);
                Buckets[bIndex].prim_count++;
            }
            
            // expand bucket reversely
            for (int i = 0; i < bucketNum; i++) {
                rBuckets[i] = Buckets[bucketNum-i-1];
                if (i > 0) {
                    rBuckets[i].bb.expand(rBuckets[i-1].bb);
                    rBuckets[i].prim_count += rBuckets[i-1].prim_count;
                }
            }
            
            
            // expand bucket sequentially
            for (int i = 1; i < bucketNum; i++) {
                Buckets[i].bb.expand(Buckets[i-1].bb);
                Buckets[i].prim_count += Buckets[i-1].prim_count;
            }
            
            // calculate partition reaching minimal cost
            for (int i = 0; i < bucketNum-1; i++) {
                Bucket& b1 = Buckets[i];
                Bucket& b2 = rBuckets[bucketNum-i-2];
                double C = (b1.bb.extent.x * b1.bb.extent.y + b1.bb.extent.x * b1.bb.extent.z + b1.bb.extent.y * b1.bb.extent.z) * b1.prim_count
                +(b2.bb.extent.x * b2.bb.extent.y + b2.bb.extent.x * b2.bb.extent.z + b2.bb.extent.y * b2.bb.extent.z) * b2.prim_count;
                if (C < minC[k]) {
                    minC[k] = C;
                    minBIndex[k] = i+1;
                }
            }
        }
        
        // select best axis
        int axis = 0;
        double cost = minC[0];
        for (int i = 1; i < 3; i++) {
            if (minC[i] < cost) {
                axis = i;
                cost = minC[i];
            }
        }
        
        // partition primitives by the partition line computed using quicksort-like algorithm.
        double ub = node->bb.max[axis];
        double lb = node->bb.min[axis];
        double pLine = lb + (ub-lb)*minBIndex[axis]/bucketNum;
        
        int i = node->start - 1;
        int j = node->start + node->range;
        BBox bb1,bb2;
        
        while (i < j) {
            do{
                i++;
                if (i >= (int)(node->start + node->range)) {
                    break;
                }
                bb1 = primitives[i]->get_bbox();
            }while((bb1.min[axis] + bb1.max[axis])*0.5 < pLine);
            
            do{
                j--;
                if (j < (int)node->start) {
                    break;
                }
                bb2 = primitives[j]->get_bbox();
            }while((bb2.min[axis] + bb2.max[axis])*0.5 > pLine);
            
            if (i < j) {
                Primitive* tmp = primitives[i];
                primitives[i] = primitives[j];
                primitives[j] = tmp;
            }
            else
                break;
        }
        
        // Recalculate bounding box
        lRange = i - node->start;
        rRange = node->range - lRange;
        
        for (int j = 0; j < node->range; j++) {
            bb1 = primitives[node->start + j]->get_bbox();
            if (j < lRange) {
                lbb.expand(bb1);
            }
            else{
                rbb.expand(bb1);
            }
        }
    }
    
    
    node->l = (lRange == 0 || rRange == 0) ? NULL : new BVHNode(lbb, node->start, lRange);
    node->r = (lRange == 0 || rRange == 0) ? NULL : new BVHNode(rbb, node->start + lRange, rRange);
    

    if (lRange <= max_leaf_size && rRange <= max_leaf_size) {
        //cout << lRange << ":" << rRange << endl;
        //maxLeaf = std::max(lRange, std::max(rRange,maxLeaf));
        return;
    }
    else if (lRange <= max_leaf_size){
        // lRange > 0 is for the case if all primitives are together.
        if (lRange > 0) {
            buildBVH(primitives, node->r, bucketNum, max_leaf_size);
        }
//        else{
//            maxLeaf = std::max(lRange, std::max(rRange,maxLeaf));
//        }
    }
    else if (rRange <= max_leaf_size){
        // rRange > 0 is for the case if all primitives are together.
        if (rRange > 0) {
            buildBVH(primitives, node->l, bucketNum, max_leaf_size);
        }
//        else{
//            maxLeaf = std::max(lRange, std::max(rRange,maxLeaf));
//        }
    }
    else{
        buildBVH(primitives, node->l, bucketNum, max_leaf_size);
        buildBVH(primitives, node->r, bucketNum, max_leaf_size);
    }
    
    
    return;
    
}
    
    
BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  this->primitives = _primitives;

  // TODO:
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bb;
  for (size_t i = 0; i < primitives.size(); ++i) {
    bb.expand(primitives[i]->get_bbox());
  }

  root = new BVHNode(bb, 0, primitives.size());

    maxLeaf = 0;
    buildBVH(primitives, root, 32, max_leaf_size);
    cout << "max:" << maxLeaf << endl;
}

//  destroy BVH nodesv
void destroyNode(BVHNode* node){
    if (node->l) {
        destroyNode(node->l);
    }
    if (node->r) {
        destroyNode(node->r);
    }
    free(node);
}
    
BVHAccel::~BVHAccel() {

  // TODO:
  // Implement a proper destructor for your BVH accelerator aggregate
    destroyNode(get_root());
}

BBox BVHAccel::get_bbox() const {
  return root->bb;
}

// node intersect helper calculating intersection
bool node_intersect(const BVHNode* node,const vector<Primitive *>& primitives, const Ray &ray,Intersection *i)    {
    if (node->l == NULL && node->r == NULL) {
        bool intersect = false;
        for (int j = 0; j < node->range; j++)
        {
            bool res = primitives[j + node->start]->intersect(ray,i);
            intersect = intersect || res;
        }
        return intersect;
    }
    else if(node->l == NULL){
        return node_intersect(node->r, primitives, ray, i);
    }
    else if(node->r == NULL){
        return node_intersect(node->l, primitives, ray, i);
    }
    else{
        double tminl = -INF_D;
        double tminr = -INF_D;
        double tmaxl = INF_D;
        double tmaxr = INF_D;
        
        // improve numerical stability
        Ray nray = ray;
        nray.d += EPS_D;
        nray.d.normalize();
        
        bool hitl = node->l->bb.intersect(nray, tminl, tmaxl);
        bool hitr = node->r->bb.intersect(nray, tminr, tmaxr);
        
        if (hitl && hitr) {
            BVHNode* first = (tminl <= tminr) ? node->l : node->r;
            BVHNode* second = (tminl <= tminr) ? node->r : node->l;
            
            hitl = node_intersect(first, primitives, ray, i);
            if (!hitl || i->t > max(tminl,tminr) ) {
                hitr = node_intersect(second, primitives, ray, i);
            }
            return hitl || hitr;
        }
        else if(hitl){
            return node_intersect(node->l, primitives, ray, i);
        }
        else if(hitr){
            return node_intersect(node->r, primitives, ray, i);
        }
        
        return false;
        
    }
    
    return false;
}

// node intersect helper without calculating intersection
bool node_intersect(const BVHNode* node,const vector<Primitive *>& primitives, const Ray &ray)    {
    
    if (node->l == NULL && node->r == NULL) {
        for (int j = 0; j < node->range; j++)
        {
            bool res = primitives[j + node->start]->intersect(ray);
            if (res) {
                return true;
            }
        }
        return false;
    }
    else if(node->l == NULL){
        return node_intersect(node->r, primitives, ray);
    }
    else if(node->r == NULL){
        return node_intersect(node->l, primitives, ray);
    }
    else{
        double tminl = -INF_D;
        double tminr = -INF_D;
        double tmaxl = INF_D;
        double tmaxr = INF_D;
        
        // improve numerical stability
        Ray nray = ray;
        nray.d += EPS_D;
        nray.d.normalize();
        
        bool hitl = node->l->bb.intersect(nray, tminl, tmaxl);
        bool hitr = node->r->bb.intersect(nray, tminr, tmaxr);
        
        if (hitl && hitr) {
            BVHNode* first = (tminl <= tminr) ? node->l : node->r;
            BVHNode* second = (tminl <= tminr) ? node->r : node->l;
            
            return node_intersect(first, primitives, ray) || node_intersect(second, primitives, ray);
        }
        else if(hitl){
            return node_intersect(node->l, primitives, ray);
        }
        else if(hitr){
            return node_intersect(node->r, primitives, ray);
        }
        return false;
    }
    return false;
}

bool BVHAccel::intersect(const Ray &ray) const {

  // TODO:
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate.
    
    
  return node_intersect(get_root(), primitives, ray);

}

bool BVHAccel::intersect(const Ray &ray, Intersection *i) const {

  // TODO:
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate. When an intersection does happen.
  // You should store the non-aggregate primitive in the intersection data
  // and not the BVH aggregate itself.
//    bool intersect = false;
//    for (int j = 0; j < primitives.size(); j++)
//    {
//        bool res = primitives[j]->intersect(ray,i);
//        intersect = intersect || res;
//    }
//    
//    return intersect;

    
  return node_intersect(get_root(), primitives, ray, i);

}

}  // namespace StaticScene
}  // namespace CMU462
