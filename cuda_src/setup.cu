/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <iostream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <thrust/sort.h>
#include <thrust/device_ptr.h>


#include <cuda_runtime.h>

#include "kernel.cu"
#include <map>


#define TILE_DIM 16

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
//using namespace std;

extern __global__ void printInfo();

CUDAPathTracer::CUDAPathTracer(PathTracer* _pathTracer)
{
    pathTracer = _pathTracer;
}

CUDAPathTracer::~CUDAPathTracer()
{
    freeBVHNode(BVHRoot);

    cudaFree(gpu_types);
    cudaFree(gpu_bsdfIndexes);
    cudaFree(gpu_positions);
    cudaFree(gpu_normals);
    cudaFree(frameBuffer);
    cudaFree(BVHPrimMap);
    cudaFree(gpu_sortedMortonCodes);
    cudaFree(gpu_leafNodes);
    cudaFree(gpu_internalNodes);
}


void CUDAPathTracer::startRayTracing()
{
    int xTileNum = TILE_DIM;
    int yTileNum = TILE_DIM;
    int width = (screenW + xTileNum - 1) / xTileNum;
    int height = (screenH + yTileNum - 1) / yTileNum;
    int blockDim = 256;
    int gridDim = (width * height + blockDim - 1) / blockDim;

    for(int i = 0; i < xTileNum; i++)
        for(int j = 0; j < yTileNum; j++)
        {
            traceScene<<<gridDim, blockDim>>>(i * width, j * height, width, height);
        }

    cudaError_t err = cudaPeekAtLastError();

    cudaDeviceSynchronize();
    cudaThreadSynchronize();


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void CUDAPathTracer::startRayTracingPT()
{
    int xTileNum = TILE_DIM;
    int yTileNum = TILE_DIM;
    int width = (screenW + xTileNum - 1) / xTileNum;
    int height = (screenH + yTileNum - 1) / yTileNum;
    int blockDim = BLOCK_DIM;
    int gridDim = 256;
    int zero = 0;

    for(int i = 0; i < xTileNum; i++)
        for(int j = 0; j < yTileNum; j++)
        {
            traceScenePT<<<gridDim, blockDim>>>(i * width, j * height, width, height);

            cudaMemcpyToSymbol(globalPoolNextRay, &zero, sizeof(int));
            cudaThreadSynchronize();

        }

    cudaError_t err = cudaPeekAtLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void CUDAPathTracer::init()
{
    loadCamera();
    loadPrimitives();
    loadLights();
    // loadBVH();
    buildBVH();
    createFrameBuffer();
    loadParameters();

    cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 24);

    //printInfo<<<1, 1>>>();
    //cudaDeviceSynchronize();

}

void CUDAPathTracer::createFrameBuffer()
{
    cudaError_t err = cudaSuccess;

    screenH = pathTracer->frameBuffer.h;
    screenW = pathTracer->frameBuffer.w;

    err = cudaMalloc((void**)&frameBuffer, 3 * screenW * screenH * sizeof(float));
    cudaMemset(frameBuffer, 0, 3 * screenW * screenH * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void CUDAPathTracer::loadCamera()
{
    //printf("load camera\n");
    //printf("camera: %p\n", pathTracer->camera);
    GPUCamera tmpCam;
    Camera* cam = pathTracer->camera;
    tmpCam.widthDivDist = cam->screenW / cam->screenDist;
    tmpCam.heightDivDist = cam->screenH / cam->screenDist;
    //printf("after loading camera\n");
    for (int i = 0; i < 9; i++) {
        tmpCam.c2w[i] = cam->c2w(i / 3, i % 3);
    }

    for (int i = 0; i < 3; i++) {
        tmpCam.pos[i] = cam->pos[i];
    }

    cudaError_t err = cudaSuccess;
    //cudaMalloc((void**)&gpu_camera,sizeof(GPUCamera));
    err = cudaMemcpyToSymbol(const_camera, &tmpCam,sizeof(GPUCamera));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void CUDAPathTracer::loadPrimitives()
{
    vector<Primitive *>& primitives = pathTracer->primitives;
    int N = primitives.size();
    int types[N];
    int bsdfs[N];
    float *positions = new float[9 * N];
    float *normals = new float[9 * N];

    primNum = N;
    map<BSDF*, int> BSDFMap;
    for (int i = 0; i < N; i++) {

        primMap[primitives[i]] = i;
        types[i] = primitives[i]->getType();
        BSDF* bsdf  = primitives[i]->get_bsdf();

        if (BSDFMap.find(bsdf) == BSDFMap.end()) {
            int index = BSDFMap.size();
            BSDFMap[bsdf] = index;
            bsdfs[i] = index;
        }
        else{
            bsdfs[i] = BSDFMap[bsdf];
        }


        if (types[i] == 0) {
            Vector3D o = ((Sphere*)primitives[i])->o;
            positions[9 * i] = o[0];
            positions[9 * i + 1] = o[1];
            positions[9 * i + 2] = o[2];
            positions[9 * i + 3] = ((Sphere*)primitives[i])->r;
        }
        else{
            const Mesh* mesh = ((Triangle*)primitives[i])->mesh;
            int v1 = ((Triangle*)primitives[i])->v1;
            int v2 = ((Triangle*)primitives[i])->v2;
            int v3 = ((Triangle*)primitives[i])->v3;

            positions[9 * i] = mesh->positions[v1][0];
            positions[9 * i + 1] = mesh->positions[v1][1];
            positions[9 * i + 2] = mesh->positions[v1][2];
            normals[9 * i] = mesh->normals[v1][0];
            normals[9 * i + 1] = mesh->normals[v1][1];
            normals[9 * i + 2] = mesh->normals[v1][2];

            positions[9 * i + 3] = mesh->positions[v2][0];
            positions[9 * i + 4] = mesh->positions[v2][1];
            positions[9 * i + 5] = mesh->positions[v2][2];
            normals[9 * i + 3] = mesh->normals[v2][0];
            normals[9 * i + 4] = mesh->normals[v2][1];
            normals[9 * i + 5] = mesh->normals[v2][2];

            positions[9 * i + 6] = mesh->positions[v3][0];
            positions[9 * i + 7] = mesh->positions[v3][1];
            positions[9 * i + 8] = mesh->positions[v3][2];
            normals[9 * i + 6] = mesh->normals[v3][0];
            normals[9 * i + 7] = mesh->normals[v3][1];
            normals[9 * i + 8] = mesh->normals[v3][2];
        }
    }
    GPUBSDF BSDFArray[BSDFMap.size()];

    for (auto itr = BSDFMap.begin(); itr != BSDFMap.end(); itr++) {
        GPUBSDF& gpu_bsdf = BSDFArray[itr->second];
        BSDF* bsdf = itr->first;
        gpu_bsdf.type = bsdf->getType();

        if (gpu_bsdf.type == 0) {
            Spectrum& albedo = ((DiffuseBSDF*)bsdf)->albedo;
            gpu_bsdf.albedo[0] = albedo.r;
            gpu_bsdf.albedo[1] = albedo.g;
            gpu_bsdf.albedo[2] = albedo.b;
        }
        else if(gpu_bsdf.type == 1){
            Spectrum& reflectance = ((MirrorBSDF*)bsdf)->reflectance;
            gpu_bsdf.reflectance[0] = reflectance.r;
            gpu_bsdf.reflectance[1] = reflectance.g;
            gpu_bsdf.reflectance[2] = reflectance.b;
        }
        else if(gpu_bsdf.type == 2){
            Spectrum& transmittance = ((RefractionBSDF*)bsdf)->transmittance;
            gpu_bsdf.transmittance[0] = transmittance.r;
            gpu_bsdf.transmittance[1] = transmittance.g;
            gpu_bsdf.transmittance[2] = transmittance.b;
            gpu_bsdf.ior = ((RefractionBSDF*)bsdf)->ior;
        }
        else if(gpu_bsdf.type == 3){
            Spectrum& reflectance = ((GlassBSDF*)bsdf)->reflectance;
            gpu_bsdf.reflectance[0] = reflectance.r;
            gpu_bsdf.reflectance[1] = reflectance.g;
            gpu_bsdf.reflectance[2] = reflectance.b;
            Spectrum& transmittance = ((GlassBSDF*)bsdf)->transmittance;
            gpu_bsdf.transmittance[0] = transmittance.r;
            gpu_bsdf.transmittance[1] = transmittance.g;
            gpu_bsdf.transmittance[2] = transmittance.b;
            gpu_bsdf.ior = ((GlassBSDF*)bsdf)->ior;
        }
        else if(gpu_bsdf.type == 4){
            Spectrum& albedo = ((EmissionBSDF*)bsdf)->radiance;
            gpu_bsdf.albedo[0] = albedo.r;
            gpu_bsdf.albedo[1] = albedo.g;
            gpu_bsdf.albedo[2] = albedo.b;

        }
    }

    cudaMalloc((void**)&gpu_types, N * sizeof(int));
    cudaMalloc((void**)&gpu_bsdfIndexes, N * sizeof(int));
    cudaMalloc((void**)&gpu_positions, 9 * N * sizeof(float));
    cudaMalloc((void**)&gpu_normals, 9 * N * sizeof(float));

    cudaMemcpy(gpu_types, types, N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_bsdfIndexes, bsdfs, N * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_positions, positions, 9 * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_normals, normals, 9 * N * sizeof(float),cudaMemcpyHostToDevice);

    //cudaMalloc((void**)&gpu_bsdfs, BSDFMap.size() * sizeof(GPUBSDF));
    delete [] positions;
    delete [] normals;

    cudaError_t err = cudaSuccess;

    err = cudaMemcpyToSymbol(const_bsdfs, BSDFArray, BSDFMap.size() * sizeof(GPUBSDF));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void CUDAPathTracer::convertBBox(BBox& bbox, GPUBBox& gpu_bbox)
{
    gpu_bbox.min[0] = bbox.min[0];
    gpu_bbox.min[1] = bbox.min[1];
    gpu_bbox.min[2] = bbox.min[2];

    gpu_bbox.max[0] = bbox.max[0];
    gpu_bbox.max[1] = bbox.max[1];
    gpu_bbox.max[2] = bbox.max[2];
}

void CUDAPathTracer::freeBVHNode(GPUBVHNode* node)
{
    GPUBVHNode gpu_node;
    cudaMemcpy(&gpu_node, node, sizeof(GPUBVHNode), cudaMemcpyDeviceToHost);

    cudaFree(node);

    if(gpu_node.left)
        freeBVHNode(gpu_node.left);

    if(gpu_node.right)
        freeBVHNode(gpu_node.right);
}

GPUBVHNode* CUDAPathTracer::generateBVHNode(BVHNode* node)
{
    GPUBVHNode gpu_node;

    gpu_node.start = node->start;
    gpu_node.range = node->range;

    //printf("(%d, %d)", (int)node->start, (int)node->range);
    convertBBox(node->bb, gpu_node.bbox);

    if (node->l)
        gpu_node.left = generateBVHNode(node->l);
    else
        gpu_node.left = NULL;

    if (node->r)
        gpu_node.right = generateBVHNode(node->r);
    else
        gpu_node.right = NULL;

    GPUBVHNode* device_node;

    cudaMalloc((void**)&device_node, sizeof(GPUBVHNode));
    cudaMemcpy(device_node, &gpu_node, sizeof(GPUBVHNode), cudaMemcpyHostToDevice);

    return device_node;
}

void CUDAPathTracer::loadBVH()
{
    vector<Primitive*> &primitives = pathTracer->bvh->primitives;

    int N = primitives.size();
    int tmpMap[N];

    for(int i = 0; i < (int)primitives.size(); i++)
    {
        tmpMap[i] = primMap[primitives[i]];
        //printf("%d ", tmpMap[i]);
    }
    //cout << endl;
    cudaMalloc((void**)&BVHPrimMap, N * sizeof(int));
    cudaMemcpy(BVHPrimMap, tmpMap, N * sizeof(int), cudaMemcpyHostToDevice);

    BVHRoot = generateBVHNode(pathTracer->bvh->root);
    // cout << endl;
    // cout << "=========================" << endl;
}

void CUDAPathTracer::buildBVH()
{
    printf("build bvh\n");
    vector<Primitive*> &primitives = pathTracer->primitives;
    
    //  can be parallelized?
    BBox sceneBox;
    for (size_t i = 0; i < pathTracer->primitives.size(); i++) {
        sceneBox.expand(pathTracer->primitives[i]->get_bbox());
    }
    Vector3D sceneMin = sceneBox.min;
    Vector3D sceneExtent = sceneBox.extent;

    int numObjects = primitives.size();
    // GPUBVHNode *leafNodes;
    // GPUBVHNode *internalNodes;
    // unsigned int *sortedMortonCodes;
    // int *sortedObjectIDs;
    printf("cudaMalloc\n");
    cudaMalloc((void**)&gpu_leafNodes, numObjects * sizeof(GPUBVHNode));
    cudaMalloc((void**)&gpu_internalNodes, (numObjects - 1) * sizeof(GPUBVHNode));
    cudaMalloc((void**)&gpu_sortedMortonCodes, numObjects * sizeof(unsigned long long));
    cudaMalloc((void**)&BVHPrimMap, numObjects * sizeof(int));

    // gpu_sortedMortonCodes = sortedMortonCodes;
    // BVHPrimMap = sortedObjectIDs;
    // gpu_leafNodes = leafNodes;
    // gpu_internalNodes = internalNodes;

    BVHParameters tmpParams;
    tmpParams.numObjects = numObjects;
    tmpParams.leafNodes = gpu_leafNodes;
    tmpParams.internalNodes = gpu_internalNodes;
    tmpParams.sortedMortonCodes = gpu_sortedMortonCodes;
    tmpParams.sortedObjectIDs = BVHPrimMap;
    tmpParams.types = gpu_types;
    tmpParams.positions = gpu_positions;

    for (int i = 0; i < 3; ++i)
    {
        tmpParams.sceneMin[i] = sceneMin[i];
        tmpParams.sceneExtent[i] = sceneExtent[i];
    }
    printf("memcpyToSymbol\n");
    cudaError_t err = cudaSuccess;

    err = cudaMemcpyToSymbol(const_bvhparams, &tmpParams, sizeof(BVHParameters));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock = 256;
    int numBlocks = (numObjects + threadsPerBlock - 1) / threadsPerBlock;
    printf("computeMorton\n");
    // assign morton code to each primitive

    computeMorton<<<numBlocks, threadsPerBlock>>>();
    cudaThreadSynchronize();

    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // sort primitive according to morton code
    //wrap raw pointer with a device_ptr to use with Thrust functions
    // unsigned int* keys = thrust::raw_pointer_cast(const_bvhparams.sortedMortonCodes);
    // int* data = thrust::raw_pointer_cast(const_bvhparams.sortedObjectIDs);
    printf("thrustSort\n");
    thrust::device_ptr<unsigned long long> keys = thrust::device_pointer_cast(gpu_sortedMortonCodes);
    thrust::device_ptr<int> data = thrust::device_pointer_cast(BVHPrimMap);
    thrust::sort_by_key(keys, keys + numObjects, data);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("generateLeaf\n");
    // generate leaf nodes
    generateLeafNode<<<numBlocks, threadsPerBlock>>>();
    cudaThreadSynchronize();
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("generateInternal\n");
    // generate internal nodes
    numBlocks = (numObjects - 1 + threadsPerBlock - 1) / threadsPerBlock;
    generateInternalNode<<<numBlocks, threadsPerBlock>>>();
    cudaThreadSynchronize();
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("buildBoundingBox\n");
    // build bouding box
    numBlocks = (numObjects + threadsPerBlock - 1) / threadsPerBlock; 
    buildBoundingBox<<<numBlocks, threadsPerBlock>>>();
    cudaThreadSynchronize();

    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    BVHRoot = gpu_internalNodes;

    // printTREE<<<1, 1>>>();
    // cudaThreadSynchronize();
    // cudaDeviceSynchronize();

    printf("build BVH done\n");
}

// Load light
void CUDAPathTracer::toGPULight(SceneLight* l, GPULight *gpuLight) {
    gpuLight->type = l->getType();
    switch(l->getType()) {
        case 0: // DirectionalLight
        {
            DirectionalLight* light = (DirectionalLight*) l;
            for (int i = 0; i < 3; ++i) {
              gpuLight->radiance[i] = light->radiance[i];
              gpuLight->dirToLight[i] = light->dirToLight[i];
            }
        }
        break;

        case 1: // InfiniteHemisphereLight
        {
            InfiniteHemisphereLight* light = (InfiniteHemisphereLight*) l;
            for (int i = 0; i < 3; ++i) {
                gpuLight->radiance[i] = light->radiance[i];
                for (int j = 0; j < 3; j++) {
                    gpuLight->sampleToWorld[3 * i + j] = light->sampleToWorld(i, j);
                }
            }

        }
        break;

        case 2: // PointLight
        {
            PointLight* light = (PointLight*) l;
            for (int i = 0; i < 3; ++i) {
              gpuLight->radiance[i] = light->radiance[i];
              gpuLight->position[i] = light->position[i];
            }
        }
        break;

        case 3: // AreaLight
        {
            AreaLight* light = (AreaLight*) l;
            for (int i = 0; i < 3; ++i) {
              gpuLight->radiance[i] = light->radiance[i];
              gpuLight->position[i] = light->position[i];
              gpuLight->direction[i] = light->direction[i];
              gpuLight->dim_x[i] = light->dim_x[i];
              gpuLight->dim_y[i] = light->dim_y[i];
              gpuLight->area = light->area;
            }
        }
        break;

        default:
        break;
    }
}

void CUDAPathTracer::loadLights() {
    int tmpLightNum = pathTracer->scene->lights.size();

    GPULight tmpLights[tmpLightNum];

    for (int i = 0; i < tmpLightNum; ++i) {
        //displayLight(pathTracer->scene->lights[i]);
        toGPULight(pathTracer->scene->lights[i], tmpLights + i);
    }
    //cudaMalloc((void**)&gpu_lights, sizeof(GPULight) * tmpLightNum);


    cudaError_t err = cudaSuccess;

    err = cudaMemcpyToSymbol(const_lights, tmpLights, sizeof(GPULight) * tmpLightNum);


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

//    GPULight rtLights[tmpLightNum];
//    cudaMemcpy(rtLights, gpu_lights, sizeof(GPULight) * tmpLightNum, cudaMemcpyDeviceToHost);
//    //printf("==================\n");
//    for (int i = 0; i < tmpLightNum; ++i)
//    {
//        displayGPULight(rtLights + i);
//    }
}

// load Parameters
void CUDAPathTracer::loadParameters() {
    Parameters tmpParams;
    tmpParams.screenW = pathTracer->frameBuffer.w;
    tmpParams.screenH = pathTracer->frameBuffer.h;
    tmpParams.max_ray_depth = pathTracer->max_ray_depth;
    tmpParams.ns_aa = pathTracer->ns_aa;
    tmpParams.ns_area_light = pathTracer->ns_area_light;
    tmpParams.lightNum = pathTracer->scene->lights.size();
    tmpParams.types = gpu_types;
    tmpParams.bsdfIndexes = gpu_bsdfIndexes;
    tmpParams.positions = gpu_positions;
    tmpParams.normals = gpu_normals;
    tmpParams.primNum = primNum;
    tmpParams.frameBuffer = frameBuffer;
    tmpParams.BVHPrimMap = BVHPrimMap;
    tmpParams.BVHRoot = BVHRoot;

    cudaError_t err = cudaSuccess;

    err = cudaMemcpyToSymbol(const_params, &tmpParams, sizeof(Parameters));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    //Parameters rtParms;
    //cudaMemcpy(&rtParms, parms, sizeof(Parameters), cudaMemcpyDeviceToHost);
    //printf("screenW: %d, screenH: %d, max_ray_depth: %d, ns_aa: %d, ns_area_light: %d, lightNum: %d\n", rtParms.screenW, rtParms.screenH, rtParms.max_ray_depth, rtParms.ns_aa, rtParms.ns_area_light, rtParms.lightNum);
}

void CUDAPathTracer::updateHostSampleBuffer() {
    float* gpuBuffer = (float*) malloc(sizeof(float) * (3 * screenW * screenH));
    cudaError_t err = cudaSuccess;

    err = cudaMemcpy(gpuBuffer, frameBuffer, sizeof(float) * (3 * screenW * screenH), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed! (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    pathTracer->updateBufferFromGPU(gpuBuffer);
    free(gpuBuffer);
}

void PathTracer::updateBufferFromGPU(float* gpuBuffer) {
    size_t w = sampleBuffer.w;
    size_t h = sampleBuffer.h;
    for (int x = 0; x < w; ++x)
    {
        for (int y = 0; y < h; ++y)
        {
            int index = 3 * (y * w + x);
            Spectrum s(gpuBuffer[index], gpuBuffer[index + 1], gpuBuffer[index + 2]);
            //cout << s.r << "," << s.g << "," << s.b << endl;
            sampleBuffer.update_pixel(s, x, y);
        }
    }
    sampleBuffer.toColor(frameBuffer, 0, 0, w, h);
}
