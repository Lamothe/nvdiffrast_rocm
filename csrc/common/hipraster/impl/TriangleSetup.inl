// TriangleSetup.hip.inl
// Native HIP implementation for nvdiffrast software rasterization pipeline

__device__ __forceinline__ void snapTriangle(
    const HRParams &p,
    float4 v0, float4 v1, float4 v2,
    int2 &p0, int2 &p1, int2 &p2, float3 &rcpW, int2 &lo, int2 &hi)
{
    float viewScaleX = (float)(p.widthPixelsVp << (HR_SUBPIXEL_LOG2 - 1));
    float viewScaleY = (float)(p.heightPixelsVp << (HR_SUBPIXEL_LOG2 - 1));

    rcpW = make_float3(1.0f / v0.w, 1.0f / v1.w, 1.0f / v2.w);

    // f32_to_s32_sat needs to be mapped to a standard HIP math macro or intrinsic in your Util.inl
    p0 = make_int2(f32_to_s32_sat(v0.x * rcpW.x * viewScaleX), f32_to_s32_sat(v0.y * rcpW.x * viewScaleY));
    p1 = make_int2(f32_to_s32_sat(v1.x * rcpW.y * viewScaleX), f32_to_s32_sat(v1.y * rcpW.y * viewScaleY));
    p2 = make_int2(f32_to_s32_sat(v2.x * rcpW.z * viewScaleX), f32_to_s32_sat(v2.y * rcpW.z * viewScaleY));

    lo = make_int2(min_min(p0.x, p1.x, p2.x), min_min(p0.y, p1.y, p2.y));
    hi = make_int2(max_max(p0.x, p1.x, p2.x), max_max(p0.y, p1.y, p2.y));
}

__device__ __forceinline__ uint32_t cover8x8_selectFlips(int32_t dx, int32_t dy)
{
    uint32_t flips = 0;
    if (dy > 0 || (dy == 0 && dx <= 0))
        flips ^= (1 << HR_FLIPBIT_FLIP_X) ^ (1 << HR_FLIPBIT_FLIP_Y) ^ (1 << HR_FLIPBIT_COMPL);
    if (dx > 0)
        flips ^= (1 << HR_FLIPBIT_FLIP_X) ^ (1 << HR_FLIPBIT_FLIP_Y);
    if (::abs(dx) < ::abs(dy))
        flips ^= (1 << HR_FLIPBIT_SWAP_XY) ^ (1 << HR_FLIPBIT_FLIP_Y);
    return flips;
}

__device__ __forceinline__ bool prepareTriangle(
    const HRParams &p,
    int2 p0, int2 p1, int2 p2, int2 lo, int2 hi,
    int2 &d1, int2 &d2, int32_t &area)
{
    d1 = make_int2(p1.x - p0.x, p1.y - p0.y);
    d2 = make_int2(p2.x - p0.x, p2.y - p0.y);
    area = d1.x * d2.y - d1.y * d2.x;

    if (area == 0)
        return false; // Degenerate

    if (area < 0 && (p.renderModeFlags & HipRaster::RenderModeFlag_EnableBackfaceCulling) != 0)
        return false; // Backfacing

    int sampleSize = 1 << HR_SUBPIXEL_LOG2;
    int biasX = (p.widthPixelsVp << (HR_SUBPIXEL_LOG2 - 1)) - (sampleSize >> 1);
    int biasY = (p.heightPixelsVp << (HR_SUBPIXEL_LOG2 - 1)) - (sampleSize >> 1);

    int lox = (int)add_add(lo.x, sampleSize - 1, biasX) & -sampleSize;
    int loy = (int)add_add(lo.y, sampleSize - 1, biasY) & -sampleSize;
    int hix = (hi.x + biasX) & -sampleSize;
    int hiy = (hi.y + biasY) & -sampleSize;

    if (lox > hix || loy > hiy)
        return false;

    int diff = add_sub(hix, hiy, lox) - loy;
    if (diff <= sampleSize)
    {
        int2 t0 = make_int2(add_sub(p0.x, biasX, lox), add_sub(p0.y, biasY, loy));
        int2 t1 = make_int2(add_sub(p1.x, biasX, lox), add_sub(p1.y, biasY, loy));
        int2 t2 = make_int2(add_sub(p2.x, biasX, lox), add_sub(p2.y, biasY, loy));

        int32_t e0 = t0.x * t1.y - t0.y * t1.x;
        int32_t e1 = t1.x * t2.y - t1.y * t2.x;
        int32_t e2 = t2.x * t0.y - t2.y * t0.x;

        if (area < 0)
        {
            e0 = -e0;
            e1 = -e1;
            e2 = -e2;
        }

        if (e0 < 0 || e1 < 0 || e2 < 0)
        {
            if (diff == 0)
                return false;

            t0 = make_int2(add_sub(p0.x, biasX, hix), add_sub(p0.y, biasY, hiy));
            t1 = make_int2(add_sub(p1.x, biasX, hix), add_sub(p1.y, biasY, hiy));
            t2 = make_int2(add_sub(p2.x, biasX, hix), add_sub(p2.y, biasY, hiy));
            e0 = t0.x * t1.y - t0.y * t1.x;
            e1 = t1.x * t2.y - t1.y * t2.x;
            e2 = t2.x * t0.y - t2.y * t0.x;

            if (area < 0)
            {
                e0 = -e0;
                e1 = -e1;
                e2 = -e2;
            }
            if (e0 < 0 || e1 < 0 || e2 < 0)
                return false;
        }
    }
    return true;
}

__device__ __forceinline__ void setupTriangle(
    const HRParams &p,
    HRTriangleHeader *th, HRTriangleData *td, int triId,
    float v0z, float v1z, float v2z,
    int2 p0, int2 p1, int2 p2, float3 rcpW,
    int2 d1, int2 d2, int32_t area)
{
    if (area < 0)
    {
        swap(d1, d2);
        swap(p1, p2);
        swap(v1z, v2z);
        swap(rcpW.y, rcpW.z);
        area = -area;
    }

    int2 wv0;
    wv0.x = p0.x + (p.widthPixelsVp << (HR_SUBPIXEL_LOG2 - 1));
    wv0.y = p0.y + (p.heightPixelsVp << (HR_SUBPIXEL_LOG2 - 1));

    float zcoef = (float)(HR_DEPTH_MAX - HR_DEPTH_MIN) * 0.5f;
    float zbias = (float)(HR_DEPTH_MAX + HR_DEPTH_MIN) * 0.5f;
    float3 zvert = make_float3(
        (v0z * zcoef) * rcpW.x + zbias,
        (v1z * zcoef) * rcpW.y + zbias,
        (v2z * zcoef) * rcpW.z + zbias);

    int2 zv0 = make_int2(
        wv0.x - (1 << (HR_SUBPIXEL_LOG2 - 1)),
        wv0.y - (1 << (HR_SUBPIXEL_LOG2 - 1)));

    uint3 zpleq = setupPleq(zvert, zv0, d1, d2, 1.0f / (float)area);
    uint32_t zmin = f32_to_u32_sat(fminf(fminf(zvert.x, zvert.y), zvert.z) - (float)HR_LERP_ERROR(0));

    *(uint4 *)td = make_uint4(zpleq.x, zpleq.y, zpleq.z, triId);

    uint32_t f01 = cover8x8_selectFlips(d1.x, d1.y);
    uint32_t f12 = cover8x8_selectFlips(d2.x - d1.x, d2.y - d1.y);
    uint32_t f20 = cover8x8_selectFlips(-d2.x, -d2.y);

    *(uint4 *)th = make_uint4(
        prmt(p0.x, p0.y, 0x5410),
        prmt(p1.x, p1.y, 0x5410),
        prmt(p2.x, p2.y, 0x5410),
        (zmin & 0xfffff000u) | (f01 << 6) | (f12 << 2) | (f20 >> 2));
}

__device__ __forceinline__ void triangleSetupImpl(const HRParams p)
{
    __shared__ float s_bary[HR_SETUP_WARPS * 32][18];
    float *bary = s_bary[threadIdx.x + threadIdx.y * 32];

    int taskIdx = threadIdx.x + 32 * (threadIdx.y + HR_SETUP_WARPS * blockIdx.x);
    int imageIdx = 0;

    if (p.instanceMode)
    {
        imageIdx = blockIdx.z;
        if (taskIdx >= p.numTriangles)
            return;
    }
    else
    {
        while (imageIdx < p.numImages)
        {
            int count = getImageParams(p, imageIdx).triCount;
            if (taskIdx < count)
                break;
            taskIdx -= count;
            imageIdx += 1;
        }
        if (imageIdx == p.numImages)
            return;
    }

    const HRImageParams &ip = getImageParams(p, imageIdx);
    HRAtomics &atomics = p.atomics[imageIdx];
    const int *indexBuffer = (const int *)p.indexBuffer;
    uint8_t *triSubtris = (uint8_t *)p.triSubtris + imageIdx * p.maxSubtris;
    HRTriangleHeader *triHeader = (HRTriangleHeader *)p.triHeader + imageIdx * p.maxSubtris;
    HRTriangleData *triData = (HRTriangleData *)p.triData + imageIdx * p.maxSubtris;

    int triIdx = taskIdx;
    if (!p.instanceMode)
        triIdx += ip.triOffset;

    if ((uint32_t)triIdx >= (uint32_t)p.numTriangles)
    {
        triSubtris[taskIdx] = 0;
        return;
    }

    uint4 vidx = make_uint4(
        indexBuffer[triIdx * 3 + 0],
        indexBuffer[triIdx * 3 + 1],
        indexBuffer[triIdx * 3 + 2],
        triIdx + 1);

    if (vidx.x >= (uint32_t)p.numVertices || vidx.y >= (uint32_t)p.numVertices || vidx.z >= (uint32_t)p.numVertices)
    {
        triSubtris[taskIdx] = 0;
        return;
    }

    const float4 *vertexBuffer = (const float4 *)p.vertexBuffer;
    if (p.instanceMode)
        vertexBuffer += p.numVertices * imageIdx;

    float4 v0 = vertexBuffer[vidx.x];
    float4 v1 = vertexBuffer[vidx.y];
    float4 v2 = vertexBuffer[vidx.z];

    v0.x = v0.x * p.xs + v0.w * p.xo;
    v0.y = v0.y * p.ys + v0.w * p.yo;
    v1.x = v1.x * p.xs + v1.w * p.xo;
    v1.y = v1.y * p.ys + v1.w * p.yo;
    v2.x = v2.x * p.xs + v2.w * p.xo;
    v2.y = v2.y * p.ys + v2.w * p.yo;

    if (v0.w >= fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fabsf(v0.z)) &&
        v1.w >= fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fabsf(v1.z)) &&
        v2.w >= fmaxf(fmaxf(fabsf(v2.x), fabsf(v2.y)), fabsf(v2.z)))
    {
        int2 p0, p1, p2, lo, hi;
        float3 rcpW;
        snapTriangle(p, v0, v1, v2, p0, p1, p2, rcpW, lo, hi);

        int2 d1, d2;
        int32_t area;
        bool res = prepareTriangle(p, p0, p1, p2, lo, hi, d1, d2, area);
        triSubtris[taskIdx] = res ? 1 : 0;

        if (res)
            setupTriangle(p, &triHeader[taskIdx], &triData[taskIdx], vidx.w, v0.z, v1.z, v2.z, p0, p1, p2, rcpW, d1, d2, area);
        return;
    }

    // Clip to view frustum
    float4 ov0 = v0;
    float4 od1 = make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, v1.w - v0.w);
    float4 od2 = make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, v2.w - v0.w);

    int numVerts = clipTriangleWithFrustum(bary, &ov0.x, &v1.x, &v2.x, &od1.x, &od2.x);

    v0.x = ov0.x + od1.x * bary[0] + od2.x * bary[1];
    v0.y = ov0.y + od1.y * bary[0] + od2.y * bary[1];
    v0.z = ov0.z + od1.z * bary[0] + od2.z * bary[1];
    v0.w = ov0.w + od1.w * bary[0] + od2.w * bary[1];

    v1.x = ov0.x + od1.x * bary[2] + od2.x * bary[3];
    v1.y = ov0.y + od1.y * bary[2] + od2.y * bary[3];
    v1.z = ov0.z + od1.z * bary[2] + od2.z * bary[3];
    v1.w = ov0.w + od1.w * bary[2] + od2.w * bary[3];
    float4 tv1 = v1;

    int numSubtris = 0;
    for (int i = 2; i < numVerts; i++)
    {
        v2.x = ov0.x + od1.x * bary[i * 2 + 0] + od2.x * bary[i * 2 + 1];
        v2.y = ov0.y + od1.y * bary[i * 2 + 0] + od2.y * bary[i * 2 + 1];
        v2.z = ov0.z + od1.z * bary[i * 2 + 0] + od2.z * bary[i * 2 + 1];
        v2.w = ov0.w + od1.w * bary[i * 2 + 0] + od2.w * bary[i * 2 + 1];

        int2 p0, p1, p2, lo, hi, d1, d2;
        float3 rcpW;
        int32_t area;
        snapTriangle(p, v0, v1, v2, p0, p1, p2, rcpW, lo, hi);
        if (prepareTriangle(p, p0, p1, p2, lo, hi, d1, d2, area))
            numSubtris++;
        v1 = v2;
    }

    triSubtris[taskIdx] = numSubtris;
    int subtriBase = taskIdx;

    if (numSubtris > 1)
    {
        subtriBase = atomicAdd(&atomics.numSubtris, numSubtris);
        triHeader[taskIdx].misc = subtriBase;
        if (subtriBase + numSubtris > p.maxSubtris)
            numVerts = 0;
    }

    v1 = tv1;
    for (int i = 2; i < numVerts; i++)
    {
        v2.x = ov0.x + od1.x * bary[i * 2 + 0] + od2.x * bary[i * 2 + 1];
        v2.y = ov0.y + od1.y * bary[i * 2 + 0] + od2.y * bary[i * 2 + 1];
        v2.z = ov0.z + od1.z * bary[i * 2 + 0] + od2.z * bary[i * 2 + 1];
        v2.w = ov0.w + od1.w * bary[i * 2 + 0] + od2.w * bary[i * 2 + 1];

        int2 p0, p1, p2, lo, hi, d1, d2;
        float3 rcpW;
        int32_t area;

        snapTriangle(p, v0, v1, v2, p0, p1, p2, rcpW, lo, hi);
        if (prepareTriangle(p, p0, p1, p2, lo, hi, d1, d2, area))
        {
            setupTriangle(p, &triHeader[subtriBase], &triData[subtriBase], vidx.w, v0.z, v1.z, v2.z, p0, p1, p2, rcpW, d1, d2, area);
            subtriBase++;
        }
        v1 = v2;
    }
}