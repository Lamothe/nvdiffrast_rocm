// FineRaster.inl
// Native HIP implementation for nvdiffrast fine rasterization (pixel writing)

#define WAVE32_BALLOT(pred) ((uint32_t)(__ballot(pred) >> ((__lane_id() >> 5) << 5)))
#define WAVE32_ANY(pred) (WAVE32_BALLOT(pred) != 0)
#define WAVE32_ALL(pred) (WAVE32_BALLOT(pred) == 0xFFFFFFFFu)

__device__ __forceinline__ void initTileZMax(uint32_t &tileZMax, bool &tileZUpd, volatile uint32_t *tileDepth)
{
    tileZMax = HR_DEPTH_MAX;
    tileZUpd = (::min(tileDepth[threadIdx.x], tileDepth[threadIdx.x + 32]) < tileZMax);
}

__device__ __forceinline__ void updateTileZMax(uint32_t &tileZMax, bool &tileZUpd, volatile uint32_t *tileDepth)
{
    if (WAVE32_ANY(tileZUpd))
    {
        uint32_t z = ::max(tileDepth[threadIdx.x], tileDepth[threadIdx.x + 32]);

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            uint32_t neighbor = __shfl_down(z, offset);
            if (threadIdx.x + offset < 32)
                z = ::max(z, neighbor);
        }

        tileZMax = __shfl(z, __lane_id() - threadIdx.x);
        tileZUpd = false;
    }
}

__device__ __forceinline__ void getTriangle(const HRParams &p, int32_t &triIdx, int32_t &dataIdx, uint4 &triHeader, int32_t &segment)
{
    const HRTriangleHeader *triHeaderPtr = (const HRTriangleHeader *)p.triHeader + blockIdx.z * p.maxSubtris;
    const int32_t *tileSegData = (const int32_t *)p.tileSegData + p.maxTileSegs * HR_TILE_SEG_SIZE * blockIdx.z;
    const int32_t *tileSegNext = (const int32_t *)p.tileSegNext + p.maxTileSegs * blockIdx.z;
    const int32_t *tileSegCount = (const int32_t *)p.tileSegCount + p.maxTileSegs * blockIdx.z;

    // NATIVE SAFEGUARD: Truncate uninitialized PyTorch ghost pointers
    if (segment < 0 || segment >= p.maxTileSegs)
        segment = -1;

    if (segment == -1 || threadIdx.x >= tileSegCount[segment])
    {
        triIdx = -1;
        dataIdx = -1;
    }
    else
    {
        int subtriIdx = tileSegData[segment * HR_TILE_SEG_SIZE + threadIdx.x];
        triIdx = subtriIdx >> 3;
        dataIdx = triIdx;
        subtriIdx &= 7;

        // NATIVE SAFEGUARD: Prevent OOB read if subtriIdx was corrupted
        if (triIdx < 0 || triIdx >= p.maxSubtris)
            triIdx = 0;

        if (subtriIdx != 7)
            dataIdx = triHeaderPtr[triIdx].misc + subtriIdx;

        if (dataIdx < 0 || dataIdx >= p.maxSubtris)
            dataIdx = 0;

        triHeader = *((uint4 *)triHeaderPtr + dataIdx);
    }

    if (segment != -1)
    {
        segment = tileSegNext[segment];
        if (segment < 0 || segment >= p.maxTileSegs)
            segment = -1;
    }
}

__device__ __forceinline__ bool earlyZCull(uint4 triHeader, uint32_t tileZMax)
{
    uint32_t zmin = triHeader.w & 0xFFFFF000;
    return (zmin > tileZMax);
}

__device__ __forceinline__ uint64_t trianglePixelCoverage(const HRParams &p, const uint4 &triHeader, int tileX, int tileY, volatile uint64_t *s_cover8x8_lut)
{
    int baseX = (tileX << (HR_TILE_LOG2 + HR_SUBPIXEL_LOG2)) - ((p.widthPixelsVp - 1) << (HR_SUBPIXEL_LOG2 - 1));
    int baseY = (tileY << (HR_TILE_LOG2 + HR_SUBPIXEL_LOG2)) - ((p.heightPixelsVp - 1) << (HR_SUBPIXEL_LOG2 - 1));

    int32_t v0x = sub_s16lo_s16lo(triHeader.x, baseX);
    int32_t v0y = sub_s16hi_s16lo(triHeader.x, baseY);
    int32_t v01x = sub_s16lo_s16lo(triHeader.y, triHeader.x);
    int32_t v01y = sub_s16hi_s16hi(triHeader.y, triHeader.x);
    int32_t v20x = sub_s16lo_s16lo(triHeader.x, triHeader.z);
    int32_t v20y = sub_s16hi_s16hi(triHeader.x, triHeader.z);

    uint32_t f01 = (triHeader.w >> 6) & 0x3C;
    uint32_t f12 = (triHeader.w >> 2) & 0x3C;
    uint32_t f20 = (triHeader.w << 2) & 0x3C;

    uint64_t c01 = cover8x8_exact_fast(v0x, v0y, v01x, v01y, f01, s_cover8x8_lut);
    uint64_t c12 = cover8x8_exact_fast(v0x + v01x, v0y + v01y, -v01x - v20x, -v01y - v20y, f12, s_cover8x8_lut);
    uint64_t c20 = cover8x8_exact_fast(v0x, v0y, v20x, v20y, f20, s_cover8x8_lut);

    return c01 & c12 & c20;
}

__device__ __forceinline__ uint32_t scan32_value(uint32_t value, uint32_t &total)
{
    uint32_t sum = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2)
    {
        uint32_t n = __shfl_up(sum, offset);
        if (threadIdx.x >= offset)
            sum += n;
    }
    total = __shfl(sum, __lane_id() - threadIdx.x + 31);
    return sum;
}

__device__ __forceinline__ int32_t findBit(uint64_t mask, int idx)
{
    uint32_t x = getLo(mask);
    int pop = __popc(x);
    bool p = (pop <= idx);
    if (p)
        x = getHi(mask);
    if (p)
        idx -= pop;
    int bit = p ? 32 : 0;

    pop = __popc(x & 0x0000ffffu);
    p = (pop <= idx);
    if (p)
        x >>= 16;
    if (p)
        bit += 16;
    if (p)
        idx -= pop;

    uint32_t tmp = x & 0x000000ffu;
    pop = __popc(tmp);
    p = (pop <= idx);
    if (p)
        tmp = x & 0x0000ff00u;
    if (p)
        idx -= pop;

    return findLeadingOne(tmp) + bit - idx;
}

__device__ __forceinline__ void executeROP(uint32_t color, uint32_t depth, volatile uint32_t *pColor, volatile uint32_t *pDepth)
{
    atomicMin((uint32_t *)pDepth, depth);
    __threadfence_block();

    bool act = (depth == *pDepth);

    if (act)
        atomicExch((uint32_t *)pDepth, 0);
    __threadfence_block();

    if (act)
        atomicMax((uint32_t *)pDepth, threadIdx.x);
    __threadfence_block();

    if (act && *pDepth == threadIdx.x)
    {
        *pDepth = depth;
        *pColor = color;
    }
    __threadfence_block();
}

__device__ __forceinline__ void fineRasterImpl(const HRParams p)
{
    __shared__ volatile uint64_t s_cover8x8_lut[HR_COVER8X8_LUT_SIZE];
    __shared__ volatile uint32_t s_tileColor[HR_FINE_MAX_WARPS][HR_TILE_SQR];
    __shared__ volatile uint32_t s_tileDepth[HR_FINE_MAX_WARPS][HR_TILE_SQR];
    __shared__ volatile uint32_t s_tilePeel[HR_FINE_MAX_WARPS][HR_TILE_SQR];
    __shared__ volatile uint32_t s_triDataIdx[HR_FINE_MAX_WARPS][64];
    __shared__ volatile uint64_t s_triangleCov[HR_FINE_MAX_WARPS][64];
    __shared__ volatile uint32_t s_triangleFrag[HR_FINE_MAX_WARPS][64];

    HRAtomics &atomics = p.atomics[blockIdx.z];
    const HRTriangleData *triData = (const HRTriangleData *)p.triData + blockIdx.z * p.maxSubtris;
    const int32_t *activeTiles = (const int32_t *)p.activeTiles + HR_MAXTILES_SQR * blockIdx.z;
    const int32_t *tileFirstSeg = (const int32_t *)p.tileFirstSeg + HR_MAXTILES_SQR * blockIdx.z;

    volatile uint32_t *tileColor = s_tileColor[threadIdx.y];
    volatile uint32_t *tileDepth = s_tileDepth[threadIdx.y];
    volatile uint32_t *tilePeel = s_tilePeel[threadIdx.y];
    volatile uint32_t *triDataIdx = s_triDataIdx[threadIdx.y];
    volatile uint64_t *triangleCov = s_triangleCov[threadIdx.y];
    volatile uint32_t *triangleFrag = s_triangleFrag[threadIdx.y];

    if (atomics.numSubtris > p.maxSubtris || atomics.numBinSegs > p.maxBinSegs || atomics.numTileSegs > p.maxTileSegs)
        return;

    cover8x8_setupLUT(s_cover8x8_lut);
    __syncthreads();

    for (;;)
    {
        uint32_t activeIdx = 0;
        if (threadIdx.x == 0)
            activeIdx = atomicAdd(&atomics.fineCounter, 1);
        activeIdx = __shfl(activeIdx, __lane_id() - threadIdx.x);

        if (activeIdx >= atomics.numActiveTiles)
            break;

        // NATIVE SAFEGUARD
        if (activeIdx < 0 || activeIdx >= HR_MAXTILES_SQR)
            continue;
        int tileIdx = activeTiles[activeIdx];
        if (tileIdx < 0 || tileIdx >= HR_MAXTILES_SQR)
            continue;

        int32_t segment = tileFirstSeg[tileIdx];

        int tileY = tileIdx / p.widthTiles;
        int tileX = tileIdx - tileY * p.widthTiles;
        int px = (tileX << HR_TILE_LOG2) + (threadIdx.x & (HR_TILE_SIZE - 1));
        int py = (tileY << HR_TILE_LOG2) + (threadIdx.x >> HR_TILE_LOG2);

        int triRead = 0, triWrite = 0;
        int fragRead = 0, fragWrite = 0;
        if (threadIdx.x == 0)
            triangleFrag[63] = 0;

        int maxFb = p.strideX * p.strideY;

        if (p.deferredClear)
        {
            tileColor[threadIdx.x] = p.clearColor;
            tileDepth[threadIdx.x] = p.clearDepth;
            tileColor[threadIdx.x + 32] = p.clearColor;
            tileDepth[threadIdx.x + 32] = p.clearDepth;
        }
        else
        {
            uint32_t *pColor = (uint32_t *)p.colorBuffer + maxFb * blockIdx.z;
            uint32_t *pDepth = (uint32_t *)p.depthBuffer + maxFb * blockIdx.z;

            int fbIdx1 = px + p.strideX * py;
            int fbIdx2 = px + p.strideX * (py + 4);
            if (fbIdx1 < 0 || fbIdx1 >= maxFb)
                fbIdx1 = 0;
            if (fbIdx2 < 0 || fbIdx2 >= maxFb)
                fbIdx2 = 0;

            tileColor[threadIdx.x] = pColor[fbIdx1];
            tileDepth[threadIdx.x] = pDepth[fbIdx1];
            tileColor[threadIdx.x + 32] = pColor[fbIdx2];
            tileDepth[threadIdx.x + 32] = pDepth[fbIdx2];
        }

        if (p.renderModeFlags & HipRaster::RenderModeFlag_EnableDepthPeeling)
        {
            uint32_t *pPeel = (uint32_t *)p.peelBuffer + maxFb * blockIdx.z;
            int fbIdx1 = px + p.strideX * py;
            int fbIdx2 = px + p.strideX * (py + 4);
            if (fbIdx1 < 0 || fbIdx1 >= maxFb)
                fbIdx1 = 0;
            if (fbIdx2 < 0 || fbIdx2 >= maxFb)
                fbIdx2 = 0;

            tilePeel[threadIdx.x] = pPeel[fbIdx1];
            tilePeel[threadIdx.x + 32] = pPeel[fbIdx2];
        }

        uint32_t tileZMax;
        bool tileZUpd;
        initTileZMax(tileZMax, tileZUpd, tileDepth);

        for (;;)
        {
            if (fragWrite - fragRead < 32 && segment >= 0)
            {
                updateTileZMax(tileZMax, tileZUpd, tileDepth);

                do
                {
                    int32_t triIdx, dataIdx;
                    uint4 triHeader;
                    getTriangle(p, triIdx, dataIdx, triHeader, segment);

                    if (triIdx >= 0 && earlyZCull(triHeader, tileZMax))
                        triIdx = -1;

                    uint64_t coverage = trianglePixelCoverage(p, triHeader, tileX, tileY, s_cover8x8_lut);
                    int32_t pop = (triIdx == -1) ? 0 : __popcll(coverage);

                    uint32_t fragTotal = 0;
                    uint32_t frag = scan32_value(pop, fragTotal);
                    frag += fragWrite;
                    fragWrite += fragTotal;

                    uint32_t goodMask = WAVE32_BALLOT(pop != 0);
                    if (pop != 0)
                    {
                        uint32_t prefixMask = (1u << threadIdx.x) - 1;
                        int idx = (triWrite + __popc(goodMask & prefixMask)) & 63;
                        triDataIdx[idx] = dataIdx;
                        triangleFrag[idx] = frag;
                        triangleCov[idx] = coverage;
                    }
                    triWrite += __popc(goodMask);
                } while (fragWrite - fragRead < 32 && segment >= 0);
            }

            if (fragRead == fragWrite)
                break;

            bool isBoundary = false;
            if (triRead + threadIdx.x < triWrite)
            {
                int idx = triangleFrag[(triRead + threadIdx.x) & 63] - fragRead;
                if (idx <= 32)
                    isBoundary = true;
            }

            uint32_t boundaryMask = 0;
#pragma unroll
            for (int i = 0; i < 32; ++i)
            {
                if (__shfl(isBoundary, __lane_id() - threadIdx.x + i))
                {
                    uint32_t shiftAmt = __shfl(triangleFrag[(triRead + i) & 63] - fragRead, __lane_id() - threadIdx.x + i) - 1;
                    if (shiftAmt < 32)
                        boundaryMask |= (1u << shiftAmt);
                }
            }

            int ropLaneIdx = threadIdx.x;
            bool hasFragment = (ropLaneIdx < fragWrite - fragRead);

            if (hasFragment)
            {
                uint32_t prefixMask = (1u << ropLaneIdx) - 1;
                int triBufIdx = (triRead + __popc(boundaryMask & prefixMask)) & 63;
                int fragIdx = add_sub(fragRead, ropLaneIdx, triangleFrag[(triBufIdx - 1) & 63]);

                uint64_t coverage = triangleCov[triBufIdx];
                int pixelInTile = findBit(coverage, fragIdx);

                // NATIVE SAFEGUARD
                if (pixelInTile < 0 || pixelInTile >= HR_TILE_SQR)
                    pixelInTile = 0;

                int dataIdx = triDataIdx[triBufIdx];
                if (dataIdx < 0 || dataIdx >= p.maxSubtris)
                    dataIdx = 0;

                uint32_t pixelX = (tileX << HR_TILE_LOG2) + (pixelInTile & 7);
                uint32_t pixelY = (tileY << HR_TILE_LOG2) + (pixelInTile >> 3);

                uint32_t depth = 0;
                uint4 td = *((uint4 *)triData + dataIdx * (sizeof(HRTriangleData) >> 4));
                depth = td.x * pixelX + td.y * pixelY + td.z;

                bool zkill = (p.renderModeFlags & HipRaster::RenderModeFlag_EnableDepthPeeling) && (depth <= tilePeel[pixelInTile]);
                if (!zkill)
                {
                    uint32_t oldDepth = tileDepth[pixelInTile];
                    if (depth > oldDepth)
                        zkill = true;
                    else if (oldDepth == tileZMax)
                        tileZUpd = true;
                }

                if (!zkill)
                    executeROP(td.w, depth, &tileColor[pixelInTile], &tileDepth[pixelInTile]);
            }

            fragRead = ::min(fragRead + 32, fragWrite);
            triRead += __popc(boundaryMask);
        }

        if (true)
        {
            int px = (tileX << HR_TILE_LOG2) + (threadIdx.x & (HR_TILE_SIZE - 1));
            int py = (tileY << HR_TILE_LOG2) + (threadIdx.x >> HR_TILE_LOG2);
            uint32_t *pColor = (uint32_t *)p.colorBuffer + maxFb * blockIdx.z;
            uint32_t *pDepth = (uint32_t *)p.depthBuffer + maxFb * blockIdx.z;

            int fbIdx1 = px + p.strideX * py;
            int fbIdx2 = px + p.strideX * (py + 4);
            if (fbIdx1 < 0 || fbIdx1 >= maxFb)
                fbIdx1 = 0;
            if (fbIdx2 < 0 || fbIdx2 >= maxFb)
                fbIdx2 = 0;

            pColor[fbIdx1] = tileColor[threadIdx.x];
            pDepth[fbIdx1] = tileDepth[threadIdx.x];
            pColor[fbIdx2] = tileColor[threadIdx.x + 32];
            pDepth[fbIdx2] = tileDepth[threadIdx.x + 32];
        }
    }
}