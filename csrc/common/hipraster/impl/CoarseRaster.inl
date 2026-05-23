// CoarseRaster.inl
// Native HIP implementation for nvdiffrast coarse rasterization

#define WAVE32_BALLOT(pred) ((uint32_t)(__ballot(pred) >> ((__lane_id() >> 5) << 5)))
#define WAVE32_ANY(pred) (WAVE32_BALLOT(pred) != 0)
#define WAVE32_ALL(pred) (WAVE32_BALLOT(pred) == 0xFFFFFFFFu)

// FIX: Wave32-scoped shuffle primitives for 64-lane AMD wavefronts
#define WAVE32_SHFL_UP(val, delta, myLane) \
    __shfl(val, (myLane) - (delta) < 0 ? 0 : (myLane) - (delta))
#define WAVE32_SHFL_DOWN(val, delta, myLane) \
    __shfl(val, (myLane) + (delta) >= 32 ? 31 : (myLane) + (delta))

__device__ __forceinline__ int globalTileIdx(int tileInBin, int widthTiles)
{
    int tileX = tileInBin & (HR_BIN_SIZE - 1);
    int tileY = tileInBin >> HR_BIN_LOG2;
    return tileX + tileY * widthTiles;
}

__device__ __forceinline__ void coarseRasterImpl(const HRParams p)
{
    __shared__ uint32_t s_workCounter;
    __shared__ uint32_t s_scanTemp[HR_COARSE_WARPS][48];

    __shared__ uint32_t s_binOrder[HR_MAXBINS_SQR];
    __shared__ int32_t s_binStreamCurrSeg[HR_BIN_STREAMS_SIZE];
    __shared__ int32_t s_binStreamFirstTri[HR_BIN_STREAMS_SIZE];
    __shared__ int32_t s_triQueue[HR_COARSE_QUEUE_SIZE];
    __shared__ int32_t s_triQueueWritePos;
    __shared__ uint32_t s_binStreamSelectedOfs;
    __shared__ uint32_t s_binStreamSelectedSize;

    __shared__ uint32_t s_warpEmitMask[HR_COARSE_WARPS][HR_BIN_SQR + 1];
    __shared__ uint32_t s_warpEmitPrefixSum[HR_COARSE_WARPS][HR_BIN_SQR + 1];
    __shared__ uint32_t s_tileEmitPrefixSum[HR_BIN_SQR + 1];
    __shared__ uint32_t s_tileAllocPrefixSum[HR_BIN_SQR + 1];
    __shared__ int32_t s_tileStreamCurrOfs[HR_BIN_SQR];

    __shared__ uint32_t s_firstAllocSeg;
    __shared__ uint32_t s_firstActiveIdx;

    HRAtomics &atomics = p.atomics[blockIdx.z];
    const HRTriangleHeader *triHeader = (const HRTriangleHeader *)p.triHeader + p.maxSubtris * blockIdx.z;
    const int32_t *binFirstSeg = (const int32_t *)p.binFirstSeg + HR_MAXBINS_SQR * HR_BIN_STREAMS_SIZE * blockIdx.z;
    const int32_t *binTotal = (const int32_t *)p.binTotal + HR_MAXBINS_SQR * HR_BIN_STREAMS_SIZE * blockIdx.z;
    const int32_t *binSegData = (const int32_t *)p.binSegData + p.maxBinSegs * HR_BIN_SEG_SIZE * blockIdx.z;
    const int32_t *binSegNext = (const int32_t *)p.binSegNext + p.maxBinSegs * blockIdx.z;
    const int32_t *binSegCount = (const int32_t *)p.binSegCount + p.maxBinSegs * blockIdx.z;
    int32_t *activeTiles = (int32_t *)p.activeTiles + HR_MAXTILES_SQR * blockIdx.z;
    int32_t *tileFirstSeg = (int32_t *)p.tileFirstSeg + HR_MAXTILES_SQR * blockIdx.z;
    int32_t *tileSegData = (int32_t *)p.tileSegData + p.maxTileSegs * HR_TILE_SEG_SIZE * blockIdx.z;
    int32_t *tileSegNext = (int32_t *)p.tileSegNext + p.maxTileSegs * blockIdx.z;
    int32_t *tileSegCount = (int32_t *)p.tileSegCount + p.maxTileSegs * blockIdx.z;

    int tileLog = HR_TILE_LOG2 + HR_SUBPIXEL_LOG2;
    int thrInBlock = threadIdx.x + threadIdx.y * 32;
    int emitShift = HR_BIN_LOG2 * 2 + 5;
    int myLane = __lane_id() & 31; // FIX: sub-wave lane ID for 64-lane AMD wavefronts

    if (atomics.numSubtris > p.maxSubtris || atomics.numBinSegs > p.maxBinSegs)
        return;

    if (thrInBlock == 0)
    {
        s_tileEmitPrefixSum[0] = 0;
        s_tileAllocPrefixSum[0] = 0;
    }
    s_scanTemp[threadIdx.y][threadIdx.x] = 0;

    for (int binIdx = thrInBlock; binIdx < p.numBins; binIdx += HR_COARSE_WARPS * 32)
    {
        int count = 0;
        for (int i = 0; i < HR_BIN_STREAMS_SIZE; i++)
            count += binTotal[(binIdx << HR_BIN_STREAMS_LOG2) + i];
        s_binOrder[binIdx] = (~count << (HR_MAXBINS_LOG2 * 2)) | binIdx;
    }

    __syncthreads();
    sortShared(s_binOrder, p.numBins);

    for (;;)
    {
        if (thrInBlock == 0)
            s_workCounter = atomicAdd(&atomics.coarseCounter, 1);
        __syncthreads();

        int workCounter = s_workCounter;
        if (workCounter >= p.numBins)
            break;

        uint32_t binOrder = s_binOrder[workCounter];
        bool binEmpty = ((~binOrder >> (HR_MAXBINS_LOG2 * 2)) == 0);
        if (binEmpty && !p.deferredClear)
            break;

        int binIdx = binOrder & (HR_MAXBINS_SQR - 1);

        int triQueueWritePos = 0;
        int triQueueReadPos = 0;

        if (thrInBlock < HR_BIN_STREAMS_SIZE)
        {
            int segIdx = binFirstSeg[(binIdx << HR_BIN_STREAMS_LOG2) + thrInBlock];
            s_binStreamCurrSeg[thrInBlock] = segIdx;
            s_binStreamFirstTri[thrInBlock] = (segIdx == -1) ? ~0u : binSegData[segIdx << HR_BIN_SEG_LOG2];
        }

        for (int tileInBin = HR_COARSE_WARPS * 32 - 1 - thrInBlock; tileInBin < HR_BIN_SQR; tileInBin += HR_COARSE_WARPS * 32)
            s_tileStreamCurrOfs[tileInBin] = -HR_TILE_SEG_SIZE;

        int binY = idiv_fast(binIdx, p.widthBins);
        int binX = binIdx - binY * p.widthBins;
        int originX = (binX << (HR_BIN_LOG2 + tileLog)) - (p.widthPixelsVp << (HR_SUBPIXEL_LOG2 - 1));
        int originY = (binY << (HR_BIN_LOG2 + tileLog)) - (p.heightPixelsVp << (HR_SUBPIXEL_LOG2 - 1));
        int maxTileXInBin = ::min(p.widthTiles - (binX << HR_BIN_LOG2), HR_BIN_SIZE) - 1;
        int maxTileYInBin = ::min(p.heightTiles - (binY << HR_BIN_LOG2), HR_BIN_SIZE) - 1;
        int binTileIdx = (binX + binY * p.widthTiles) << HR_BIN_LOG2;

        if (!binEmpty)
        {
            do
            {
                while (triQueueWritePos - triQueueReadPos <= HR_COARSE_WARPS * 32)
                {
                    if (thrInBlock == 0)
                    {
                        uint32_t minTri = ~0u;
                        int bestStream = -1;
                        for (int i = 0; i < HR_BIN_STREAMS_SIZE; i++)
                        {
                            uint32_t val = s_binStreamFirstTri[i];
                            if (val < minTri)
                            {
                                minTri = val;
                                bestStream = i;
                            }
                        }
                        s_scanTemp[0][0] = bestStream;
                    }
                    __syncthreads();

                    int winner = s_scanTemp[0][0];
                    if (winner == -1)
                    {
                        if (thrInBlock == 0)
                            s_binStreamSelectedOfs = -1;
                    }
                    else if (thrInBlock == winner)
                    {
                        int segIdx = s_binStreamCurrSeg[thrInBlock];
                        s_binStreamSelectedOfs = segIdx << HR_BIN_SEG_LOG2;
                        if (segIdx != -1)
                        {
                            int segSize = binSegCount[segIdx];
                            int segNext = binSegNext[segIdx];
                            s_binStreamSelectedSize = segSize;
                            s_triQueueWritePos = triQueueWritePos + segSize;
                            s_binStreamCurrSeg[thrInBlock] = segNext;
                            if (segNext == -1)
                                s_binStreamFirstTri[thrInBlock] = ~0u;
                            else
                                s_binStreamFirstTri[thrInBlock] = binSegData[segNext << HR_BIN_SEG_LOG2];
                        }
                    }
                    __syncthreads();

                    triQueueWritePos = s_triQueueWritePos;
                    int segOfs = s_binStreamSelectedOfs;
                    if (segOfs < 0)
                        break;

                    int segSize = s_binStreamSelectedSize;
                    __syncthreads();

                    for (int idxInSeg = HR_COARSE_WARPS * 32 - 1 - thrInBlock; idxInSeg < segSize; idxInSeg += HR_COARSE_WARPS * 32)
                    {
                        int32_t triIdx = binSegData[segOfs + idxInSeg];
                        s_triQueue[(triQueueWritePos - segSize + idxInSeg) & (HR_COARSE_QUEUE_SIZE - 1)] = triIdx;
                    }
                }

                for (int maskIdx = thrInBlock; maskIdx < HR_COARSE_WARPS * HR_BIN_SQR; maskIdx += HR_COARSE_WARPS * 32)
                    s_warpEmitMask[maskIdx >> (HR_BIN_LOG2 * 2)][maskIdx & (HR_BIN_SQR - 1)] = 0;
                __syncthreads();

                int triIdx = -1;
                if (triQueueReadPos + thrInBlock < triQueueWritePos)
                    triIdx = s_triQueue[(triQueueReadPos + thrInBlock) & (HR_COARSE_QUEUE_SIZE - 1)];

                uint4 triData = make_uint4(0, 0, 0, 0);
                if (triIdx != -1)
                {
                    int dataIdx = triIdx >> 3;
                    int subtriIdx = triIdx & 7;
                    if (subtriIdx != 7)
                        dataIdx = triHeader[dataIdx].misc + subtriIdx;
                    triData = *((uint4 *)triHeader + dataIdx);
                }

                if (WAVE32_ANY(triIdx != -1))
                {
                    int32_t v0x = sub_s16lo_s16lo(triData.x, originX);
                    int32_t v0y = sub_s16hi_s16lo(triData.x, originY);
                    int32_t d01x = sub_s16lo_s16lo(triData.y, triData.x);
                    int32_t d01y = sub_s16hi_s16hi(triData.y, triData.x);
                    int32_t d02x = sub_s16lo_s16lo(triData.z, triData.x);
                    int32_t d02y = sub_s16hi_s16hi(triData.z, triData.x);

                    int lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> tileLog, 0, maxTileXInBin);
                    int loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> tileLog, 0, maxTileYInBin);
                    int hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> tileLog, 0, maxTileXInBin);
                    int hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> tileLog, 0, maxTileYInBin);

                    int sizex = add_sub(hix, 1, lox);
                    int sizey = add_sub(hiy, 1, loy);
                    int area = sizex * sizey;

                    uint8_t *currPtr = (uint8_t *)&s_warpEmitMask[threadIdx.y][lox + (loy << HR_BIN_LOG2)];
                    int ptrYInc = HR_BIN_SIZE * 4 - (sizex << 2);
                    uint32_t maskBit = 1u << threadIdx.x;

                    if (WAVE32_ALL(sizex <= 2 && sizey <= 2))
                    {
                        if (triIdx != -1)
                        {
                            atomicOr((uint32_t *)currPtr, maskBit);
                            if (sizex == 2)
                                atomicOr((uint32_t *)(currPtr + 4), maskBit);
                            if (sizey == 2)
                                atomicOr((uint32_t *)(currPtr + HR_BIN_SIZE * 4), maskBit);
                            if (sizex == 2 && sizey == 2)
                                atomicOr((uint32_t *)(currPtr + 4 + HR_BIN_SIZE * 4), maskBit);
                        }
                    }
                    else
                    {
                        uint32_t aabbMask = add_sub(2 << hix, 0x20000 << hiy, 1 << lox) - (0x10000 << loy);
                        if (triIdx == -1)
                            aabbMask = 0;

#pragma unroll
                        for (int offset = 1; offset < 32; offset *= 2)
                        {
                            uint32_t neighbor = WAVE32_SHFL_UP(aabbMask, offset, myLane);
                            if (myLane >= offset)
                                aabbMask |= neighbor;
                        }

                        aabbMask = __shfl(aabbMask, myLane - threadIdx.x + 31);

                        uint32_t maskX = aabbMask & 0xFFFF;
                        uint32_t maskY = aabbMask >> 16;
                        int wlox = __clz(__brev(maskX ^ (maskX - 1)));
                        int wloy = __clz(__brev(maskY ^ (maskY - 1)));
                        int whix = 31 - __clz(maskX);
                        int whiy = 31 - __clz(maskY);
                        int warea = (add_sub(whix, 1, wlox)) * (add_sub(whiy, 1, wloy));

                        int32_t d12x = d02x - d01x, d12y = d02y - d01y;
                        v0x -= lox << tileLog;
                        v0y -= loy << tileLog;

                        int32_t t01 = v0x * d01y - v0y * d01x;
                        int32_t t02 = v0y * d02x - v0x * d02y;
                        int32_t t12 = d01x * d12y - d01y * d12x - t01 - t02;

                        int32_t b01 = add_sub(t01 >> tileLog, ::max(d01x, 0), ::min(d01y, 0));
                        int32_t b02 = add_sub(t02 >> tileLog, ::max(d02y, 0), ::min(d02x, 0));
                        int32_t b12 = add_sub(t12 >> tileLog, ::max(d12x, 0), ::min(d12y, 0));

                        d01x += sizex * d01y;
                        d02x += sizex * d02y;
                        d12x += sizex * d12y;

                        if (WAVE32_ANY(warea * 4 <= area * 8))
                        {
                            bool act = (triIdx != -1);
                            if (act)
                            {
                                for (int y = wloy; y <= whiy; y++)
                                {
                                    bool yIn = (y >= loy && y <= hiy);
                                    uint32_t yMask = WAVE32_BALLOT(yIn);
                                    if (yIn)
                                    {
                                        for (int x = wlox; x <= whix; x++)
                                        {
                                            bool xyIn = (x >= lox && x <= hix);
                                            uint32_t xyMask = WAVE32_BALLOT(xyIn) & yMask;
                                            if (xyIn)
                                            {
                                                uint32_t res = WAVE32_BALLOT(b01 >= 0 && b02 >= 0 && b12 >= 0) & xyMask;
                                                if (threadIdx.x == 31 - __clz(xyMask))
                                                    *(uint32_t *)currPtr = res;
                                                currPtr += 4;
                                                b01 -= d01y;
                                                b02 += d02y;
                                                b12 -= d12y;
                                            }
                                        }
                                        currPtr += ptrYInc;
                                        b01 += d01x;
                                        b02 -= d02x;
                                        b12 += d12x;
                                    }
                                }
                            }
                        }
                        else
                        {
                            if (triIdx != -1)
                            {
                                uint8_t *skipPtr = currPtr + (sizex << 2);
                                uint8_t *endPtr = currPtr + (sizey << (HR_BIN_LOG2 + 2));
                                do
                                {
                                    if (b01 >= 0 && b02 >= 0 && b12 >= 0)
                                        atomicOr((uint32_t *)currPtr, maskBit);
                                    currPtr += 4;
                                    b01 -= d01y;
                                    b02 += d02y;
                                    b12 -= d12y;
                                    if (currPtr == skipPtr)
                                    {
                                        currPtr += ptrYInc;
                                        b01 += d01x;
                                        b02 -= d02x;
                                        b12 += d12x;
                                        skipPtr += HR_BIN_SIZE * 4;
                                    }
                                } while (currPtr != endPtr);
                            }
                        }
                    }
                }
                __syncthreads();

                for (int tileInBin_base = 0; tileInBin_base < HR_BIN_SQR; tileInBin_base += HR_COARSE_WARPS * 32)
                {
                    int tileInBin = tileInBin_base + thrInBlock;
                    bool act = (tileInBin < HR_BIN_SQR);
                    if (act)
                    {
                        uint8_t *srcPtr = (uint8_t *)&s_warpEmitMask[0][tileInBin];
                        uint8_t *dstPtr = (uint8_t *)&s_warpEmitPrefixSum[0][tileInBin];
                        int tileEmits = 0;
                        for (int i = 0; i < HR_COARSE_WARPS; i++)
                        {
                            tileEmits += __popc(*(uint32_t *)srcPtr);
                            *(uint32_t *)dstPtr = tileEmits;
                            srcPtr += (HR_BIN_SQR + 1) * 4;
                            dstPtr += (HR_BIN_SQR + 1) * 4;
                        }

                        int spaceLeft = -s_tileStreamCurrOfs[tileInBin] & (HR_TILE_SEG_SIZE - 1);
                        int tileAllocs = (tileEmits - spaceLeft + HR_TILE_SEG_SIZE - 1) >> HR_TILE_SEG_LOG2;
                        uint32_t sum = (tileEmits << emitShift) | tileAllocs;

#pragma unroll
                        for (int offset = 1; offset < 32; offset *= 2)
                        {
                            uint32_t n = WAVE32_SHFL_UP(sum, offset, myLane);
                            if (myLane >= offset)
                                sum += n;
                        }
                        s_tileEmitPrefixSum[tileInBin + 1] = sum;

                        if (myLane == 31)
                            s_scanTemp[0][(tileInBin >> 5) + 16] = sum;
                    }
                }

                __syncthreads();

                if (threadIdx.y == 0)
                {
                    uint32_t sum = (myLane < HR_BIN_SQR / 32) ? s_scanTemp[0][myLane + 16] : 0;
#pragma unroll
                    for (int offset = 1; offset < 32; offset *= 2)
                    {
                        uint32_t n = WAVE32_SHFL_UP(sum, offset, myLane);
                        if (myLane >= offset)
                            sum += n;
                    }
                    if (myLane < HR_BIN_SQR / 32)
                        s_scanTemp[0][myLane + 16] = sum;
                }
                __syncthreads();

                for (int tileInBin = thrInBlock; tileInBin < HR_BIN_SQR; tileInBin += HR_COARSE_WARPS * 32)
                {
                    uint32_t blockOffset = (tileInBin >= 32) ? s_scanTemp[0][(tileInBin >> 5) + 15] : 0;
                    uint32_t sum = s_tileEmitPrefixSum[tileInBin + 1] + blockOffset;
                    int numEmits = sum >> emitShift;
                    int numAllocs = sum & ((1 << emitShift) - 1);

                    s_tileEmitPrefixSum[tileInBin + 1] = numEmits;
                    s_tileAllocPrefixSum[tileInBin + 1] = numAllocs;

                    if (tileInBin == HR_BIN_SQR - 1 && numAllocs != 0)
                    {
                        int t = atomicAdd(&atomics.numTileSegs, numAllocs);
                        s_firstAllocSeg = (t + numAllocs <= p.maxTileSegs) ? t : 0;
                    }
                }
                __syncthreads();

                int firstAllocSeg = s_firstAllocSeg;
                int totalEmits = s_tileEmitPrefixSum[HR_BIN_SQR];
                int totalAllocs = s_tileAllocPrefixSum[HR_BIN_SQR];

                if (!firstAllocSeg && totalAllocs != 0)
                    break;

                for (int emitInBin = thrInBlock; emitInBin < totalEmits; emitInBin += HR_COARSE_WARPS * 32)
                {
                    uint8_t *tileBase = (uint8_t *)&s_tileEmitPrefixSum[0];
                    uint8_t *tilePtr = tileBase;
                    uint8_t *ptr;

#if (HR_BIN_SQR > 128)
                    ptr = tilePtr + 0x80 * 4;
                    if (emitInBin >= (*(uint32_t *)ptr >> emitShift))
                        tilePtr = ptr;
#endif
#if (HR_BIN_SQR > 64)
                    ptr = tilePtr + 0x40 * 4;
                    if (emitInBin >= (*(uint32_t *)ptr >> emitShift))
                        tilePtr = ptr;
#endif
#if (HR_BIN_SQR > 32)
                    ptr = tilePtr + 0x20 * 4;
                    if (emitInBin >= (*(uint32_t *)ptr >> emitShift))
                        tilePtr = ptr;
#endif
#if (HR_BIN_SQR > 16)
                    ptr = tilePtr + 0x10 * 4;
                    if (emitInBin >= (*(uint32_t *)ptr >> emitShift))
                        tilePtr = ptr;
#endif
#if (HR_BIN_SQR > 8)
                    ptr = tilePtr + 0x08 * 4;
                    if (emitInBin >= (*(uint32_t *)ptr >> emitShift))
                        tilePtr = ptr;
#endif
#if (HR_BIN_SQR > 4)
                    ptr = tilePtr + 0x04 * 4;
                    if (emitInBin >= (*(uint32_t *)ptr >> emitShift))
                        tilePtr = ptr;
#endif
#if (HR_BIN_SQR > 2)
                    ptr = tilePtr + 0x02 * 4;
                    if (emitInBin >= (*(uint32_t *)ptr >> emitShift))
                        tilePtr = ptr;
#endif
#if (HR_BIN_SQR > 1)
                    ptr = tilePtr + 0x01 * 4;
                    if (emitInBin >= (*(uint32_t *)ptr >> emitShift))
                        tilePtr = ptr;
#endif

                    int tileInBin = (tilePtr - tileBase) >> 2;
                    int emitInTile = emitInBin - (*(uint32_t *)tilePtr >> emitShift);

                    int warpStep = (HR_BIN_SQR + 1) * 4;
                    uint8_t *warpBase = (uint8_t *)&s_warpEmitPrefixSum[0][tileInBin] - warpStep;
                    uint8_t *warpPtr = warpBase;

#if (HR_COARSE_WARPS > 8)
                    ptr = warpPtr + 0x08 * warpStep;
                    if (emitInTile >= *(uint32_t *)ptr)
                        warpPtr = ptr;
#endif
#if (HR_COARSE_WARPS > 4)
                    ptr = warpPtr + 0x04 * warpStep;
                    if (emitInTile >= *(uint32_t *)ptr)
                        warpPtr = ptr;
#endif
#if (HR_COARSE_WARPS > 2)
                    ptr = warpPtr + 0x02 * warpStep;
                    if (emitInTile >= *(uint32_t *)ptr)
                        warpPtr = ptr;
#endif
#if (HR_COARSE_WARPS > 1)
                    ptr = warpPtr + 0x01 * warpStep;
                    if (emitInTile >= *(uint32_t *)ptr)
                        warpPtr = ptr;
#endif

                    int warpInTile = (warpPtr - warpBase) / warpStep;

                    uint32_t emitMask = *(uint32_t *)(warpPtr + warpStep + ((uint8_t *)s_warpEmitMask - (uint8_t *)s_warpEmitPrefixSum));
                    int emitInWarp = emitInTile - *(uint32_t *)(warpPtr + warpStep) + __popc(emitMask);

                    int threadInWarp = 0;
                    int pop = __popc(emitMask & 0xFFFF);
                    bool pred = (emitInWarp >= pop);
                    if (pred)
                    {
                        emitInWarp -= pop;
                        emitMask >>= 0x10;
                        threadInWarp += 0x10;
                    }

                    pop = __popc(emitMask & 0xFF);
                    pred = (emitInWarp >= pop);
                    if (pred)
                    {
                        emitInWarp -= pop;
                        emitMask >>= 0x08;
                        threadInWarp += 0x08;
                    }

                    pop = __popc(emitMask & 0xF);
                    pred = (emitInWarp >= pop);
                    if (pred)
                    {
                        emitInWarp -= pop;
                        emitMask >>= 0x04;
                        threadInWarp += 0x04;
                    }

                    pop = __popc(emitMask & 0x3);
                    pred = (emitInWarp >= pop);
                    if (pred)
                    {
                        emitInWarp -= pop;
                        emitMask >>= 0x02;
                        threadInWarp += 0x02;
                    }

                    if (emitInWarp >= (emitMask & 1))
                        threadInWarp++;

                    int currOfs = s_tileStreamCurrOfs[tileInBin];
                    int spaceLeft = -currOfs & (HR_TILE_SEG_SIZE - 1);
                    int outOfs = emitInTile;

                    if (outOfs < spaceLeft)
                        outOfs += currOfs;
                    else
                    {
                        int allocLo = firstAllocSeg + s_tileAllocPrefixSum[tileInBin];
                        outOfs += (allocLo << HR_TILE_SEG_LOG2) - spaceLeft;
                    }

                    int queueIdx = warpInTile * 32 + threadInWarp;
                    int triIdx = s_triQueue[(triQueueReadPos + queueIdx) & (HR_COARSE_QUEUE_SIZE - 1)];
                    tileSegData[outOfs] = triIdx;
                }

                for (int i = HR_COARSE_WARPS * 32 - 1 - thrInBlock; i < totalAllocs; i += HR_COARSE_WARPS * 32)
                {
                    int segIdx = firstAllocSeg + i;
                    tileSegNext[segIdx] = segIdx + 1;
                    tileSegCount[segIdx] = HR_TILE_SEG_SIZE;
                }

                __syncthreads();
                for (int tileInBin = HR_COARSE_WARPS * 32 - 1 - thrInBlock; tileInBin < HR_BIN_SQR; tileInBin += HR_COARSE_WARPS * 32)
                {
                    int oldOfs = s_tileStreamCurrOfs[tileInBin];
                    int newOfs = oldOfs + s_warpEmitPrefixSum[HR_COARSE_WARPS - 1][tileInBin];
                    int allocLo = s_tileAllocPrefixSum[tileInBin];
                    int allocHi = s_tileAllocPrefixSum[tileInBin + 1];

                    if (allocLo != allocHi)
                    {
                        int32_t *nextPtr = &tileSegNext[(oldOfs - 1) >> HR_TILE_SEG_LOG2];

                        // FIX: Prevent array underflow by correctly checking <= 0
                        if (oldOfs <= 0)
                            nextPtr = &tileFirstSeg[binTileIdx + globalTileIdx(tileInBin, p.widthTiles)];
                        *nextPtr = firstAllocSeg + allocLo;

                        newOfs--;
                        newOfs &= HR_TILE_SEG_SIZE - 1;
                        newOfs |= (firstAllocSeg + allocHi - 1) << HR_TILE_SEG_LOG2;
                        newOfs++;
                    }
                    s_tileStreamCurrOfs[tileInBin] = newOfs;
                }

                triQueueReadPos += HR_COARSE_WARPS * 32;
            } while (triQueueReadPos < triQueueWritePos);
        }

        __syncthreads();

        for (int tileInBin_base = 0; tileInBin_base < HR_BIN_SQR; tileInBin_base += HR_COARSE_WARPS * 32)
        {
            int tileInBin = tileInBin_base + thrInBlock;
            bool act = (tileInBin < HR_BIN_SQR);
            uint32_t actMask = WAVE32_BALLOT(act);

            if (act)
            {
                int tileX = tileInBin & (HR_BIN_SIZE - 1);
                int tileY = tileInBin >> HR_BIN_LOG2;
                bool force = (p.deferredClear && tileX <= maxTileXInBin && tileY <= maxTileYInBin);
                int ofs = s_tileStreamCurrOfs[tileInBin];
                int segIdx = (ofs - 1) >> HR_TILE_SEG_LOG2;
                int segCount = ofs & (HR_TILE_SEG_SIZE - 1);

                // FIX: Prevent array underflow by correctly checking > 0
                if (ofs > 0)
                    tileSegNext[segIdx] = -1;
                else if (force)
                {
                    s_tileStreamCurrOfs[tileInBin] = 0;
                    tileFirstSeg[binTileIdx + tileX + tileY * p.widthTiles] = -1;
                }

                if (segCount != 0)
                    tileSegCount[segIdx] = segCount;

                uint32_t res = WAVE32_BALLOT(ofs > 0 || force) & actMask;
                if (threadIdx.x == 0)
                    s_scanTemp[0][(tileInBin >> 5) + 16] = __popc(res);
            }
        }

        __syncthreads();

        if (threadIdx.y == 0)
        {
            uint32_t sum = (myLane < HR_BIN_SQR / 32) ? s_scanTemp[0][myLane + 16] : 0;
#pragma unroll
            for (int offset = 1; offset < 32; offset *= 2)
            {
                uint32_t n = WAVE32_SHFL_UP(sum, offset, myLane);
                if (myLane >= offset)
                    sum += n;
            }
            if (myLane < HR_BIN_SQR / 32)
                s_scanTemp[0][myLane + 16] = sum;
            if (myLane == HR_BIN_SQR / 32 - 1)
                s_firstActiveIdx = atomicAdd(&atomics.numActiveTiles, sum);
        }

        __syncthreads();

        for (int tileInBin_base = 0; tileInBin_base < HR_BIN_SQR; tileInBin_base += HR_COARSE_WARPS * 32)
        {
            int tileInBin = tileInBin_base + thrInBlock;
            bool act = (tileInBin < HR_BIN_SQR) && (s_tileStreamCurrOfs[tileInBin] >= 0);
            uint32_t actMask = WAVE32_BALLOT(act);

            if (act)
            {
                int activeIdx = s_firstActiveIdx;
                activeIdx += (tileInBin >= 32) ? s_scanTemp[0][(tileInBin >> 5) + 15] : 0;

                uint32_t prefixMask = (1u << threadIdx.x) - 1;
                activeIdx += __popc(actMask & prefixMask);
                activeTiles[activeIdx] = binTileIdx + globalTileIdx(tileInBin, p.widthTiles);
            }
        }
    }
}