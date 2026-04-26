// BinRaster.inl
// Native HIP implementation for nvdiffrast macro-binning

#define WAVE32_BALLOT(pred) ((uint32_t)(__ballot(pred) >> ((__lane_id() >> 5) << 5)))
#define WAVE32_ANY(pred) (WAVE32_BALLOT(pred) != 0)
#define WAVE32_ALL(pred) (WAVE32_BALLOT(pred) == 0xFFFFFFFFu)

__device__ __forceinline__ void binRasterImpl(const HRParams p)
{
    __shared__ uint32_t s_broadcast[HR_BIN_WARPS + 16];
    __shared__ int32_t s_outOfs[HR_MAXBINS_SQR];
    __shared__ int32_t s_outTotal[HR_MAXBINS_SQR];
    __shared__ int32_t s_overIndex[HR_MAXBINS_SQR];

    __shared__ uint32_t s_outMask[HR_BIN_WARPS][HR_MAXBINS_SQR + 1];
    __shared__ uint32_t s_outCount[HR_BIN_WARPS][HR_MAXBINS_SQR + 1];
    __shared__ int32_t s_triBuf[HR_BIN_WARPS * 32 * 4];

    __shared__ uint32_t s_batchPos;
    __shared__ uint32_t s_bufCount;
    __shared__ uint32_t s_overTotal;
    __shared__ uint32_t s_allocBase;

    const HRImageParams &ip = getImageParams(p, blockIdx.z);
    HRAtomics &atomics = p.atomics[blockIdx.z];
    const uint8_t *triSubtris = (const uint8_t *)p.triSubtris + p.maxSubtris * blockIdx.z;
    const HRTriangleHeader *triHeader = (const HRTriangleHeader *)p.triHeader + p.maxSubtris * blockIdx.z;
    int32_t *binFirstSeg = (int32_t *)p.binFirstSeg + HR_MAXBINS_SQR * HR_BIN_STREAMS_SIZE * blockIdx.z;
    int32_t *binTotal = (int32_t *)p.binTotal + HR_MAXBINS_SQR * HR_BIN_STREAMS_SIZE * blockIdx.z;
    int32_t *binSegData = (int32_t *)p.binSegData + p.maxBinSegs * HR_BIN_SEG_SIZE * blockIdx.z;
    int32_t *binSegNext = (int32_t *)p.binSegNext + p.maxBinSegs * blockIdx.z;
    int32_t *binSegCount = (int32_t *)p.binSegCount + p.maxBinSegs * blockIdx.z;

    if (atomics.numSubtris > p.maxSubtris)
        return;

    int thrInBlock = threadIdx.x + threadIdx.y * 32;
    int batchPos = 0;

    if (thrInBlock < 16)
        s_broadcast[thrInBlock] = 0;

    if (thrInBlock < p.numBins)
    {
        binFirstSeg[(thrInBlock << HR_BIN_STREAMS_LOG2) + blockIdx.x] = -1;
        s_outOfs[thrInBlock] = -HR_BIN_SEG_SIZE;
        s_outTotal[thrInBlock] = 0;
    }

    for (;;)
    {
        if (thrInBlock == 0)
            s_batchPos = atomicAdd(&atomics.binCounter, ip.binBatchSize);
        __syncthreads();
        batchPos = s_batchPos;

        if (batchPos >= ip.triCount)
            break;

        int bufIndex = 0;
        int bufCount = 0;
        int batchEnd = ::min(batchPos + ip.binBatchSize, ip.triCount);

        do
        {
            while (bufCount < HR_BIN_WARPS * 32 && batchPos < batchEnd)
            {
                int triIdx = batchPos + thrInBlock;
                uint32_t num = 0;
                if (triIdx < batchEnd)
                    num = triSubtris[triIdx];

                uint32_t sum = num;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2)
                {
                    uint32_t neighbor = __shfl_up(sum, offset);
                    if (threadIdx.x >= offset)
                        sum += neighbor;
                }

                uint32_t myIdx = sum - num;
                uint32_t warpTotal = __shfl(sum, __lane_id() - threadIdx.x + 31);

                if (threadIdx.x == 31)
                    s_broadcast[threadIdx.y + 16] = warpTotal;
                __syncthreads();

                if (threadIdx.y == 0)
                {
                    uint32_t val = (threadIdx.x < HR_BIN_WARPS) ? s_broadcast[threadIdx.x + 16] : 0;
                    uint32_t valSum = val;
#pragma unroll
                    for (int offset = 1; offset < 32; offset *= 2)
                    {
                        uint32_t neighbor = __shfl_up(valSum, offset);
                        if (threadIdx.x >= offset)
                            valSum += neighbor;
                    }
                    if (threadIdx.x < HR_BIN_WARPS)
                        s_broadcast[threadIdx.x + 16] = valSum - val;
                    if (threadIdx.x == HR_BIN_WARPS - 1)
                    {
                        s_batchPos = batchPos + HR_BIN_WARPS * 32;
                        s_bufCount = bufCount + valSum;
                    }
                }
                __syncthreads();

                if (num)
                {
                    uint32_t pos = bufCount + myIdx + s_broadcast[threadIdx.y + 16];
                    if (pos + num <= HR_ARRAY_SIZE(s_triBuf))
                    {
                        pos += bufIndex;
                        pos &= HR_ARRAY_SIZE(s_triBuf) - 1;
                        if (num == 1)
                        {
                            s_triBuf[pos] = triIdx * 8 + 7;
                        }
                        else
                        {
                            for (int i = 0; i < num; i++)
                            {
                                s_triBuf[pos] = triIdx * 8 + i;
                                pos++;
                                pos &= HR_ARRAY_SIZE(s_triBuf) - 1;
                            }
                        }
                    }
                    else if (pos <= HR_ARRAY_SIZE(s_triBuf))
                    {
                        s_batchPos = batchPos + thrInBlock;
                        s_bufCount = pos;
                    }
                }
                __syncthreads();
                batchPos = s_batchPos;
                bufCount = s_bufCount;
            }

            for (int i = threadIdx.x; i < p.numBins; i += 32)
                s_outMask[threadIdx.y][i] = 0;
            __syncthreads();

            uint4 triData = make_uint4(0, 0, 0, 0);
            if (thrInBlock < bufCount)
            {
                uint32_t triPos = bufIndex + thrInBlock;
                triPos &= HR_ARRAY_SIZE(s_triBuf) - 1;

                int triIdx = s_triBuf[triPos];
                int dataIdx = triIdx >> 3;
                int subtriIdx = triIdx & 7;
                if (subtriIdx != 7)
                    dataIdx = triHeader[dataIdx].misc + subtriIdx;
                triData = *(((const uint4 *)triHeader) + dataIdx);
            }

            int32_t lox, loy, hix, hiy;
            bool hasTri = (thrInBlock < bufCount);

            if (hasTri)
            {
                int32_t v0x = add_s16lo_s16lo(triData.x, p.widthPixelsVp * (HR_SUBPIXEL_SIZE >> 1));
                int32_t v0y = add_s16hi_s16lo(triData.x, p.heightPixelsVp * (HR_SUBPIXEL_SIZE >> 1));
                int32_t d01x = sub_s16lo_s16lo(triData.y, triData.x);
                int32_t d01y = sub_s16hi_s16hi(triData.y, triData.x);
                int32_t d02x = sub_s16lo_s16lo(triData.z, triData.x);
                int32_t d02y = sub_s16hi_s16hi(triData.z, triData.x);
                int binLog = HR_BIN_LOG2 + HR_TILE_LOG2 + HR_SUBPIXEL_LOG2;

                lox = add_clamp_0_x((v0x + min_min(d01x, 0, d02x)) >> binLog, 0, p.widthBins - 1);
                loy = add_clamp_0_x((v0y + min_min(d01y, 0, d02y)) >> binLog, 0, p.heightBins - 1);
                hix = add_clamp_0_x((v0x + max_max(d01x, 0, d02x)) >> binLog, 0, p.widthBins - 1);
                hiy = add_clamp_0_x((v0y + max_max(d01y, 0, d02y)) >> binLog, 0, p.heightBins - 1);

                uint32_t bit = 1u << threadIdx.x;
                bool complex = (hix > lox + 1 || hiy > loy + 1);
                if (!WAVE32_ANY(complex))
                {
                    int binIdx = lox + p.widthBins * loy;
                    atomicOr(&s_outMask[threadIdx.y][binIdx], bit);
                    if (hix > lox)
                        atomicOr(&s_outMask[threadIdx.y][binIdx + 1], bit);
                    if (hiy > loy)
                        atomicOr(&s_outMask[threadIdx.y][binIdx + p.widthBins], bit);
                    if (hix > lox && hiy > loy)
                        atomicOr(&s_outMask[threadIdx.y][binIdx + p.widthBins + 1], bit);
                }
                else
                {
                    int32_t d12x = d02x - d01x, d12y = d02y - d01y;
                    v0x -= lox << binLog;
                    v0y -= loy << binLog;

                    int32_t t01 = v0x * d01y - v0y * d01x;
                    int32_t t02 = v0y * d02x - v0x * d02y;
                    int32_t t12 = d01x * d12y - d01y * d12x - t01 - t02;
                    int32_t b01 = add_sub(t01 >> binLog, ::max(d01x, 0), ::min(d01y, 0));
                    int32_t b02 = add_sub(t02 >> binLog, ::max(d02y, 0), ::min(d02x, 0));
                    int32_t b12 = add_sub(t12 >> binLog, ::max(d12x, 0), ::min(d12y, 0));

                    int width = hix - lox + 1;
                    d01x += width * d01y;
                    d02x += width * d02y;
                    d12x += width * d12y;

                    uint8_t *currPtr = (uint8_t *)&s_outMask[threadIdx.y][lox + loy * p.widthBins];
                    uint8_t *skipPtr = (uint8_t *)&s_outMask[threadIdx.y][(hix + 1) + loy * p.widthBins];
                    uint8_t *endPtr = (uint8_t *)&s_outMask[threadIdx.y][lox + (hiy + 1) * p.widthBins];
                    int stride = p.widthBins * 4;
                    int ptrYInc = stride - width * 4;

                    do
                    {
                        if (b01 >= 0 && b02 >= 0 && b12 >= 0)
                            atomicOr((uint32_t *)currPtr, bit);
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
                            skipPtr += stride;
                        }
                    } while (currPtr != endPtr);
                }
            }

            if (thrInBlock == 0)
                s_overTotal = 0;
            __syncthreads();

            int overIndex = -1;
            bool act = (thrInBlock < p.numBins);
            if (act)
            {
                uint8_t *srcPtr = (uint8_t *)&s_outMask[0][thrInBlock];
                uint8_t *dstPtr = (uint8_t *)&s_outCount[0][thrInBlock];
                int total = 0;
                for (int i = 0; i < HR_BIN_WARPS; i++)
                {
                    total += __popc(*(uint32_t *)srcPtr);
                    *(uint32_t *)dstPtr = total;
                    srcPtr += (HR_MAXBINS_SQR + 1) * 4;
                    dstPtr += (HR_MAXBINS_SQR + 1) * 4;
                }

                int ofs = s_outOfs[thrInBlock];
                bool ovr = (((ofs - 1) >> HR_BIN_SEG_LOG2) != (((ofs - 1) + total) >> HR_BIN_SEG_LOG2));

                uint32_t ovrMask = WAVE32_BALLOT(ovr);

                if (ovrMask != 0)
                {
                    uint32_t prefixMask = (1u << threadIdx.x) - 1;
                    if (ovr)
                        overIndex = __popc(ovrMask & prefixMask);

                    uint32_t warpOverTotal = __popc(ovrMask);
                    uint32_t baseOver = 0;

                    int firstOvrLane = __ffs(ovrMask) - 1;
                    if (threadIdx.x == firstOvrLane)
                    {
                        baseOver = atomicAdd(&s_overTotal, warpOverTotal);
                    }

                    baseOver = __shfl(baseOver, __lane_id() - threadIdx.x + firstOvrLane);
                    if (ovr)
                    {
                        overIndex += baseOver;
                        s_overIndex[thrInBlock] = overIndex;
                    }
                }
            }
            __syncthreads();

            uint32_t overTotal = s_overTotal;
            uint32_t allocBase = 0;
            if (overTotal > 0)
            {
                if (thrInBlock == 0)
                {
                    uint32_t base = atomicAdd(&atomics.numBinSegs, overTotal);
                    s_allocBase = (base + overTotal <= p.maxBinSegs) ? base : 0;
                }
                __syncthreads();
                allocBase = s_allocBase;

                if (overIndex != -1)
                {
                    int segIdx = allocBase + overIndex;

                    // FIX: Prevent array underflow by correctly checking <= 0
                    if (s_outOfs[thrInBlock] <= 0)
                        binFirstSeg[(thrInBlock << HR_BIN_STREAMS_LOG2) + blockIdx.x] = segIdx;
                    else
                        binSegNext[(s_outOfs[thrInBlock] - 1) >> HR_BIN_SEG_LOG2] = segIdx;

                    binSegNext[segIdx] = -1;
                    binSegCount[segIdx] = HR_BIN_SEG_SIZE;
                }
            }

            if (thrInBlock < bufCount)
            {
                int triPos = (bufIndex + thrInBlock) & (HR_ARRAY_SIZE(s_triBuf) - 1);
                int currBin = lox + loy * p.widthBins;
                int skipBin = (hix + 1) + loy * p.widthBins;
                int endBin = lox + (hiy + 1) * p.widthBins;
                int binYInc = p.widthBins - (hix - lox + 1);

                do
                {
                    uint32_t outMask = s_outMask[threadIdx.y][currBin];
                    if (outMask & (1u << threadIdx.x))
                    {
                        uint32_t prefixMask = (1u << threadIdx.x) - 1;
                        int idx = __popc(outMask & prefixMask);
                        if (threadIdx.y > 0)
                            idx += s_outCount[threadIdx.y - 1][currBin];

                        int base = s_outOfs[currBin];
                        int free = (-base) & (HR_BIN_SEG_SIZE - 1);

                        if (idx >= free)
                            idx += ((allocBase + s_overIndex[currBin]) << HR_BIN_SEG_LOG2) - free;
                        else
                            idx += base;

                        binSegData[idx] = s_triBuf[triPos];
                    }
                    currBin++;
                    if (currBin == skipBin)
                    {
                        currBin += binYInc;
                        skipBin += p.widthBins;
                    }
                } while (currBin != endBin);
            }

            __syncthreads();

            if (thrInBlock < p.numBins)
            {
                uint32_t total = s_outCount[HR_BIN_WARPS - 1][thrInBlock];
                uint32_t oldOfs = s_outOfs[thrInBlock];
                if (overIndex == -1)
                    s_outOfs[thrInBlock] = oldOfs + total;
                else
                {
                    int addr = oldOfs + total;
                    addr = ((addr - 1) & (HR_BIN_SEG_SIZE - 1)) + 1;
                    addr += (allocBase + overIndex) << HR_BIN_SEG_LOG2;
                    s_outOfs[thrInBlock] = addr;
                }
                s_outTotal[thrInBlock] += total;
            }

            int count = ::min(bufCount, HR_BIN_WARPS * 32);
            bufCount -= count;
            bufIndex += count;
            bufIndex &= HR_ARRAY_SIZE(s_triBuf) - 1;
        } while (bufCount > 0 || batchPos < batchEnd);

        if (thrInBlock < p.numBins)
        {
            int ofs = s_outOfs[thrInBlock];
            if (ofs & (HR_BIN_SEG_SIZE - 1))
            {
                int seg = ofs >> HR_BIN_SEG_LOG2;
                binSegCount[seg] = ofs & (HR_BIN_SEG_SIZE - 1);
                s_outOfs[thrInBlock] = (ofs + HR_BIN_SEG_SIZE - 1) & -HR_BIN_SEG_SIZE;
            }
        }
    }

    if (thrInBlock < p.numBins)
        binTotal[(thrInBlock << HR_BIN_STREAMS_LOG2) + blockIdx.x] = s_outTotal[thrInBlock];
}