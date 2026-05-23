#include "hip/hip_runtime.h"
// Copyright (c) 2009-2022, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "../HipRaster.hpp"
#include "PrivateDefs.hpp"
#include "Constants.hpp"
#include "Util.inl"

namespace HR
{

//------------------------------------------------------------------------
// Stage implementations.
//------------------------------------------------------------------------

#include "TriangleSetup.inl"
#include "BinRaster.inl"
#include "CoarseRaster.inl"
#include "FineRaster.inl"

}

//------------------------------------------------------------------------
// Stage entry points.
//------------------------------------------------------------------------

// FIX: Change * 32 to * 64 for AMD 64-lane wavefronts
__global__ void __launch_bounds__(HR_SETUP_WARPS * 64, HR_SETUP_OPT_BLOCKS)  triangleSetupKernel (const HR::HRParams p)  { HR::triangleSetupImpl(p); }
__global__ void __launch_bounds__(HR_BIN_WARPS * 64, 1)                      binRasterKernel     (const HR::HRParams p)  { HR::binRasterImpl(p); }
__global__ void __launch_bounds__(HR_COARSE_WARPS * 64, 1)                   coarseRasterKernel  (const HR::HRParams p)  { HR::coarseRasterImpl(p); }
__global__ void __launch_bounds__(HR_FINE_MAX_WARPS * 64, 1)                 fineRasterKernel    (const HR::HRParams p)  { HR::fineRasterImpl(p); }

//------------------------------------------------------------------------
