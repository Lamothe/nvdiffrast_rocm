#pragma once

#include <hip/hip_runtime.h>
#undef warpSize
#ifndef NVDR_FRAMEWORK_H
#define NVDR_FRAMEWORK_H

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>


#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)

    // Map CUDA reciprocal intrinsics to standard division for ROCm
    #ifndef __frcp_rz
    #define __frcp_rz(x) (1.0f / (x))
    #endif

    #ifndef __frcp_rn
    #define __frcp_rn(x) (1.0f / (x))
    #endif

#endif

namespace at {
    namespace cuda {
        static inline int current_device() {
            int dev = 0;
            (void)hipGetDevice(&dev);
            return dev;
        }

        static inline bool check_device(at::ArrayRef<at::Tensor> tensors) {
            if (tensors.empty()) return true;
            auto device = tensors[0].device();
            for (const auto& t : tensors) {
                if (t.device() != device) return false;
            }
            return true;
        }
    }
}

// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
// ... (Standard License Header) ...

#ifdef NVDR_TORCH
    #ifndef __HIPCC__
        #if !defined(USE_ROCM) && !defined(__HIP_PLATFORM_AMD__)
            #include <ATen/hip/HIPContext.h>
            #include <ATen/hip/HIPUtils.h>
            #include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
        #endif
        #include <pybind11/numpy.h>
    #endif

    #define NVDR_CHECK(COND, ERR) do { TORCH_CHECK(COND, ERR) } while(0)
    #define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { hipError_t err = CUDA_CALL; TORCH_CHECK(!err, "Cuda error: ", hipGetLastError(), "[", #CUDA_CALL, ";]"); } while(0)
#endif

#endif // NVDR_FRAMEWORK_H
