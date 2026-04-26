#pragma once
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <vector>

// --- Macros ---
#define NVDR_CHECK(CONDITION, ...) TORCH_CHECK((CONDITION), __VA_ARGS__)

#ifndef NVDR_CHECK_CUDA_ERROR
#define NVDR_CHECK_CUDA_ERROR(VAL) {     hipError_t err = (VAL);     if (err != hipSuccess) {         TORCH_CHECK(false, "HIP Error: ", hipGetErrorString(err));     } }
#endif

// --- Helper Functions (Variadic recursive checks) ---

// 1. Device Check
inline void check_device_impl(const char* func_name, const torch::Tensor& t) {
    NVDR_CHECK(t.is_cuda(), func_name, "(): Tensor must be on GPU");
}
template <typename... Args>
inline void check_device_impl(const char* func_name, const torch::Tensor& t, Args... args) {
    check_device_impl(func_name, t);
    check_device_impl(func_name, args...);
}
#define NVDR_CHECK_DEVICE(...) check_device_impl(__func__, __VA_ARGS__)

// 2. Contiguous Check
inline void check_contiguous_impl(const char* func_name, const torch::Tensor& t) {
    NVDR_CHECK(t.is_contiguous(), func_name, "(): Tensor must be contiguous");
}
template <typename... Args>
inline void check_contiguous_impl(const char* func_name, const torch::Tensor& t, Args... args) {
    check_contiguous_impl(func_name, t);
    check_contiguous_impl(func_name, args...);
}
#define NVDR_CHECK_CONTIGUOUS(...) check_contiguous_impl(__func__, __VA_ARGS__)

// 3. Float32 Check
inline void check_f32_impl(const char* func_name, const torch::Tensor& t) {
    NVDR_CHECK(t.dtype() == torch::kFloat32, func_name, "(): Tensor must be float32");
}
template <typename... Args>
inline void check_f32_impl(const char* func_name, const torch::Tensor& t, Args... args) {
    check_f32_impl(func_name, t);
    check_f32_impl(func_name, args...);
}
#define NVDR_CHECK_F32(...) check_f32_impl(__func__, __VA_ARGS__)

// 4. Int32 Check
inline void check_i32_impl(const char* func_name, const torch::Tensor& t) {
    NVDR_CHECK(t.dtype() == torch::kInt32, func_name, "(): Tensor must be int32");
}
template <typename... Args>
inline void check_i32_impl(const char* func_name, const torch::Tensor& t, Args... args) {
    check_i32_impl(func_name, t);
    check_i32_impl(func_name, args...);
}
#define NVDR_CHECK_I32(...) check_i32_impl(__func__, __VA_ARGS__)

inline void nvdr_check_contiguous(const torch::Tensor& t, const char* func_name, const char* msg) {
    NVDR_CHECK(t.is_contiguous(), func_name, msg);
}

inline void nvdr_check_contiguous(const std::vector<torch::Tensor>& tensors, const char* func_name, const char* msg) {
    for (const auto& tensor : tensors) {
        nvdr_check_contiguous(tensor, func_name, msg);
    }
}

inline void nvdr_check_f32(const torch::Tensor& t, const char* func_name, const char* msg) {
    NVDR_CHECK(t.dtype() == torch::kFloat32, func_name, msg);
}

inline void nvdr_check_f32(const std::vector<torch::Tensor>& tensors, const char* func_name, const char* msg) {
    for (const auto& tensor : tensors) {
        nvdr_check_f32(tensor, func_name, msg);
    }
}

inline void nvdr_check_cpu(const torch::Tensor& t, const char* func_name, const char* msg) {
    NVDR_CHECK(t.is_cpu(), func_name, msg);
}

// Macro definition (if not already present)
#ifndef NVDR_CHECK_CPU
#define NVDR_CHECK_CPU(x) nvdr_check_cpu(x, __func__, "(): " #x " must be a CPU tensor")
#endif