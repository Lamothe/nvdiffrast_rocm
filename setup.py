
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Force clean compile flags
extra_compile_args = {'cxx': ['-std=c++17', '-w'], 'nvcc': ['-std=c++17', '-w']}

sources = [
    'csrc/common/common.cpp',
    'csrc/common/texture.cpp',
    'csrc/common/rasterize.hip',
    'csrc/common/antialias.hip',
    'csrc/common/interpolate.hip',
    'csrc/common/hipraster/impl/Buffer.cpp',    
    'csrc/common/hipraster/impl/CudaRaster.cpp',    
    'csrc/common/hipraster/impl/RasterImpl.cpp',    
    'csrc/common/hipraster/impl/RasterImpl_kernel.hip',
    'csrc/common/texture_kernel.hip',
    'csrc/torch/torch_rasterize.cpp',
    'csrc/torch/torch_interpolate.cpp',
    'csrc/torch/torch_texture.cpp',
    'csrc/torch/torch_antialias.cpp',
    'csrc/torch/torch_bindings.cpp',
]

# Filter to existing files only
real_sources = [s for s in sources if os.path.exists(s)]

setup(
    name='nvdiffrast',
    version='0.4.0',
    description='nvdiffrast - redistributable reference implementation',
    packages=['nvdiffrast', 'nvdiffrast.torch', 'nvdiffrast.opengl'],
    package_dir={'nvdiffrast': 'nvdiffrast'},
    ext_modules=[
        CUDAExtension(
            name='_nvdiffrast_c',
            sources=real_sources,
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
