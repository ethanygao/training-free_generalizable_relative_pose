from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gaussian_filter_cuda",
    ext_modules=[
        CUDAExtension(
            "gaussian_filter_cuda",
            ["gaussian_filter.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)