from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="cscr.attention",
            sources=[
                "cscr/attention.cpp",
                "cscr/attention_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
