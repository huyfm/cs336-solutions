import os

import torch
from torch.utils.cpp_extension import load_inline


def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)


def build_cuda_gelu():
    # ✅ CUDA kernel (as string)
    cuda_gelu_src = """
    #include <math.h>
    #include <torch/extension.h>
    #include <c10/cuda/CUDAException.h>

    __global__ void gelu_kernel(float* in, float* out, int num_elements) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < num_elements) {
            out[i] = 0.5f * in[i] * (1.0f + tanhf(0.79788456f * (in[i] + 0.044715f * in[i] * in[i] * in[i])));
        }
    }

    inline unsigned int cdiv(unsigned int a, unsigned int b) {
        return (a + b - 1) / b;
    }

    torch::Tensor gelu(torch::Tensor x) {
        TORCH_CHECK(x.device().is_cuda());
        TORCH_CHECK(x.is_contiguous());

        auto y = torch::empty_like(x);
        int num_elements = x.numel();
        int block_size = 256;
        int num_blocks = cdiv(num_elements, block_size);

        gelu_kernel<<<num_blocks, block_size>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            num_elements
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return y;
    }
    """

    # ✅ C++ header declaration
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    ensure_directory_exists("var/cuda_gelu")

    if not torch.cuda.is_available():
        return None

    module = load_inline(
        name="inline_gelu",
        cpp_sources=[cpp_gelu_src],
        cuda_sources=[cuda_gelu_src],
        functions=["gelu"],
        build_directory="var/cuda_gelu",
        extra_cflags=["-O2"],
        verbose=True,
    )

    return getattr(module, "gelu")


# Build & load
build_cuda_gelu()
