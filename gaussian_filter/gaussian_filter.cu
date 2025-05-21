#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward declarations of CUDA kernels
template<typename scalar_t>
__global__ void horizontal_conv_forward_kernel(
    const scalar_t* input,
    const scalar_t* kernel,
    scalar_t* output,
    int N, int C, int H, int W,
    int window_size,
    int padding) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (tid >= total) return;

    const int w = tid % W;
    const int h = (tid / W) % H;
    const int c = (tid / (W * H)) % C;
    const int n = tid / (W * H * C);

    scalar_t sum = 0;
    for (int k = 0; k < window_size; ++k) {
        const int wp = w + k - padding;
        if (wp >= 0 && wp < W) {
            const int input_idx = ((n * C + c) * H + h) * W + wp;
            const int kernel_idx = c * window_size + k;
            sum += input[input_idx] * kernel[kernel_idx];
        }
    }
    output[tid] = sum;
}

template<typename scalar_t>
__global__ void vertical_conv_forward_kernel(
    const scalar_t* input,
    const scalar_t* kernel,
    scalar_t* output,
    int N, int C, int H, int W,
    int window_size,
    int padding) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (tid >= total) return;

    const int w = tid % W;
    const int h = (tid / W) % H;
    const int c = (tid / (W * H)) % C;
    const int n = tid / (W * H * C);

    scalar_t sum = 0;
    for (int k = 0; k < window_size; ++k) {
        const int hp = h + k - padding;
        if (hp >= 0 && hp < H) {
            const int input_idx = ((n * C + c) * H + hp) * W + w;
            const int kernel_idx = c * window_size + k;
            sum += input[input_idx] * kernel[kernel_idx];
        }
    }
    output[tid] = sum;
}

template<typename scalar_t>
__global__ void vertical_conv_backward_input_kernel(
    const scalar_t* grad_output,
    const scalar_t* kernel,
    scalar_t* grad_input,
    int N, int C, int H, int W,
    int window_size,
    int padding) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (tid >= total) return;

    const int w = tid % W;
    const int h_in = (tid / W) % H;
    const int c = (tid / (W * H)) % C;
    const int n = tid / (W * H * C);

    scalar_t sum = 0;
    for (int k = 0; k < window_size; ++k) {
        const int h = h_in - k + padding;
        if (h >= 0 && h < H) {
            const int go_idx = ((n * C + c) * H + h) * W + w;
            const int kernel_idx = c * window_size + k;
            sum += grad_output[go_idx] * kernel[kernel_idx];
        }
    }
    atomicAdd(&grad_input[tid], sum);
}

template<typename scalar_t>
__global__ void horizontal_conv_backward_input_kernel(
    const scalar_t* grad_intermediate,
    const scalar_t* kernel,
    scalar_t* grad_input,
    int N, int C, int H, int W,
    int window_size,
    int padding) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (tid >= total) return;

    const int w_in = tid % W;
    const int h = (tid / W) % H;
    const int c = (tid / (W * H)) % C;
    const int n = tid / (W * H * C);

    scalar_t sum = 0;
    for (int k = 0; k < window_size; ++k) {
        const int w = w_in - k + padding;
        if (w >= 0 && w < W) {
            const int gi_idx = ((n * C + c) * H + h) * W + w;
            const int kernel_idx = c * window_size + k;
            sum += grad_intermediate[gi_idx] * kernel[kernel_idx];
        }
    }
    atomicAdd(&grad_input[tid], sum);
}

// Wrapper functions
torch::Tensor horizontal_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& kernel,
    int padding) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(kernel);
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int window_size = kernel.size(3);
    
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "horizontal_conv_forward", ([&] {
        horizontal_conv_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            kernel.data<scalar_t>(),
            output.data<scalar_t>(),
            N, C, H, W, window_size, padding);
    }));
    
    return output;
}

torch::Tensor vertical_conv_forward(
    const torch::Tensor& input,
    const torch::Tensor& kernel,
    int padding) {
    
    CHECK_INPUT(input);
    CHECK_INPUT(kernel);
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int window_size = kernel.size(2);
    
    auto output = torch::zeros_like(input);
    
    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "vertical_conv_forward", ([&] {
        vertical_conv_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            kernel.data<scalar_t>(),
            output.data<scalar_t>(),
            N, C, H, W, window_size, padding);
    }));
    
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> vertical_conv_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& intermediate,
    const torch::Tensor& kernel,
    int padding) {
    
    auto grad_output_cont = grad_output.contiguous();
    auto intermediate_cont = intermediate.contiguous();
    auto kernel_cont = kernel.contiguous();

    CHECK_INPUT(grad_output);
    CHECK_INPUT(intermediate);
    CHECK_INPUT(kernel);

    const int N = intermediate.size(0);
    const int C = intermediate.size(1);
    const int H = intermediate.size(2);
    const int W = intermediate.size(3);
    const int window_size = kernel.size(2);

    auto grad_input = torch::zeros_like(intermediate);

    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(intermediate_cont.type(), "vertical_conv_backward_input", ([&] {
        vertical_conv_backward_input_kernel<scalar_t><<<blocks, threads>>>(
            grad_output_cont.data<scalar_t>(),  
            kernel_cont.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            N, C, H, W, window_size, padding);
    }));

    return {grad_input, torch::Tensor()};
}

std::tuple<torch::Tensor, torch::Tensor> horizontal_conv_backward(
    const torch::Tensor& grad_intermediate,
    const torch::Tensor& input,
    const torch::Tensor& kernel,
    int padding) {
    
    auto grad_intermediate_cont = grad_intermediate.contiguous();
    auto input_cont = input.contiguous();
    auto kernel_cont = kernel.contiguous();

    CHECK_INPUT(grad_intermediate);
    CHECK_INPUT(input);
    CHECK_INPUT(kernel);

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int window_size = kernel.size(3);

    auto grad_input = torch::zeros_like(input);

    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input_cont.type(), "horizontal_conv_backward_input", ([&] {
        horizontal_conv_backward_input_kernel<scalar_t><<<blocks, threads>>>(
            grad_intermediate_cont.data<scalar_t>(),  
            kernel_cont.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            N, C, H, W, window_size, padding);
    }));

    return {grad_input, torch::Tensor()};
}


class GaussianFilterFunction : public torch::autograd::Function<GaussianFilterFunction> {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const torch::Tensor& x,
            const torch::Tensor& window_1d,
            bool use_padding) {
            
            int padding = 0;
            if (use_padding) {
                int window_size = window_1d.size(3);
                padding = window_size / 2;
            }
            
            // Forward pass
            auto intermediate = horizontal_conv_forward(x, window_1d, padding);
            auto vertical_kernel = window_1d.transpose(2, 3);
            auto output = vertical_conv_forward(intermediate, vertical_kernel, padding);
            
            ctx->save_for_backward({x, window_1d, intermediate.contiguous()});  
            ctx->saved_data["padding"] = padding;  
            
            return output;
        }
    
        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
            
            auto saved = ctx->get_saved_variables();
            auto x = saved[0].contiguous();                  
            auto window_1d = saved[1].contiguous();         
            auto intermediate = saved[2].contiguous();       
            int padding = ctx->saved_data["padding"].toInt();
            
            auto grad_output = grad_outputs[0].contiguous(); 
            
            auto vertical_kernel = window_1d.transpose(2, 3).contiguous();  
            auto [grad_intermediate, _] = vertical_conv_backward(
                grad_output, 
                intermediate, 
                vertical_kernel, 
                padding
            );
    
            auto [grad_x, __] = horizontal_conv_backward(
                grad_intermediate.contiguous(),   
                x.contiguous(),                    
                window_1d.contiguous(),            
                padding
            );

            return {grad_x, torch::Tensor(), torch::Tensor()}; 
        }
    };

torch::Tensor gaussian_filter(
    const torch::Tensor& x,
    const torch::Tensor& window_1d,
    bool use_padding) {
    return GaussianFilterFunction::apply(x, window_1d, use_padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_filter", &gaussian_filter, "Gaussian filter CUDA implementation");
}