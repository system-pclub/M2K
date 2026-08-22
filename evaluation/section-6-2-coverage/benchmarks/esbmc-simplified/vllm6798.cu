// vllm6798.cu
// Minimal CUDA example for ESBMC verification
// Demonstrates nullptr crash bug from vLLM PR #6798

#include <stdio.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(expr) expr

// Configuration - Bug scenario from PR #6798
#define K_GLOBAL 8192
#define GROUP_SIZE 8192
#define TP_SIZE 2
#define K_PARTITION (K_GLOBAL / TP_SIZE)
#define NUM_ELEMENTS (K_PARTITION * 128)

// ❌ BUGGY: num_scales = k_partition / group_size = 4096 / 8192 = 0
// This results in nullptr being passed to kernel, causing crash!
#define NUM_SCALES_BUGGY (K_PARTITION / GROUP_SIZE)

// A mock "Marlin" kernel simulating the read of scales
__global__ void mock_marlin_kernel(
    float* output, 
    const float* input, 
    const float* scales)
{
    // Hardcoded parameters to reduce arguments
    const int num_elements = NUM_ELEMENTS;
    const int group_size = GROUP_SIZE;
    const int k_partition = K_PARTITION;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        // Logic: Map the element index to a row, then to a scale index.
        // In the bug scenario, we are processing rows 0..4095.
        // We need the scale for these rows.
        
        // Simulating access:
        // Even if the scale is "global" (index 0), if 'scales' is nullptr, this crashes.
        int scale_idx = 0; 
        
        // ❌ CRASH POINT: If scales was alloc'd with size 0, it is nullptr.
        // BUG: num_scales = 4096 / 8192 = 0 (integer division)
        // This means scales array has size 0 -> nullptr dereference!
        float s = scales[scale_idx]; 
        
        output[idx] = input[idx] * s;
    }
}

int main() {
    // BUG SCENARIO from PR #6798:
    // When loading 'wNa16' (quantized weights) with compressed tensors
    // in Row Parallel mode (Tensor Parallel > 1), the input dimension (K)
    // is split across GPUs.
    //
    // If quantization is 'channelwise', the group_size covers entire K dimension.
    // Bug calculation: size = input_size_per_partition // group_size
    //
    // Example: K=8192, TP=2 => input_size_per_partition = 4096
    //          group_size = 8192
    //          size = 4096 // 8192 = 0 (integer division!)
    //
    // This results in nullptr being passed to kernel, causing CRASH!
    
    // Allocate host memory for ESBMC verification
    float h_input[NUM_ELEMENTS];
    float h_output[NUM_ELEMENTS];
    float h_scales[NUM_SCALES_BUGGY];  // ❌ BUG: SIZE = 0!
    
    // Initialize input
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        h_input[i] = 1.0f;
    }
    
    // Launch kernel with ESBMC_verify_kernel
    int threads = 2;
    int blocks = 2;
    dim3 block(threads);
    dim3 grid(blocks);
    
    // BUG TRIGGER:
    // K_GLOBAL = 8192, TP_SIZE = 2, GROUP_SIZE = 8192
    // k_partition = 8192 / 2 = 4096
    // num_scales = 4096 / 8192 = 0 (integer division)
    // 
    // h_scales has size 0, so it's effectively nullptr
    // Kernel tries to access scales[0], causing nullptr dereference!
    // 
    // This bug occurs in Row Parallel mode with channelwise quantization
    // when partition size is smaller than group size.
    
    ESBMC_verify_kernel(mock_marlin_kernel, grid, block,
        h_output,
        h_input,
        h_scales);  // ❌ BUG: Array with size 0 -> nullptr crash!
    
    return 0;
}
