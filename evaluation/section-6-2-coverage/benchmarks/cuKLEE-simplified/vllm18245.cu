// marlin_act_order_buggy.cu
//
// Minimal buggy example inspired by vLLM PR #18245:
//
// [Bugfix] fix `an illegal memory access was encountered` of marlin kernel + act_order
//
// Bug pattern:
//   - act_order path computes a slice index `slice_k_start`
//   - sometimes `slice_k_start >= prob_k`
//   - kernel still does `g_idx[slice_k_start]` → OOB global read → illegal memory access.
//
// This file reproduces that pattern in a tiny kernel.

#include <cstdio>
#include <cuda_runtime.h>

// Buggy Marlin-like kernel with realistic structure
// This simulates the real Marlin kernel's act_order processing loop
// where slice_k_start can exceed prob_k, causing OOB access
__global__ void marlin_act_order_kernel_buggy(
    const int* __restrict__ g_idx,
    const float* __restrict__ scales,
    float* __restrict__ output,
    int prob_k,
    int prob_n,
    int tb_k,
    int stages,
    int slice_k_start_base,
    int num_iters)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // Simulate Marlin's main processing loop structure
    // Real Marlin processes multiple slices/tiles
    for (int iter = 0; iter < num_iters; iter++) {
        // Calculate slice starting position for this iteration
        // This mimics how real Marlin advances through K dimension
        int slice_k_start = slice_k_start_base + iter * tb_k;
        
        // Same pattern as the real kernel:
        //   slice_k_start += tb_k * stages;
        slice_k_start += tb_k * stages;
        
        // Compute last_g_idx with clamp (this part is safe)
        int last_g_idx = slice_k_start + stages * tb_k * 2;
        if (last_g_idx >= prob_k) {
            last_g_idx = prob_k - 1;
        }
        
        // ❌ BUG:
        // slice_k_start can be >= prob_k, but we still index g_idx[slice_k_start].
        // This will be an out-of-bounds global memory read when that happens.
        int first_group_id = g_idx[slice_k_start];  // <-- OOB read if slice_k_start >= prob_k
        int last_group_id  = g_idx[last_g_idx];     // this one is clamped and safe
        
        // Simulate scale loading loop (typical in quantization kernels)
        // Real Marlin loads scales for dequantization based on group_id
        int num_groups = last_group_id - first_group_id + 1;
        for (int g = 0; g < num_groups; g++) {
            int group_idx = first_group_id + g;
            if (group_idx < prob_n && tid < 32) {  // warp-level processing
                // Load scale for this group
                float scale = scales[group_idx * 32 + tid];
                
                // Simulate some computation with the scale
                // Real Marlin uses this for dequantizing weights
                float result = scale * 127.0f;  // typical int8 dequant
                
                // Write to output (bounds checked to avoid secondary issues)
                int out_idx = (bid * num_iters * 32) + (iter * 32) + tid;
                if (out_idx < prob_n * 32) {
                    output[out_idx] = result;
                }
            }
        }
        
        // Synchronize threads in block (typical in tiled kernels)
        __syncthreads();
        
        // Simulate inner tile processing loop
        // Real Marlin processes 16x16 or similar tiles
        for (int tile = 0; tile < stages; tile++) {
            int k_idx = slice_k_start + tile * tb_k + tid;
            
            // Check if we're still in bounds for this tile
            // (but the damage is already done from the OOB g_idx access above)
            if (k_idx < prob_k) {
                // Simulate weight processing
                // Real Marlin loads and dequantizes weight tiles here
                int weight_group = g_idx[k_idx];
                float scale_val = scales[weight_group];
                
                // More computation...
                float tmp = scale_val * 2.0f;
                if (tid == 0) {
                    output[bid] += tmp;
                }
            }
        }
    }
}

int main() {
    // We choose a small prob_k, but force slice_k_start >= prob_k via tb_k * stages.
    const int prob_k            = 16;   // length of g_idx
    const int prob_n            = 32;   // output dimension
    const int tb_k              = 4;
    const int stages            = 4;
    const int slice_k_start_base = 16;  // base already at the end
    const int num_iters         = 2;    // number of processing iterations

    // slice_k_start = 16 + 4 * 4 = 32  >= prob_k (16)
    const int computed_slice_k_start = slice_k_start_base + tb_k * stages;

    printf("=== Marlin Act Order Bug Reproduction ===\n");
    printf("prob_k = %d (size of g_idx array)\n", prob_k);
    printf("slice_k_start_base = %d\n", slice_k_start_base);
    printf("tb_k = %d, stages = %d\n", tb_k, stages);
    printf("computed slice_k_start = %d + %d * %d = %d\n",
           slice_k_start_base, tb_k, stages, computed_slice_k_start);
    printf("BUG: slice_k_start (%d) >= prob_k (%d)\n", computed_slice_k_start, prob_k);
    printf("This causes OOB access: g_idx[%d] when array size is %d\n\n",
           computed_slice_k_start, prob_k);

    // Allocate and initialize g_idx[prob_k] on host.
    int* h_g_idx = new int[prob_k];
    for (int i = 0; i < prob_k; ++i) {
        h_g_idx[i] = i;  // simple pattern
    }

    // Allocate scales array (used in realistic loop)
    float* h_scales = new float[prob_n * 32];
    for (int i = 0; i < prob_n * 32; i++) {
        h_scales[i] = 0.007874f;  // typical int8 scale
    }

    // Allocate output array
    float* h_output = new float[prob_n * 32 * num_iters];
    for (int i = 0; i < prob_n * 32 * num_iters; i++) {
        h_output[i] = 0.0f;
    }

    // Copy to device.
    int* d_g_idx = nullptr;
    float* d_scales = nullptr;
    float* d_output = nullptr;
    
    cudaMalloc(&d_g_idx, prob_k * sizeof(int));
    cudaMalloc(&d_scales, prob_n * 32 * sizeof(float));
    cudaMalloc(&d_output, prob_n * 32 * num_iters * sizeof(float));
    
    cudaMemcpy(d_g_idx, h_g_idx, prob_k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales, prob_n * 32 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, prob_n * 32 * num_iters * sizeof(float), cudaMemcpyHostToDevice);

    // Launch with realistic thread block size (warp = 32 threads)
    dim3 grid(1);
    dim3 block(32);  // typical warp size

    printf("Launching kernel...\n");
    marlin_act_order_kernel_buggy<<<grid, block>>>(
        d_g_idx,
        d_scales,
        d_output,
        prob_k,
        prob_n,
        tb_k,
        stages,
        slice_k_start_base,
        num_iters);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("Expected: 'an illegal memory access was encountered'\n");
    } else {
        printf("Kernel completed (unexpected - bug may not have triggered)\n");
    }

    // Cleanup
    cudaFree(d_g_idx);
    cudaFree(d_scales);
    cudaFree(d_output);
    delete[] h_g_idx;
    delete[] h_scales;
    delete[] h_output;
    
    return 0;
}
