import os
from pathlib import Path


original_results = {"68": "cuKLEE: Bug Detected: attention_kernels.cu:159: out of bound pointer",
                    "1295": ["cuKLEE: Bug Detected: gemm_kernels.cu:93: integer overflow", "cuKLEE: Bug Detected: gemm_kernels.cu:291: integer overflow"],
                    "1514-1": "cuKLEE: Bug Detected: attention_kernels.cu:192: integer overflow",
                    "1514-2": ["cuKLEE: Bug Detected: attention_kernels.cu:26: integer overflow", "cuKLEE: Bug Detected: attention_kernels.cu:27: integer overflow", "cuKLEE: Bug Detected: attention_kernels.cu:29: integer overflow", "cuKLEE: Bug Detected: attention_kernels.cu:30: integer overflow"],
                    "1841": "cuKLEE: Bug Detected: /usr/local/cuda/include/cuda_fp16.hpp:1685: out of bound pointer", # bug is at line 22 or 28 in pos_encoding_kernels.cu
                    "1959": "cuKLEE: Bug Detected: attention_kernels.cu:43: out of bound pointer",
                    "2164": "cuKLEE: Bug Detected: pos_encoding_kernels.cu:63: integer overflow",
                    "5169": "cuKLEE: Bug Detected: /usr/local/cuda/include/cuda/pipeline:538: out of bound pointer", # bug is at line 73-74 in punica/bgmv/bgmv_impl.cuh
                    "6649": ["cuKLEE: Bug Detected: quantization/fp8/common.cu:167: integer overflow", "cuKLEE: Bug Detected: quantization/fp8/common.cu:173: integer overflow"],
                    "6798": "cuKLEE: Bug Detected: gptq_marlin/gptq_marlin.cu:865: out of bound pointer",
                    "9391-1": "cuKLEE: Bug Detected: quantization/compressed_tensors/int8_quant_kernels.cu:104: integer overflow",
                    "9391-2": "cuKLEE: Bug Detected: quantization/compressed_tensors/int8_quant_kernels.cu:119: integer overflow",
                    "9391-3": ["cuKLEE: Bug Detected: quantization/compressed_tensors/int8_quant_kernels.cu:135: integer overflow", "cuKLEE: Bug Detected: quantization/compressed_tensors/int8_quant_kernels.cu:154: integer overflow"],
                    "9391-4": ["cuKLEE: Bug Detected: quantization/compressed_tensors/int8_quant_kernels.cu:168: integer overflow", "cuKLEE: Bug Detected: quantization/compressed_tensors/int8_quant_kernels.cu:203: integer overflow"],
                    "9425": "cuKLEE: Bug Detected: quantization/fp8/common.cu:207: integer overflow",
                    "10928": "cuKLEE: Bug Detected: /usr/local/cuda/include/cub/block/block_store.cuh:331: out of bound pointer", # bug is at line 361 or 434 or 441 in causal_conv1d.cu
                    "12413": ["cuKLEE: Bug Detected: moe/moe_align_sum_kernels.cu:39: out of bound pointer", "cuKLEE: Bug Detected: moe/moe_align_sum_kernels.cu:48: out of bound pointer"],
                    "26192": ["cuKLEE: Bug Detected: layernorm_kernels.cu:23: out of bound pointer", "cuKLEE: Bug Detected: layernorm_kernels.cu:37: out of bound pointer"],
                    "18245": "cuKLEE: Bug Detected: marlin_template.h:1771: out of bound pointer",
                    "9838": "cuKLEE: Bug Detected: causal_conv1d.cu:449: out of bound pointer"}

simplified_results = {"1514-2": ["cuKLEE: Bug Detected: vllm1514-2.cu:43: integer overflow", "cuKLEE: Bug Detected: vllm1514-2.cu:44: integer overflow", "cuKLEE: Bug Detected: vllm1514-2.cu:48: integer overflow", "cuKLEE: Bug Detected: vllm1514-2.cu:49: integer overflow"],
                      "2164": "cuKLEE: Bug Detected: vllm2164.cu:85: integer overflow",
                      "26192": "cuKLEE: Bug Detected: vllm26192.cu:39: out of bound pointer",
                      "6798": "cuKLEE: Bug Detected: vllm6798.cu:51: null pointer dereference",
                      "18245": "cuKLEE: Bug Detected: src/vllm18245.cu:54: out of bound pointer",
                      "9391": "cuKLEE: Bug Detected: vllm9391.cu:40: integer overflow",
                      "9391-2": ["cuKLEE: Bug Detected: vllm9391-2.cu:42: integer overflow", "cuKLEE: Bug Detected: vllm9391-2.cu:43: integer overflow"],
                      "9391-3": ["cuKLEE: Bug Detected: vllm9391-3.cu:56: integer overflow", "cuKLEE: Bug Detected: vllm9391-3.cu:76: integer overflow"],}

gklee_results = {"vllm68": 52, "vllm1295": 120, "vllm1514-1": 45, "vllm1514-2": [39, 43], "vllm1841": [37, 43], "vllm1959": 54, 
                 "vllm2164": 83, "vllm5169":45, "vllm6649": 77, "vllm6798": 51, "vllm9391": 35, "vllm9391-2": 38, "vllm9391-3": 54, "vllm9391-4": 77, "vllm9425": 33, "vllm10928": 44, "vllm12413": 59, "vllm26192": 39, "vllm18245": 54, "vllm9838": 47}

esmbc_results = {"vllm68": 51, "vllm1295": 155, "vllm1514-1": 51, "vllm1514-2": [38, 42], "vllm1841": 70, "vllm1959": 38, 
                 "vllm2164": 78, "vllm5169":47, "vllm6649": 42, "vllm6798": 46, "vllm9391": 53, "vllm9391-2": 51, "vllm9391-3": 63, "vllm9391-4": 83, "vllm9425": 41, "vllm10928": 49, "vllm12413": 62, "vllm26192": 57, "vllm18245": 66, "vllm9838": 64}

current_section_dir = Path(__file__).resolve().parent

def verify_cuKLEE_results():
    count = set()
    for fname in os.listdir(os.path.join(current_section_dir, "cuKLEE/log-original")):
        with open(os.path.join(current_section_dir, "cuKLEE/log-original", fname), "r") as f:
            content = f.read()
        found_bug = False
        for key in original_results:
            if fname.startswith(key):
                if isinstance(original_results[key], list):
                    for i in original_results[key]:
                        if i in content:
                            found_bug = True
                            count.add(key)
                            print(i)
                            break
                else:
                    if original_results[key] in content:
                        found_bug = True
                        count.add(key)
                        print(original_results[key])
                        break
                if found_bug:
                    break
    
    print(f"cuKLEE has found {len(count)} bugs on the original dataset.\n")
    
    for fname in os.listdir(os.path.join(current_section_dir, "cuKLEE/log-simplified")):
        with open(os.path.join(current_section_dir, "cuKLEE/log-simplified", fname), "r") as f:
            content = f.read()
        found_bug = False
        for key in simplified_results:
            if key + "_" in fname:
                if isinstance(simplified_results[key], list):
                    for i in simplified_results[key]:
                        if i in content:
                            found_bug = True
                            if key == "9391" and "9391-1" in count:
                                break
                            count.add(key)
                            print(i)
                            break
                else:
                    if simplified_results[key] in content:
                        found_bug = True
                        if key == "9391" and "9391-1" in count:
                            break
                        count.add(key)
                        print(simplified_results[key])
                        break
                if found_bug:
                    break
    
    print(f"cuKLEE has found {len(count)} bugs in simplified dataset.\n")

def verify_gklee_results():
    count = 0
    for dirname in os.listdir(os.path.join(current_section_dir, "gklee/results")):
        with open(os.path.join(current_section_dir, "gklee/results", dirname, "gklee.log"), "r") as f:
            content = f.read()
        found_bug = False
        for key in gklee_results:
            if dirname.startswith(key):
                if isinstance(gklee_results[key], list):
                    for i in gklee_results[key]:
                        if ":" + str(i) in content:
                            found_bug = True
                            count += 1
                            print(f"GKLEE has found the bug at line {i} in {dirname}")
                            break
                else:
                    if ":" + str(gklee_results[key]) in content:
                        found_bug = True
                        count += 1
                        print(f"GKLEE has found the bug at line {gklee_results[key]} in {dirname}")
                        break
                if found_bug:
                    break
    
    print(f"GKLEE has found {count} bugs in total.\n")

def verify_esmbc_results():
    count = 0
    for fname in os.listdir(os.path.join(current_section_dir, "esbmc/results")):
        with open(os.path.join(current_section_dir, "esbmc/results", fname), "r") as f:
            content = f.read()
        found_bug = False
        for key in esmbc_results:
            if fname.startswith(key):
                if isinstance(esmbc_results[key], list):
                    for i in esmbc_results[key]:
                        if "line " + str(i) in content:
                            found_bug = True
                            count += 1
                            print(f"ESBMC has found the bug at line {i} in {fname}")
                            break
                else:
                    if "line " + str(esmbc_results[key]) in content:
                        found_bug = True
                        count += 1
                        print(f"ESBMC has found the bug at line {esmbc_results[key]} in {fname}")
                        break
                if found_bug:
                    break
    
    print(f"ESBMC has found {count} bugs in total.\n")

if __name__ == "__main__":
    verify_cuKLEE_results()
    verify_gklee_results()
    verify_esmbc_results()