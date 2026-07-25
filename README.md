# M2K: Source Code, Scripts, and Data for SOSP 2026 Artifact Evaluation

**Version:** 1.0  

**Last updated:** July 13, 2026

**Paper:** M2K: Making the Model-Kernel Interface Explicit for Reliable CUDA Kernel Verification

This document provides instructions for reproducing the experimental results reported in the M2K paper. We first describe the hardware and software requirements for running the artifact and explain how to install it. We then walk through the example in Figure 2 to demonstrate how the tool works. Finally, we explain how to reproduce the empirical study of kernel memory bugs (Section 2.3), the bug-detection results in the wild (Section 6.1), the coverage and advancement measurements (Section 6.2), and the ablation study on component rationality (Section 6.3).

## 1. System Requirements

- **OS:** Ubuntu 22.04
- **Memory (RAM):** >=256 GB
- **Disk space:** >=155GB
- **GPU:** H200 (for the ablation experiment)
- **CUDA:** 12.1
- **C/C++ compiler:** GCC/G++ 11, including `gcc-11`, `g++-11`, and `libstdc++-11-dev`
- **Python:** 3.10

## 2. Installation

### Option-1: Native installation
```bash
bash setup.sh
source env.sh
```
### Option-2: Containerized installation 
```bash
docker build --build-arg BUILD_JOBS="$(nproc)" --platform=linux/amd64 -t m2k-env .
docker run --platform=linux/amd64 -it m2k-env
```


## 3. Tool Demonstration 

This section walks through the example in Figure 2 to show the basic workflow of M2K.

### a. Using HFProbe to profile Qwen2ForCausalLM

**Step 1: Static call graph computation**

**Command:**

```bash
cd HFProbe/call-graph
bash x scan --vllm-model-arch=Qwen2ForCausalLM --kernel-info-out=opout
```
(take ~2min)

`bash x scan` accepts the following options:

- `--kernel-info-out=<dir>` — directory that stores information about the possibly-triggered kernels (default: `HFProbe/call-graph/opout`).
- `--vllm-dir=<dir>` — directory of the vLLM repository (ensure not to contain "." in the absolute path); if unset, copy the installed vLLM to HFProbe/vllm (ensure the absolute path of HFProbe does not contain ".").
- `--vllm-model-arch=<arch>` — model architecture of the target model; if unset, all vLLM model architectures are analyzed.

**Console output:**

```text
Analyzing Qwen2ForCausalLM in vLLM...
...
Completed analysis for vllm model architecture Qwen2ForCausalLM!
Kernel information is stored in opout.
```

**File output:**

The file `opout/Qwen2ForCausalLM.json` contains information about all CUDA kernels invoked by the Qwen2ForCausalLM model. For example, the excerpt below from `opout/Qwen2ForCausalLM.json` shows that the dynamic_scaled_int8_quant kernel is invoked by the model from `vllm/_custom_ops.py`, with the corresponding call site spanning lines 1258--1293.

``` json
{
    "dynamic_scaled_int8_quant": {
        "filePath": "vllm/_custom_ops.py",
        "lines": [
            1258,
            1293
        ]
    },
}
```


#### Step 2: Running the profiling backend

**Command:**

```bash
cd ../..
# run from the repository root
python3 -m HFProbe.backend.config_agent_vllm --model-id=Qwen/Qwen2-0.5B-Instruct --profile-out-dir=HFProbe/results/vllm
# run with model config example/config/dynamic_scaled_fp8_quant.json
# python3 -m HFProbe.backend.config_agent_vllm --model-id=Qwen/Qwen2-0.5B-Instruct --profile-out-dir=HFProbe/results/vllm --config-file=example/config/dynamic_scaled_fp8_quant.json
# run all configs in <profile-out-dir>/config/<model_architecture>
# python3 -m HFProbe.backend.config_agent_vllm --model-id=Qwen/Qwen2-0.5B-Instruct --model-architecture=Qwen2ForCausalLM --profile-out-dir=HFProbe/results/vllm --mutate --use-existent-config
```

(take ~20s)

`HFProbe.backend.config_agent_vllm` accepts the following options:

- `--model-id=<modelID>` — the ID of the target model. 
- `--model-architecture=<arch>` — the architecture of the target model, as specified in the model config.
- `--seed-config-file=<file>` — the seed config file used as the basis for mutation.
- `--profile-out-dir=<dir>` — the output directory (see the layout below).
- `--kernel-info-out=<dir>` — the directory containing the kernel innovation information reported by the previous step.
- `--kernel-info-file=<file>` — the file containing the kernel innovation information reported by the previous step.
- `--mutate` — mutate the input configs and without this flag, the model is only profiled with the default config.
- `--use-existent-config` — reuse previously generated configs.
- `--config-file=<file>` — the path to the config file to profile and the file name should be the target kernel name.
- `--kernel-name` — the name of the kernel to trigger.



**Console output:**

```text
INFO 07-11 10:53:59 [__init__.py:247] No platform detected, vLLM is running on UnspecifiedPlatform
patch current platform as cuda platform
Running model  Qwen/Qwen2-0.5B-Instruct ...
INFO 07-11 10:54:09 [__init__.py:31] Available plugins for group vllm.general_plugins:
...
INFO 07-11 10:54:42 [llm_engine.py:428] init engine (profile, create kv cache, warmup model) took 3.90 seconds
batch_size=1, seq_len=1 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1198.72it/s]
Processed prompts: 100%|██████████████████████████████████████| 1/1 [00:00<00:00,  1.12it/s, est. speed input: 1.12 toks/s, output: 17.98 toks/s]
batch_size=1, seq_len=7 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2149.82it/s]
Processed prompts: 100%|██████████████████████████████████████| 1/1 [00:00<00:00,  1.12it/s, est. speed input: 7.84 toks/s, output: 17.92 toks/s]
batch_size=1, seq_len=17 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1475.83it/s]
Processed prompts: 100%|█████████████████████████████████████| 1/1 [00:00<00:00,  1.14it/s, est. speed input: 19.31 toks/s, output: 18.17 toks/s]
batch_size=3, seq_len=1 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 3491.37it/s]
Processed prompts: 100%|██████████████████████████████████████| 3/3 [00:00<00:00,  3.05it/s, est. speed input: 3.05 toks/s, output: 48.77 toks/s]
batch_size=3, seq_len=7 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 2478.90it/s]
Processed prompts: 100%|█████████████████████████████████████| 3/3 [00:01<00:00,  2.86it/s, est. speed input: 20.03 toks/s, output: 45.77 toks/s]
batch_size=3, seq_len=17 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 1852.61it/s]
Processed prompts: 100%|█████████████████████████████████████| 3/3 [00:01<00:00,  2.92it/s, est. speed input: 49.72 toks/s, output: 46.79 toks/s]
batch_size=5, seq_len=1 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 3082.69it/s]
Processed prompts: 100%|██████████████████████████████████████| 5/5 [00:01<00:00,  4.85it/s, est. speed input: 4.85 toks/s, output: 77.67 toks/s]
batch_size=5, seq_len=7 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 2637.93it/s]
Processed prompts: 100%|█████████████████████████████████████| 5/5 [00:01<00:00,  4.73it/s, est. speed input: 33.09 toks/s, output: 75.64 toks/s]
batch_size=5, seq_len=17 ...
Adding requests: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 2071.26it/s]
Processed prompts: 100%|█████████████████████████████████████| 5/5 [00:01<00:00,  4.63it/s, est. speed input: 78.67 toks/s, output: 74.04 toks/s]
[Qwen/Qwen2-0.5B-Instruct] Cache removed: .cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d
```

**Output directory:**

```text
config/
├── <model_architecture>/
│   └── <kernel_name>.json  # config file generated by mutation for the kernel
data/
├── <model_id>/
│   └── <kernel_name>.json  # profiling information with the mutated config for the kernel
└── <model_id>.json         # profiling information with the default config
input/          # interface constraints for cuKLEE (in step 4)
└── <kernel_name>.json 
```



`<profile-out-dir>/data/Qwen_Qwen2-0.5B-Instruct.json` contains the profiling results generated using the default model configuration. The example below shows the profiling information for the `rms_norm` kernel from two executions. In both executions, the batch size is `1`, while the sequence lengths are `1` and `7`, respectively. For each kernel invocation, the profiler records the call stack and the types of all parameters. If a parameter is a tensor, its shape is also recorded.


```json
{
    "vllm.rms_norm": {
        "(('_custom_ops.py', 276), ('model_executor/layers/layernorm.py', 23), ('model_executor/layers/layernorm.py', 171), ('model_executor/custom_op.py', 23), ('model_executor/models/qwen2.py', 253))": {
            "(1, 1)": [
                [
                {
                    "shape": [
                        1,
                        896
                    ],
                    "dtype": "torch.float16",
                    "type": "torch.Tensor"
                },
                {
                    "shape": [
                        1,
                        896
                    ],
                    "dtype": "torch.float16",
                    "type": "torch.Tensor"
                },
                {
                    "shape": [
                        896
                    ],
                    "dtype": "torch.float16",
                    "type": "torch.Tensor"
                },
                {
                    "value": 1e-06,
                    "type": "float"
                }
                ]
            ],
            "(1, 7)": [
                [
                {
                    "shape": [
                        7,
                        896
                    ],
                    "dtype": "torch.float16",
                    "type": "torch.Tensor"
                },
                {
                    "shape": [
                        7,
                        896
                    ],
                    "dtype": "torch.float16",
                    "type": "torch.Tensor"
                },
                {
                    "shape": [
                        896
                    ],
                    "dtype": "torch.float16",
                    "type": "torch.Tensor"
                },
                {
                    "value": 1e-06,
                    "type": "float"
                }
                ],
            ]
        }
    }
}
```

#### Step 3: Mutating Model Configuration

**Command:**

```bash
export OPENAI_API_KEY=<Openai API Key>
python3 -m HFProbe.backend.config_agent_vllm --model-architecture=Qwen2ForCausalLM --profile-out-dir=HFProbe/results/vllm --kernel-name=dynamic_scaled_fp8_quant --seed-config-file=example/Qwen2ForCausalLM_model_config.json --kernel-info-file=example/Qwen2ForCausalLM_kernel_info.json
# to mutate configs for all kernels and run profiling backend in one batch
# python3 -m HFProbe.backend.config_agent_vllm --model-id=Qwen/Qwen2-0.5B-Instruct --model-architecture=Qwen2ForCausalLM --profile-out-dir=HFProbe/results/vllm --kernel-info-file=example/Qwen2ForCausalLM_kernel_info.json --mutate --seed-config-file=example/Qwen2ForCausalLM_model_config.json
```

(take ~2min)

**Console output:**

```text
prompt: For models using Qwen2ForCausalLM, analyze the python code in vllm. Generate model config to trigger ...
****************************************
GPT response:
{
    "architectures": [
        "Qwen2ForCausalLM"
    ],
    ...
}

The config file generated from mutation to trigger kernel dynamic_scaled_fp8_quant is saved at HFProbe/results/vllm/config/Qwen2ForCausalLM/dynamic_scaled_fp8_quant.json.
```

The content of `<profile-out-dir>/config/Qwen2ForCausalLM/dynamic_scaled_fp8_quant.json` is shown as follows. 

```json
{
    "architectures": [
        "Qwen2ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 896,
    "initializer_range": 0.02,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "max_window_layers": 24,
    "model_type": "qwen2",
    "num_attention_heads": 14,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.1",
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 151936,
    "quantization_config": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic"
    }
}
```

#### Step 4: Generating Interface Constraints


**Command:**

```bash
# compile cuda files to LLVM bitcode
# running profiling backend will trigger multiple kernels. we have to compile all cuda files in vLLM.
python3 cuKLEE/compile_cuda.py --cuda-source-dir=evaluation/section-6-1-bug-detection/benchmarks/vllm/cuda_files --compiled-kernel-dir=cuKLEE/compiled/vllm
# some compiled files are large. you can skip the compilation and use existing compiled files (--compiled-kernel-dir=evaluation/section-6-1-bug-detection/benchmarks/vllm/compiled_files).
python3 HFProbe/input_generate.py --vllm --add-memory-max-num-tokens --profile-out-dir=HFProbe/results/vllm --compiled-kernel-dir=cuKLEE/compiled/vllm --cuda-source-dir=evaluation/section-6-1-bug-detection/benchmarks/vllm/cuda_files
```

(take ~40mins, most of which is spent compiling the cuda source files)


`compile_cuda.py` accepts the following options:
- `--cuda-source-dir=<dir>` — directory containing the CUDA source files, and any required header files should be placed in this directory or in cuKLEE/include.
- `--compiled-kernel-dir=<dir>` — directory for the compiled kernel files, and if not specified, the input directory is used. 

`input_generate.py` accepts the following options:

- `--profile-out-dir=<dir>` — the profiling output directory specified in Step 2.
- `--compiled-kernel-dir=<dir>` — directory containing the compiled CUDA files.
- `--cuda-source-dir=<dir>` — directory containing the CUDA and C++ source files, or the root directory of the repository.
- `--vllm` — indicating that the profiling results are generated from vLLM.
- `--add-memory-max-num-tokens` - add the `num_tokens` limit corresponding to a 288GB GPU memory configuration.

Possible issue:
```text
.../include/c++/12/bits/shared_ptr_base.h:196:22: error: expected expression
/usr/local/cuda/include/crt/host_defines.h:83:33: note: expanded from macro '__noinline__'
```
Solution: `compile_cuda.py` uses LLVM 13 Clang with CUDA 12.1 and requires GCC/G++ 11. Install the required packages:
```bash
sudo apt-get install -y gcc-11 g++-11 libstdc++-11-dev
```
If Clang still auto-detects GCC 12 or newer C++ headers, use a clean Ubuntu 22.04 environment with GCC 11 or the provided Docker environment. Merely changing the system default `gcc`/`g++` alternatives may not change Clang's auto-detected C++ headers.

**Console output:**
```
Compiling CUDA files...
Compiling common.cu...
Running command: clang++ -g -x cuda --cuda-gpu-arch=sm_80 ... common.cu

Running command: llvm-link-13 -o common_combined.bc common.bc common-cuda-nvptx64-nvidia-cuda-sm_80.bc

Running command: llvm-dis-13 common_combined.bc
...

Done!
The input files are stored in the directory: results/vllm/input
```

**File output:**

The inferred interface constraints for the `dynamic_scaled_fp8_quant` kernel are stored in `<profile-out-dir>/vllm/input/dynamic_scaled_fp8_quant.json`, as shown below. The first two parameters are two-dimensional tensors. Their first dimension is equal to the batch size multiplied by the sequence length, while their second dimension is the model-specific constant `896`. The third parameter is a one-dimensional tensor with a single element.
```json
[
    {
        "py_function": "dynamic_scaled_fp8_quant",
        "cuda_function": "dynamic_scaled_fp8_quant",
        "input_file_path": "example/fp8_common_combined.bc",
        "args": [
            {
                "shape": [
                    "1*b*s",
                    896
                ],
                "dtype": "torch.float8_e4m3fn",
                "type": "torch.Tensor"
            },
            {
                "shape": [
                    "1.0*s*b",
                    896
                ],
                "dtype": "torch.float16",
                "type": "torch.Tensor"
            },
            {
                "shape": [
                    1
                ],
                "dtype": "torch.float32",
                "type": "torch.Tensor"
            }
        ],
        "seq_len": 32768,
        "num_tokens": 2864606
    }
]
```

### b. Using cuKLEE to perform symbolic execution on the dynamic_scaled_fp8_quant kernel 


**Step 5: Executing cuKLEE** 

**Command:**

```bash
mkdir example/out
cuKLEE --timeout=3600 --cuklee-out-dir=example/out example/dynamic_scaled_fp8_quant.json
# to run multiple files in a batch
# python3 cuKLEE/run.py --input-dir=HFProbe/results/vllm/input --cuklee-out-dir=cuKLEE/results/out --log-dir=cuKLEE/results/log
```

(take ~35s)

`cuKLEE/run.py` accepts the following options:
- `--input-dir=<dir>` — directory containing the interface constraints generated in Step 4.
- `--cuklee-out-dir=<dir>` — directory storing z3 constraints for each detected bug.
- `--log-dir=<dir>` — directory storing console output for each input file.

possible issue:

The compiled CUDA files in the `evaluation/section-6-1-bug-detection/benchmarks/<huggingface/vllm/research_papers>/compiled_files` and `example` directories are large and are tracked using Git LFS. If you clone the repository without downloading the Git LFS objects, running cuKLEE may produce an error such as:
```text
ERROR: error loading program...
...: error: expected top-level entity
```

solution:

Download the Git LFS objects by running:
```bash
git lfs pull
```

**Console output:**

```
running opt-13 -load-pass-plugin libSignednessPropagationPass.so -passes=signedness-prop example/fp8_common_combined.bc > example/fp8_common_combined_modified.bc
Pass applied. Modified file: example/fp8_common_combined_modified.bc
running klee --disable-verify --warnings-only-to-file --single-object-resolution=true --external-calls=over-approx --entry-point=_Z24dynamic_scaled_fp8_quantRN2at6TensorERKS0_S1_ --cuda-type=caller --func-config=example/dynamic_scaled_fp8_quant.json --jindex=0 --output-dir=example/out example/fp8_common_combined_modified.bc
cuKLEE: output directory is "example/out/_Z24dynamic_scaled_fp8_quantRN2at6TensorERKS0_S1_/klee-out-jindex-0-0"
cuKLEE: Using Z3 solver backend
...
cuKLEE: Bug Detected: vllm-0.9.0/csrc/quantization/fp8/common.cu:19: integer overflow
cuKLEE: Bug Detected: vllm-0.9.0/csrc/quantization/fp8/common.cuh:128: integer overflow

cuKLEE: done: total instructions = 13680
cuKLEE: done: completed paths = 5
cuKLEE: done: partially completed paths = 22
```

The above console output shows that two integer overflow bugs are detected, one is in line 19 and the other is in line 128. 

**Step 6: Validating Reported Bugs** 

**Command:**

```bash
# if you haven't run previous HFProbe steps, copy example/dynamic_scaled_fp8_quant.json to HFProbe/results/vllm/input 
python3 HFProbe/validation/run_vllm_validation.py --profile-out-dir=HFProbe/results/vllm --cuklee-out-dir=example/out --kernel-name=dynamic_scaled_fp8_quant --index=0 --model-id=Qwen/Qwen2-0.5B-Instruct --config-file=example/config/dynamic_scaled_fp8_quant.json
```
(take ~26min)

`run_vllm_validation.py` accepts the following options:
- `--profile-out-dir=<dir>` — the profiling output directory generated in Step 2.
- `--cuklee-out-dir=<dir>` — the directory containing z3 constraints generated in Step 5. 
- `--kernel-name=<name>` — the name of the target CUDA kernel.
- `--index=<int>` — the index in the input json file in the previous step (default value is 0).
- `--model-id=<modelID>` — the model ID invoking the target kernel.
- `--config-file=<filepath>` — the model config file generated in Step 3.

possible issue:
```text
AttributeError: module 'z3' has no attribute 'parse_smt2_string'
```
solution:
```bash
python3 -m pip uninstall z3
python3 -m pip install --force-reinstall z3-solver
```

**Console output:** 

```text
INFO 07-11 13:12:01 [__init__.py:247] No platform detected, vLLM is running on UnspecifiedPlatform
patch current platform as cuda platform
Running model Qwen/Qwen2-0.5B-Instruct with example/config/dynamic_scaled_fp8_quant.json, batch_size: 65 seq_len: 32264.
INFO 07-11 13:12:13 [__init__.py:31] Available plugins for group vllm.general_plugins:
...
INFO 07-11 13:18:45 [llm_engine.py:428] init engine (profile, create kv cache, warmup model) took 362.18 seconds
real_seq_len=32264 ...
Adding requests: 100%|█████████████████████████████████████| 65/65 [00:11<00:00,  5.56it/s]
Processed prompts: 100%|█| 65/65 [05:51<00:00,  5.41s/it, est. speed input: 5960.85 toks/s]
Running model Qwen/Qwen2-0.5B-Instruct with batch_size: 65 seq_len: 32264 config_file: example/config/dynamic_scaled_fp8_quant.json can trigger bug 19_18_119-asm-15118_4337_11163_io.txt for dynamic_scaled_fp8_quant.
Running model Qwen/Qwen2-0.5B-Instruct with example/config/dynamic_scaled_fp8_quant.json, batch_size: 64 seq_len: 32768.
INFO 07-11 13:25:01 [config.py:520] Overriding HF config with 
...
INFO 07-11 13:31:10 [llm_engine.py:428] init engine (profile, create kv cache, warmup model) took 364.53 seconds
real_seq_len=32768 ...
Adding requests: 100%|█████████████████████████████████████| 64/64 [00:11<00:00,  5.43it/s]
Processed prompts: 100%|█| 64/64 [06:07<00:00,  5.74s/it, est. speed input: 5704.52 toks/s, output: 0.17]
Running model Qwen/Qwen2-0.5B-Instruct with batch_size: 64 seq_len: 32768 config_file: example/config/dynamic_scaled_fp8_quant.json can trigger bug 128_24_18-asm-15242_15133_4337_io.txt for dynamic_scaled_fp8_quant.
```

The validation results are stored in the `${REPO_ROOT}/HFProbe/validation` directory. For example, the following snippet from `${REPO_ROOT}/HFProbe/validation/Qwen_Qwen2-0.5B-Instruct_dynamic_scaled_fp8_quant_0_validate_results.json` shows that the bug reported at line 19 in Step 5 is validated as a true positive.


```json
{
  "19_18_119-asm-15118_4337_11163_io.txt": {
    "status": "success",
    "batch_size": 65,
    "seq_len": 32264,
    "buggy_line": 19,
    "config": "example/config/dynamic_scaled_fp8_quant.json"
  },
}
```

## 4. Kernel Memory Bugs in Inference Systems (Section 2.3)

In Section 2.3, we collect 20 CUDA kernel bugs from the vLLM repository and analyze their characteristics. The pull requests and labels for each bug are available in [this Google Sheet](https://docs.google.com/spreadsheets/d/1Q_6QZbl2I0xCotst-8ei2ZysvldA6D2HHegWKmv9y1o/edit?gid=0#gid=0). The column names match the `Breakdown` column in Table 1. 

## 5. Bug Detection in the Wild (Section 6.1)
We provide two scripts to reproduce this experiment:

1. **`run_small.sh`** analyzes only the model-kernel combinations for which M2K reports at least one potential bug (either a true positive or a false positive). It does not perform bug validation and completes in a few hours.

2. **`run_benchmark.sh`** reproduces all experiments reported in this section. Running the script with a single process takes more than one month to complete.

We recommend that reviewers use `run_small.sh`, as it reproduces the key results while requiring substantially less time.

**Execute `run_small.sh`:**

```bash
# apply on https://huggingface.co/settings/tokens
export HF_TOKEN=<Hugging Face Token>
cd evaluation/section-6-1-bug-detection
bash run_small.sh
```
(take ~5h)

The output is stored at `${REPO_ROOT}/cuKLEE/results-small/<vllm/huggingface/papers>/log`. The detailed labels for each report are summarized in [this Google Sheet](https://docs.google.com/spreadsheets/d/1Q_6QZbl2I0xCotst-8ei2ZysvldA6D2HHegWKmv9y1o/edit?gid=411201267#gid=411201267). The `S` column in the Google Sheet reports the estimated GPU memory size for each detected bug. Figure 5 is drawn based on this column and the memory size of different GPU models. The value of each data point in the figure is available in [this Google Sheet](https://docs.google.com/spreadsheets/d/1Q_6QZbl2I0xCotst-8ei2ZysvldA6D2HHegWKmv9y1o/edit?gid=292022127#gid=292022127).

**Execute `run_benchmark.sh`:**

```bash
# apply on https://huggingface.co/settings/tokens
export HF_TOKEN=<Hugging Face Token>
export OPENAI_API_KEY=<Openai API Key>
cd evaluation/section-6-1-bug-detection
bash run_benchmark.sh
```

The output of cuKLEE is stored at  `${REPO_ROOT}/cuKLEE/results/<vllm/huggingface/papers>/log`. 
The output of the validation is stored at `${REPO_ROOT}/evaluation/section-6-1-bug-detection/new_results/<vllm/huggingface>/benchmark_validation_results.json`. 


## 6. Coverage and Advancement (Section 6.2)

The results reported by each tool for every evaluated bug are summarized in [this Google Sheet](https://docs.google.com/spreadsheets/d/1Q_6QZbl2I0xCotst-8ei2ZysvldA6D2HHegWKmv9y1o/edit?gid=1688430075#gid=1688430075).

**(1) run cuKLEE:**

```bash
cd evaluation/section-6-2-coverage/cuKLEE
python3 run_cuKLEE.py
```
(take ~2.5h)

Results are stored in `${REPO_ROOT}/evaluation/section-6-2-coverage/cuKLEE/`:
- log-original contains the output of cuKLEE running on the original dataset
- log-simplified contains the output of cuKLEE running on the simplified dataset

To quickly localize detected bugs, reviewers can try the following command. 

```bash
grep -R -E "Bug Detected:" evaluation/section-6-2-coverage/cuKLEE/log-*
```

**(2) run GKLEE:**
Run the following commands on your host machine—for example, in your local terminal—not inside a Docker container, because they use docker build and docker run.
```bash
cd evaluation/section-6-2-coverage/gklee
bash run.sh
```

Results are stored in `${REPO_ROOT}/evaluation/section-6-2-coverage/gklee/results/<input>/`:

- `compile.log` contains the `gklee-nvcc` output. Inspect this file if an input
  fails to compile.
- `gklee.log` contains the verification output. `KLEE: ERROR` reports a
  detected issue and includes its generated-source location and error type,
  such as `memory error: out of bound pointer`. `KLEE: done` reports the final
  instruction, path, and generated-test counts.
- The runner prints status `124` for a timeout and status `127` when a required
  command is unavailable. A timed-out log is incomplete and must not be
  interpreted as verification success.

To quickly localize detected bugs, reviewers can try the following command. 

```bash
grep -R -E "KLEE: ERROR|KLEE: done" \
  evaluation/section-6-2-coverage/gklee/results
```

**(3) run Honeycomb:**
Run the following commands on your host machine—for example, in your local terminal—not inside a Docker container, because they use docker build and docker run.
```bash
cd evaluation/section-6-2-coverage/honeycomb
bash run.sh
```

Each result is stored in
`${REPO_ROOT}/evaluation/section-6-2-coverage/honeycomb/results/<input>.log`. The last line
has the form `Generates N remarks`. `0 remarks` means Honeycomb did not report
any policy violation; a nonzero value means that the listed remarks require
inspection. Status `124` printed by the runner means the input timed out.

To quickly localize detected bugs, reviewers can try the following command. 

```bash
grep -R -E "Generates [0-9]+ remarks" \
  evaluation/section-6-2-coverage/honeycomb/results
```

**(4) run ESBMC:**
Run the following commands on your host machine—for example, in your local terminal—not inside a Docker container, because they use docker build and docker run.
```bash
cd evaluation/section-6-2-coverage/esbmc
bash run.sh
```

Each result is stored in
`${REPO_ROOT}/evaluation/section-6-2-coverage/esbmc/results/<input>.log`.
`VERIFICATION FAILED` means ESBMC found a counterexample; the preceding
`Violated property` and state trace identify the property, source location,
and failing execution. `VERIFICATION SUCCESSFUL` means no violation was found
within the explored bounds. Status `124` printed by the runner means timeout;
the corresponding partial log must not be interpreted as verification success.

To quickly localize detected bugs, reviewers can try the following command. 

```bash
grep -R -E "Violated property|VERIFICATION (FAILED|SUCCESSFUL)" \
  evaluation/section-6-2-coverage/esbmc/results
```

## 7. Rationality of Components (Section 6.3)

A summary of the results is available in [this Google Sheet](https://docs.google.com/spreadsheets/d/1Q_6QZbl2I0xCotst-8ei2ZysvldA6D2HHegWKmv9y1o/edit?gid=650464833#gid=650464833).
The results are organized along each analyzed vLLM kernel. 


**(1) Without HFProbe:**

```bash
cd evaluation/section-6-3-rationality
python3 run.py --without=H
```
(take ~1d)

The analysis results are stored in the  `${REPO_ROOT}/evaluation/section-6-3-rationality/wo_h/log` directory.

**(2) Without cuKLEE:**

This needs to run on the GPU.

```bash
cd evaluation/section-6-3-rationality
python3 run.py --without=C
```
(take ~1d)

The analysis results are stored in the  `${REPO_ROOT}/evaluation/section-6-3-rationality/wo_c_bug_results.json`.

**(3) Without configuration mutation:**

```bash
cd evaluation/section-6-3-rationality
python3 run.py --without=M
```
(take ~1d)

The analysis results are stored in the  `${REPO_ROOT}/evaluation/section-6-3-rationality/wo_m/profile/benchmark_validation_results.json`.

**(4) Without validation:**

```bash
cd evaluation/section-6-3-rationality
python3 run.py --without=V
```
(take ~1 month)

The analysis results are stored in the  `${REPO_ROOT}/evaluation/section-6-3-rationality/wo_v/cuklee/log` directory.

Here is a clearer README version:

## 8. Git LFS Download Issue

The Git LFS budget for the account that owns this repository has been exceeded. As a result, downloading some large files may fail during `git lfs pull`.

To clone the repository without downloading the Git LFS objects, run:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <repository-url>
```

The LFS-tracked paths will contain Git LFS pointer files instead of the actual `.bc` and `.ll` contents. You can restore the required files using either of the following methods.

### Option 1: Regenerate the files

From the repository root directory, run:

```bash
python3 evaluation/section-6-1-bug-detection/fix_lfs.py
```

This script regenerates all required LLVM bitcode (`.bc`) and LLVM IR (`.ll`) files, including the files under `example/`.

Depending on your machine, this process may take more than one hour.

### Option 2: Download the files

Download and extract the `large_files.zip` from [this Google Drive Folder](https://drive.google.com/drive/folders/123W8DaDUUhJvgpT_5I8a3vxHXjQwjHUc?usp=drive_link)

Then replace the following directories with the corresponding directories from the extracted archive:

```text
evaluation/section-6-2-coverage/benchmarks/original/
evaluation/section-6-2-coverage/benchmarks/cuKLEE-simplified/
evaluation/section-6-1-bug-detection/benchmarks/
```

Also replace the four corresponding LFS-tracked files under:

```text
example/
```

Make sure that the extracted files preserve the same directory structure and filenames as the files in the repository.




