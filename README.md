<h1 align="center">M2K</h1>

<p align="center">
  <b>Find bugs in the CUDA kernels used in LLM inference.</b>
</p>

<p align="center">
  <a href="https://github.com/vllm-project/vllm/security/advisories/GHSA-7m6h-x95x-82q5"><img alt="CVE" src="https://img.shields.io/badge/CVE-2026--73558-orange"></a>
  <a href="#citation"><img alt="SOSP 2026" src="https://img.shields.io/badge/paper-SOSP%202026-blue"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

<p align="center">
  <a href="#bugs-found-by-m2k">Bugs Found</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a>
</p>

---

M2K is a verification tool for the CUDA kernels behind LLM inference. It checks the kernels in serving frameworks such as **vLLM** and **SGLang**, as well as the custom kernels that ship with models themselves, for example as remote code on the Hugging Face Hub or in the code released with research papers. M2K detects four classes of bugs:

- integer overflow
- out-of-bounds memory access
- null pointer dereference
- data race

Whether a kernel is buggy depends on the inputs it actually receives, and those inputs are determined by the model: its config, the batch size, and the sequence length. **M2K makes this model-kernel interface explicit.** For each kernel, it infers how the kernel's arguments relate to the model config and request shape, symbolically executes the kernel under those constraints, and validates each report by running the model with a config and input that trigger the bug.

## Highlights

- **Covers frameworks and models.** M2K checks the kernels in inference frameworks (vLLM, SGLang) and the custom kernels shipped with models (Hugging Face Hub remote code, research-paper models).
- **Finds real bugs.** Bugs found by M2K have been confirmed and fixed by vLLM and SGLang developers, and one was assigned [CVE-2026-73558](https://github.com/vllm-project/vllm/security/advisories/GHSA-7m6h-x95x-82q5).
- **Reports you can reproduce.** Each validated report comes with the model config, batch size, and sequence length that trigger it in the real framework.
- **No GPU needed for detection.**

## Bugs Found by M2K

M2K has found bugs in vLLM, SGLang, and custom kernels from Hugging Face models. Developers have confirmed and fixed many of them, and one led to the vLLM security advisory [GHSA-7m6h-x95x-82q5](https://github.com/vllm-project/vllm/security/advisories/GHSA-7m6h-x95x-82q5) (CVE-2026-73558).

📋 **See the [M2K Bug Tracker](https://docs.google.com/spreadsheets/d/1DL56JueoLBoU6qtJmp_TLZ_iiUjSksNCZ2tLUFQvaNg/edit?usp=sharing) for the full, up-to-date list, with links to each issue, fix PR, and security advisory.**

If you use M2K and find a bug, we would love to add it. Please open an issue.

## How It Works

<p align="center">
  <img src="docs/images/overview.png" alt="M2K overview: HFProbe infers interface constraints from the model config and code; cuKLEE symbolically executes the CUDA code under those constraints and validates each bug" width="100%">
</p>

M2K has two components:

- **[HFProbe](HFProbe/)** synthesizes the model-kernel interface. It builds a call graph from the model code to find the kernels the model can reach, uses an LLM to mutate `config.json` so that more of those kernels are triggered, and profiles each kernel call (argument shapes, dtypes, and values). From these traces it synthesizes interface constraints, for example "argument 0 has shape `[batch * seq_len, 896]`".
- **[cuKLEE](cuKLEE/)** is a symbolic execution engine for CUDA kernels built on [KLEE](https://klee-se.org/). It analyzes both the host-side wrapper and the kernel under the interface constraints, and reports integer overflows, out-of-bounds accesses, null pointer dereferences, and data races. A solver turns each finding into a concrete buggy input, which M2K validates in two steps: it checks that the input fits in GPU memory, then reruns the model in the framework with that config, batch size, and sequence length to confirm the bug.

## Quick Start

**Requirements:** Ubuntu 22.04, Python 3.10, CUDA 12.1, GCC 11. Detection does not need a GPU. See the [full requirements](docs/getting-started.md#1-system-requirements).

```bash
git clone https://github.com/system-pclub/M2K.git && cd M2K

# Option 1: native install
bash setup.sh && source env.sh

# Option 2: Docker
docker build --build-arg BUILD_JOBS="$(nproc)" --platform=linux/amd64 -t m2k-env .
docker run --platform=linux/amd64 -it m2k-env
```

Run cuKLEE on the bundled example, the FP8 quantization kernel in vLLM 0.9.0:

```bash
mkdir -p example/out
cuKLEE --timeout=3600 --cuklee-out-dir=example/out example/dynamic_scaled_fp8_quant.json
```

```text
cuKLEE: Bug Detected: vllm-0.9.0/csrc/quantization/fp8/common.cu:19: integer overflow
cuKLEE: Bug Detected: vllm-0.9.0/csrc/quantization/fp8/common.cuh:128: integer overflow
```

To run the whole pipeline on a model, from call-graph analysis to validation, follow the [walkthrough](docs/getting-started.md#3-walkthrough-finding-a-bug-in-dynamic_scaled_fp8_quant).

## Documentation

| Document | Contents |
| --- | --- |
| [Getting Started](docs/getting-started.md) | Requirements, installation, an end-to-end walkthrough of every HFProbe and cuKLEE command and its options, troubleshooting |
| [Artifact Evaluation](docs/artifact-evaluation.md) | Scripts to reproduce every experiment in the SOSP 2026 paper |
| [Bug Tracker](https://docs.google.com/spreadsheets/d/1DL56JueoLBoU6qtJmp_TLZ_iiUjSksNCZ2tLUFQvaNg/edit?usp=sharing) | All bugs found by M2K|

## Citation

If you use M2K in your research, please cite our paper:

```bibtex
@inproceedings{m2k-sosp26,
  title     = {M2K: Making the Model-Kernel Interface Explicit for Reliable CUDA Kernel Verification},
  booktitle = {Proceedings of the 32nd ACM Symposium on Operating Systems Principles (SOSP)},
  year      = {2026}
}
```

## License

M2K is released under the [MIT License](LICENSE). cuKLEE is derived from KLEE, which is distributed under its own [license](cuKLEE/klee/LICENSE.TXT).
