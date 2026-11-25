# Artifact Evaluation: Reproducing the M2K Paper (SOSP 2026)

**Paper:** M2K: Making the Model-Kernel Interface Explicit for Reliable CUDA Kernel Verification

This document explains how to reproduce the experimental results reported in the paper: the empirical study of kernel memory bugs (Section 2.3), bug detection in the wild (Section 6.1), coverage and advancement (Section 6.2), and the ablation study on component rationality (Section 6.3).

Before you start, install M2K and try the walkthrough in [getting-started.md](getting-started.md). If `git lfs pull` fails, see [Troubleshooting: Git LFS Download Issue](getting-started.md#4-troubleshooting-git-lfs-download-issue).

## Kernel Memory Bugs in Inference Systems (Section 2.3)

In Section 2.3, we collect 20 CUDA kernel bugs from the vLLM repository and analyze their characteristics. The pull requests and labels for each bug are available in [this Google Sheet](https://docs.google.com/spreadsheets/d/1Q_6QZbl2I0xCotst-8ei2ZysvldA6D2HHegWKmv9y1o/edit?gid=0#gid=0). The column names match the `Breakdown` column in Table 1. 

## Bug Detection in the Wild (Section 6.1)
We provide two scripts to reproduce this experiment:

1. **`run_small.sh`** analyzes only the model-kernel combinations for which M2K reports at least one potential bug (either a true positive or a false positive). It does not perform bug validation and completes in a few hours.

2. **`run_benchmark.sh`** reproduces all experiments reported in this section. Running the script with a single process takes more than one month to complete.

(both do not need GPU)

We recommend that reviewers use `run_small.sh`, as it reproduces the key results while requiring substantially less time.

**Execute `run_small.sh`:**

```bash
# apply on https://huggingface.co/settings/tokens
export HF_TOKEN=<Hugging Face Token>
cd evaluation/section-6-1-bug-detection
bash run_small.sh
# evaluation/section-6-1-bug-detection/intermediate_results/<vllm/huggingface/papers>/input contains the cuKLEE inputs used in our paper. To skip HFProbe, pass this path to '--input-dir' when running 'python3 cuKLEE/run.py'.
```
(take ~5h)

Symbolic-execution logs and outputs are stored in `${REPO_ROOT}/cuKLEE/results-small/<vllm/huggingface/papers>/log` and `${REPO_ROOT}/cuKLEE/results-small/<vllm/huggingface/papers>/out`.
Validation results are written to `${REPO_ROOT}/evaluation/section-6-1-bug-detection/small_dataset_results/<vllm/huggingface>/benchmark_validation_results.json`.

The detailed labels for each report are summarized in [this Google Sheet](https://docs.google.com/spreadsheets/d/1Q_6QZbl2I0xCotst-8ei2ZysvldA6D2HHegWKmv9y1o/edit?gid=411201267#gid=411201267). The `S` column in the Google Sheet reports the estimated GPU memory size for each detected bug. Figure 5 is drawn based on this column and the memory size of different GPU models. The value of each data point in the figure is available in [this Google Sheet](https://docs.google.com/spreadsheets/d/1Q_6QZbl2I0xCotst-8ei2ZysvldA6D2HHegWKmv9y1o/edit?gid=292022127#gid=292022127).

**Execute `run_benchmark.sh`:**

```bash
# apply on https://huggingface.co/settings/tokens
export HF_TOKEN=<Hugging Face Token>
export OPENAI_API_KEY=<Openai API Key>
cd evaluation/section-6-1-bug-detection
bash run_benchmark.sh
# evaluation/section-6-1-bug-detection/intermediate_results/<vllm/huggingface/papers>/input contains the cuKLEE inputs used in our paper. To skip HFProbe, pass this path to '--input-dir' when running 'python3 cuKLEE/run.py'.
```

Symbolic-execution logs and outputs are stored in `${REPO_ROOT}/cuKLEE/results/<vllm/huggingface/papers>/log` and `${REPO_ROOT}/cuKLEE/results/<vllm/huggingface/papers>/out`.
Validation results are written to `${REPO_ROOT}/evaluation/section-6-1-bug-detection/new_results/<vllm/huggingface>/benchmark_validation_results.json`.


## Coverage and Advancement (Section 6.2)

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

The simplified cuKLEE benchmarks cover 11 of the 20 bugs. For the remaining 9 bugs, cuKLEE uses the corresponding benchmarks from the `original/` directory.

The buggy source-code locations are listed in `evaluation/section-6-2-coverage/benchmarks/cuKLEE-simplified/buggy_lines.txt` and `evaluation/section-6-2-coverage/benchmarks/original/buggy_lines.txt`.

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

The buggy source-code locations are listed in `evaluation/section-6-2-coverage/benchmarks/gklee-simplified/buggy_lines.txt`.

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

The buggy source-code locations are listed in `evaluation/section-6-2-coverage/benchmarks/esbmc-simplified/buggy_lines.txt`.

The following command summarizes the tool outputs.
```bash
cd evaluation/section-6-2-coverage
python3 verify_results.py
```

## Rationality of Components (Section 6.3)

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

