"""Validate cuKLEE results against research-system (RS) model executions."""

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(root_dir)

if __package__ is None or __package__ == "":
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from HFProbe.validation.verify_input import (
        compare_json_arrays,
        getFuncName,
        locate_index,
        solve_with_bounds,
    )
else:
    from .verify_input import (
        compare_json_arrays,
        getFuncName,
        locate_index,
        solve_with_bounds,
    )


MODEL_IDS = {
    "ShiftAddLLM": "facebook/opt-125m",
    "AQLM": "meta-llama/Llama-2-7b-hf",
    "Mixture-Compressor-MoE": "meta-llama/Llama-2-7b-hf",
    "any-precision-llm": "meta-llama/Llama-2-7b-chat-hf",
}

executed_configs = {}
_rs_models = None
RS_INPUT_MATCH_IGNORED = {"maxV", "minV", "symRanges"}


def _get_rs_models():
    """Import the heavyweight RS backend only when a model must be run."""
    global _rs_models
    if _rs_models is None:
        _rs_models = importlib.import_module("HFProbe.backend.run_rs_models")
    return _rs_models


def _get_model_limits(model_id):
    """Load model-limit helpers only for constraint solving/model execution."""
    input_generate = importlib.import_module("HFProbe.input_generate")
    return (
        input_generate.get_max_token_vllm(model_id),
        input_generate.get_max_model_len(model_id),
    )


def _normalise_op_name(name):
    """Return the kernel-map spelling of a recorded Python/C++ call."""
    return name.split("::")[-1].split(".")[-1]


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def _find_input_path(profile_dir, cuda_func):
    """Find an input JSON, including files named after a wrapper function."""
    input_dir = os.path.join(profile_dir, "input")
    direct_path = os.path.join(input_dir, f"{cuda_func}.json")
    if os.path.exists(direct_path):
        return direct_path

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue
        candidate = os.path.join(input_dir, filename)
        data = _load_json(candidate)
        if any(
            isinstance(item, dict) and item.get("cuda_function") == cuda_func
            for item in data
        ):
            return candidate
    return direct_path


def _run_framework(framework_name, config, op_name, case_dir):
    run_rs_models = _get_rs_models()
    if framework_name == "ShiftAddLLM":
        run_rs_models.testShiftAdd(
            override_configs=config,
            out_dir=os.path.join(case_dir, "out"),
            op_name=op_name,
        )
    elif framework_name == "AQLM":
        run_rs_models.testAqlmManual(
            override_configs=config,
            out_dir=os.path.join(case_dir, "out"),
            data_dir=os.path.join(case_dir, "data"),
            op_name=op_name,
        )
    elif framework_name == "Mixture-Compressor-MoE":
        run_rs_models.testMCM(
            override_configs=config,
            out_dir=os.path.join(case_dir, "out"),
            data_dir=os.path.join(case_dir, "data"),
            op_name=op_name,
        )
    elif framework_name == "any-precision-llm":
        run_rs_models.testAnyPrecision(
            override_configs=config,
            out_dir=os.path.join(case_dir, "out"),
            data_dir=os.path.join(case_dir, "data"),
            op_name=op_name,
        )
    else:
        supported = ", ".join(sorted(MODEL_IDS))
        raise ValueError(
            f"Unsupported RS framework {framework_name!r}; expected one of: {supported}"
        )


def run_rs_case(
    framework_name,
    override_configs,
    op_name,
    batch_size,
    seq_len,
    lineno,
    index,
    result_dir,
    rerun=False,
):
    """Run one RS validation case and return its concrete recorded calls."""
    validation_dir = os.path.join(result_dir, "validation", framework_name)
    cache_path = os.path.join(
        validation_dir, f"{op_name}-{lineno}-{index}-b{batch_size}-s{seq_len}.json"
    )
    if os.path.exists(cache_path) and not rerun:
        return _load_json(cache_path)

    case_name = f"{op_name}-{lineno}-{index}-b{batch_size}-s{seq_len}"
    case_dir = os.path.join(validation_dir, "cases", case_name)

    run_rs_models = _get_rs_models()
    old_batch_sizes = run_rs_models.BATCH_SIZE_CONFIGS
    old_seq_lens = run_rs_models.SEQ_LENS_CONFIGS
    try:
        run_rs_models.BATCH_SIZE_CONFIGS = [batch_size]

        # The Llama tokenizers used by the RS runners prepend one special token.
        # Include both candidates so the recorded (real) sequence length contains
        # the solver's value regardless of tokenizer/model special-token policy.
        requested_seq_lens = []
        for candidate in (max(1, seq_len - 1), seq_len):
            if candidate not in requested_seq_lens:
                requested_seq_lens.append(candidate)
        run_rs_models.SEQ_LENS_CONFIGS = requested_seq_lens

        _run_framework(
            framework_name,
            copy.deepcopy(override_configs),
            op_name,
            case_dir,
        )
    finally:
        run_rs_models.BATCH_SIZE_CONFIGS = old_batch_sizes
        run_rs_models.SEQ_LENS_CONFIGS = old_seq_lens

    candidates = [
        os.path.join(case_dir, "data", framework_name, f"{op_name}.json"),
        os.path.join(case_dir, "out", framework_name, f"{op_name}.json"),
    ]
    for output_path in candidates:
        if os.path.exists(output_path):
            data = _load_json(output_path)
            # Validation needs concrete runs. A dict is the symbolic aggregate,
            # while a list contains the per-batch call records.
            if isinstance(data, list):
                _write_json(cache_path, data)
                return data
    return None


def _find_matching_call(params_data, py_func, target_params, batch_size, seq_len):
    for item in params_data:
        if item.get("batch_size") != batch_size or item.get("seq_len") != seq_len:
            continue
        for call in item.get("calls", []):
            if _normalise_op_name(call.get("name", "")) != py_func:
                continue
            if compare_json_arrays(
                target_params,
                call.get("args", []),
                {"s": seq_len, "b": batch_size},
            ):
                return True
    return False


def _has_recorded_batch_seq(params_data, batch_size, seq_len):
    return any(
        item.get("batch_size") == batch_size and item.get("seq_len") == seq_len
        for item in params_data
    )


def _first_recorded_batch_seq(params_data):
    for item in params_data:
        batch_size = item.get("batch_size")
        seq_len = item.get("seq_len")
        if batch_size is not None and seq_len is not None:
            return int(batch_size), int(seq_len)
    return None


def _run_rs_case_cached(
    framework_name,
    config_file,
    model_config,
    py_func,
    batch_size,
    seq_len,
    filename,
    index,
    profile_dir,
):
    cache_key = (
        framework_name,
        config_file,
        py_func,
        batch_size,
        seq_len,
        index,
    )
    if cache_key not in executed_configs:
        print(
            f"Running {framework_name} with {config_file}, "
            f"batch_size: {batch_size} seq_len: {seq_len}."
        )
        try:
            executed_configs[cache_key] = run_rs_case(
                framework_name,
                model_config,
                py_func,
                batch_size,
                seq_len,
                filename,
                index,
                profile_dir,
            )
        except Exception as error:
            print(f"RS model execution failed: {error}")
            executed_configs[cache_key] = None
    return executed_configs[cache_key]


def rs_validate_one_inner(
    klee_function_out_dir,
    cuda_func,
    py_func,
    index,
    framework_name,
    config_file,
    profile_dir,
):
    if not klee_function_out_dir:
        return None

    index = int(index or 0)
    constraint_path = os.path.join(
        klee_function_out_dir, f"klee-out-jindex-{index}-0"
    )
    if not os.path.isdir(constraint_path):
        return None

    model_id = MODEL_IDS.get(framework_name)
    max_num_tokens, max_model_len = (
        _get_model_limits(model_id) if model_id else (None, None)
    )
    model_config = (
        _load_json(config_file)
        if config_file and os.path.exists(config_file)
        else None
    )

    input_path = _find_input_path(profile_dir, cuda_func)
    op_param_data = _load_json(input_path)
    if index >= len(op_param_data):
        raise IndexError(f"Input index {index} is out of range for {input_path}")
    target_params = op_param_data[index]["args"]

    results = {}
    if "matmul_kbit" in klee_function_out_dir:
        return results
    
    for filename in sorted(os.listdir(constraint_path)):
        if not filename.endswith(("io.txt", "oob.txt", "dr.txt")):
            continue

        buggy_source_line = filename.split("-")[0].split("_")[0]
        smt_file_path = os.path.join(constraint_path, filename)
        print(buggy_source_line, smt_file_path)
        batch_size, seq_len = solve_with_bounds(
            smt_file_path, max_num_tokens, max_model_len
        )
        common = {"config": config_file, "buggy_line": buggy_source_line}
        print("b,s:", batch_size, seq_len)
        if batch_size is None or seq_len is None:
            results[filename] = {
                "status": "failed",
                "reason": "no solution for token limit",
                **common,
            }
            continue
        if batch_size == -1 or seq_len == -1:
            continue

        params_data = _run_rs_case_cached(
            framework_name,
            config_file,
            model_config,
            py_func,
            batch_size,
            seq_len,
            filename,
            index,
            profile_dir,
        )
        run_info = {"batch_size": batch_size, "seq_len": seq_len, **common}
        if params_data is None:
            results[filename] = {
                "status": "failed",
                "reason": "kernel is not triggered",
                **run_info,
            }
        elif _find_matching_call(
            params_data, py_func, target_params, batch_size, seq_len
        ):
            results[filename] = {"status": "success", **run_info}
        else:
            fallback_pair = None
            if not _has_recorded_batch_seq(params_data, batch_size, seq_len):
                fallback_pair = _first_recorded_batch_seq(params_data)

            if fallback_pair:
                fallback_batch_size, fallback_seq_len = solve_with_bounds(
                    smt_file_path,
                    max_num_tokens,
                    max_model_len,
                    preferred_pair=fallback_pair,
                )
                if (fallback_batch_size, fallback_seq_len) == fallback_pair:
                    fallback_params_data = _run_rs_case_cached(
                        framework_name,
                        config_file,
                        model_config,
                        py_func,
                        fallback_batch_size,
                        fallback_seq_len,
                        filename,
                        index,
                        profile_dir,
                    )
                    fallback_run_info = {
                        "batch_size": fallback_batch_size,
                        "seq_len": fallback_seq_len,
                        "fallback_from": {
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                        },
                        **common,
                    }
                    if fallback_params_data is None:
                        results[filename] = {
                            "status": "failed",
                            "reason": "kernel is not triggered",
                            **fallback_run_info,
                        }
                        continue
                    if _find_matching_call(
                        fallback_params_data,
                        py_func,
                        target_params,
                        fallback_batch_size,
                        fallback_seq_len,
                    ):
                        results[filename] = {
                            "status": "success",
                            **fallback_run_info,
                        }
                        continue

            results[filename] = {
                "status": "failed",
                "reason": "Parameter mismatch",
                **run_info,
            }

    return results


def _iter_recorded_calls(data):
    """Yield (operation name, argument list) from symbolic or concrete output."""
    if isinstance(data, dict):
        for name, calls in data.items():
            if not isinstance(calls, list):
                continue
            for args in calls:
                if isinstance(args, list):
                    yield _normalise_op_name(name), args
    elif isinstance(data, list):
        for run in data:
            if not isinstance(run, dict):
                continue
            for call in run.get("calls", []):
                if isinstance(call, dict) and isinstance(call.get("args"), list):
                    yield _normalise_op_name(call.get("name", "")), call["args"]


def init_rs_input_check(profile_dir, result_path):
    results = {}
    output_dir = os.path.join(profile_dir, "out")
    kernel_map_dir = os.path.join(
        project_dir,
        "evaluation",
        "section-6-1-bug-detection",
        "intermediate_results",
        "kernel_map",
    )

    for entry in sorted(os.listdir(output_dir)):
        entry_path = os.path.join(output_dir, entry)
        framework_name = entry[:-5] if entry.endswith(".json") else entry
        kernel_map_path = os.path.join(
            kernel_map_dir, f"kernel_map_{framework_name}.json"
        )
        if not os.path.isfile(kernel_map_path):
            continue
        kernel_map = _load_json(kernel_map_path)

        if os.path.isdir(entry_path):
            profile_files = [
                os.path.join(entry_path, name)
                for name in sorted(os.listdir(entry_path))
                if name.endswith(".json")
            ]
        elif entry.endswith(".json"):
            profile_files = [entry_path]
        else:
            continue

        for profile_path in profile_files:
            config_file = None
            if os.path.dirname(profile_path) == entry_path and os.path.isdir(
                entry_path
            ):
                config_file = os.path.join(
                    profile_dir,
                    "config",
                    framework_name,
                    Path(profile_path).stem + ".json",
                )
                if not os.path.exists(config_file):
                    config_file = None

            for py_func, args in _iter_recorded_calls(_load_json(profile_path)):
                if py_func not in kernel_map:
                    continue
                cuda_func = kernel_map[py_func]["func_name"]
                input_path = _find_input_path(profile_dir, cuda_func)
                index = locate_index(args, input_path, RS_INPUT_MATCH_IGNORED)
                if index < 0:
                    continue
                kernel_result = results.setdefault(
                    cuda_func, {"py_func": py_func}
                )
                framework_result = kernel_result.setdefault(index, {}).setdefault(
                    framework_name, {}
                )
                if config_file:
                    framework_result.setdefault("config", config_file)

    _write_json(result_path, results)
    return results


def _find_klee_function_dir(klee_out_dir, cuda_func, directory_map):
    if cuda_func in directory_map:
        return os.path.join(klee_out_dir, directory_map[cuda_func])
    for dirname in os.listdir(klee_out_dir):
        if str(len(cuda_func)) + cuda_func in dirname:
            directory_map[cuda_func] = dirname
            return os.path.join(klee_out_dir, dirname)
    return None


def rs_benchmark_validate(klee_out_dir, profile_dir):
    result_path = os.path.join(profile_dir, "benchmark_validation_results.json")
    results = _load_json(result_path) if os.path.exists(result_path) else {}
    if not results:
        results = init_rs_input_check(profile_dir, result_path)

    directory_map = {}
    for dirname in os.listdir(klee_out_dir):
        cuda_func = getFuncName(dirname)
        if cuda_func:
            directory_map[cuda_func] = dirname

    for cuda_func, kernel_result in results.items():
        start_time = time.time()
        py_func = kernel_result.get("py_func", cuda_func)
        klee_function_out_dir = _find_klee_function_dir(
            klee_out_dir, cuda_func, directory_map
        )
        for index, frameworks in kernel_result.items():
            if index == "py_func":
                continue
            for framework_name, profile in frameworks.items():
                if profile is None:
                    continue
                config_file = (
                    profile.get("config") if isinstance(profile, dict) else None
                )
                frameworks[framework_name] = rs_validate_one_inner(
                    klee_function_out_dir,
                    cuda_func,
                    py_func,
                    index,
                    framework_name,
                    config_file,
                    profile_dir,
                )
                _write_json(result_path, results)
        end_time = time.time()
        print(f"Validated {cuda_func} in {end_time - start_time:.2f} seconds.")

    # _write_json(result_path, results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate research paper results."
    )
    parser.add_argument("--profile-out-dir", help="profiling backend output directory")
    parser.add_argument("--cuklee-out-dir", help="cuKLEE output directory")
    args = parser.parse_args()

    rs_benchmark_validate(args.cuklee_out_dir, args.profile_out_dir)

if __name__ == "__main__":
    # main()
    start_time = time.time()
    rs_benchmark_validate("/data/mvh6224-home/M2K-Artifact/cuKLEE/results/papers/out", "/data/mvh6224-home/M2K-Artifact/evaluation/section-6-1-bug-detection/intermediate_results/research_paper")
    end_time = time.time()
    print(f"Validation completed in {end_time - start_time:.2f} seconds.")
