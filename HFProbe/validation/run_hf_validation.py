import argparse
import os, json
import sys
import ast

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(root_dir)
backend_dir = os.path.join(root_dir, "backend")

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

if __package__ is None or __package__ == "":
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    from HFProbe.input_generate import get_max_token_vllm, get_max_model_len
    from HFProbe.validation.verify_input import solve_with_bounds, getFuncName, init_hf_input_check, compare_json_arrays
    import run_transformers_check as rt
else:
    from ..input_generate import get_max_token_vllm, get_max_model_len
    from .verify_input import solve_with_bounds, getFuncName, init_hf_input_check, compare_json_arrays
    from . import run_transformers_check as rt

def run_case(model_id, override_configs, op_name, batch_size, seq_len, lineno, index, result_dir):
    out_dir = os.path.join(result_dir, "validation", model_id.replace('/', '_'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{op_name}-{lineno}-{index}.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)

    rt.run(
        model_id=model_id,
        override_configs=override_configs,
        suffix=op_name,
        output_dir=None,
        data_dir=None,
        out_path=out_path,
        batch_size_configs=[batch_size],
        seq_lens_configs=[seq_len],
        record_only=True,
    )

    if os.path.exists(out_path):
        with open(out_path) as f:
            return json.load(f)
    
    return None

def hf_validate_one(klee_out_dir, cuda_func, py_func, index, model_id, config_file, result_dir):
    klee_function_out_path = None
    for dirname in os.listdir(klee_out_dir):
        if str(len(cuda_func)) + cuda_func in dirname:
            klee_function_out_path = os.path.join(klee_out_dir, dirname)
            break
    
    if not klee_function_out_path:
        print("no constraints files found")
        return
    
    if not index:
        index = 0
    constraint_path = os.path.join(klee_function_out_path, f"klee-out-jindex-{index}-0")
    max_num_tokens = get_max_token_vllm(model_id)
    max_model_len = get_max_model_len(model_id)
    
    model_config = None
    if os.path.exists(config_file):
        with open(config_file) as cf:
            model_config = json.load(cf)

    res = {}
    for lineno in os.listdir(constraint_path):
        if not lineno.endswith("io.txt") and not lineno.endswith("oob.txt") and not lineno.endswith("dr.txt"):
            continue

        buggy_source_line = lineno.split("-")[0].split("_")[0]
        smt_file_path = os.path.join(constraint_path, lineno)
        batch_size, seq_len = solve_with_bounds(smt_file_path, max_num_tokens, max_model_len)
        if batch_size is None or seq_len is None:
            res[lineno] = {"status": "failed", "reason": "no solution for token limit", "config": config_file, "buggy_line": buggy_source_line}
            if max_num_tokens:
                print(f"no solution found for batch_size and seq_len for {max_num_tokens} token limit.")
            else:
                print(f"no solution found for batch_size and seq_len.")
            continue
        
        if batch_size ==-1 or seq_len == -1:
            continue
        
        print(f"Running model {model_id} with {config_file}, batch_size: {batch_size} seq_len: {seq_len}.")
        params_data = run_case(model_id, model_config, py_func, batch_size, seq_len, lineno, index, result_dir)

        if params_data is None:
            error_msg = "model config is invalid"
            res[lineno] = {"status": "failed", "reason": error_msg, "config": config_file, "buggy_line": buggy_source_line}
            print(f"config_file: {config_file} is not valid for model {model_id}.")
            continue

        i = int(index)
        with open(f"{result_dir}/input/{cuda_func}.json") as f:
            op_param_data = json.load(f)
        target_params = op_param_data[i]["args"]
        
        match_found = False
        for item in params_data:
            batch_size = item["batch_size"]
            seq_len = item["seq_len"]
                
            for t in item["calls"]:
                if t["name"] != py_func:
                    continue
                if compare_json_arrays(target_params, t["args"], {"s": seq_len, "b": batch_size}):
                    match_found = True
                    break
            if match_found:
                break
        
        if match_found:
            res[lineno] = {"status": "success", "batch_size": batch_size, "seq_len": seq_len, "config": config_file, "buggy_line": buggy_source_line}
            print(f"Running model {model_id} with batch_size: {batch_size} seq_len: {seq_len} config_file: {config_file} can trigger bug {lineno} for {cuda_func}.")
        else:
            res[lineno] = {"status": "failed", "reason": "Parameter mismatch", "batch_size": batch_size, "seq_len": seq_len, "config": config_file, "buggy_line": buggy_source_line}
            print(f"Running model {model_id} with batch_size: {batch_size} seq_len: {seq_len} config_file: {config_file} can trigger {cuda_func}, but cannot trigger bug {lineno} due to different parameters.")

    with open(f"{current_dir}/{model_id.replace('/', '_')}_{cuda_func}_{index}_validate_results.json", "w") as wf:
        json.dump(res, wf, indent=2)

def hf_validate_one_inner(klee_function_out_dir, cuda_func, py_func, index, model_id, config_file, result_dir):
    if not klee_function_out_dir:
        return

    if not "/" in model_id:
        model_id = model_id.replace('_', '/', 1)
    
    if not index:
        index = 0
    constraint_path = os.path.join(klee_function_out_dir, f"klee-out-jindex-{index}-0")
    if not os.path.exists(constraint_path):
        return None
    
    max_num_tokens = get_max_token_vllm(model_id)
    max_model_len = get_max_model_len(model_id)
    
    model_config = None
    if config_file and os.path.exists(config_file):
        with open(config_file) as cf:
            model_config = json.load(cf)

    res = {}
    for lineno in os.listdir(constraint_path):
        if not lineno.endswith("io.txt") and not lineno.endswith("oob.txt") and not lineno.endswith("dr.txt"):
            continue

        buggy_source_line = lineno.split("-")[0].split("_")[0]
        smt_file_path = os.path.join(constraint_path, lineno)
        batch_size, seq_len = solve_with_bounds(smt_file_path, max_num_tokens, max_model_len)
        if batch_size is None or seq_len is None:
            res[lineno] = {"status": "failed", "reason": "no solution for token limit", "config": config_file, "buggy_line": buggy_source_line}
            continue
        
        if batch_size ==-1 or seq_len == -1:
            continue
        
        print(f"Running model {model_id} with {config_file}, batch_size: {batch_size} seq_len: {seq_len}.")
        params_data = run_case(model_id, model_config, py_func, batch_size, seq_len, lineno, index, result_dir)

        if params_data is None:
            error_msg = "model config is invalid"
            res[lineno] = {"status": "failed", "reason": error_msg, "config": config_file, "buggy_line": buggy_source_line}
            print(f"config_file: {config_file} is not valid for model {model_id}.")
            continue

        i = int(index)
        with open(f"{result_dir}/input/{cuda_func}.json") as f:
            op_param_data = json.load(f)
        target_params = op_param_data[i]["args"]
        
        match_found = False
        for item in params_data:
            batch_size = item["batch_size"]
            seq_len = item["seq_len"]
                
            for t in item["calls"]:
                if t["name"] != py_func:
                    continue
                if compare_json_arrays(target_params, t["args"], {"s": seq_len, "b": batch_size}):
                    match_found = True
                    break
            if match_found:
                break
        
        if match_found:
            res[lineno] = {"status": "success", "batch_size": batch_size, "seq_len": seq_len, "config": config_file, "buggy_line": buggy_source_line}
        else:
            res[lineno] = {"status": "failed", "reason": "Parameter mismatch", "batch_size": batch_size, "seq_len": seq_len, "config": config_file, "buggy_line": buggy_source_line}

    return res

def hf_benchmark_validate(klee_out_dir, profile_dir):
    result_path = os.path.join(profile_dir, "benchmark_validation_results.json")
    res = {}
    if os.path.exists(result_path):
        with open(result_path) as rf:
            res = json.load(rf)
    if not res:
        res = init_hf_input_check(profile_dir, None, result_path)
    
    klee_func_out_map = {}
    for dirname in os.listdir(klee_out_dir):
        cuda_func = getFuncName(dirname)
        klee_func_out_map[cuda_func] = dirname
    
    for cuda_func in res:
        py_func = res[cuda_func].get("py_func", cuda_func)
        for index in res[cuda_func]:
            if isinstance(index, str) and index == "py_func":
                continue

            for model_id in res[cuda_func][index]:
                config_file = None
                if not res[cuda_func][index][model_id]:
                    continue

                if "config" in res[cuda_func][index][model_id]:
                    config_file = res[cuda_func][index][model_id]["config"]

                klee_func_name = None
                if cuda_func in klee_func_out_map:
                    klee_func_name = klee_func_out_map[cuda_func]
                else:
                    for dirname in os.listdir(klee_out_dir):
                        if str(len(cuda_func)) + cuda_func in dirname:
                            klee_func_name = dirname
                            klee_func_out_map[cuda_func] = dirname
                            break

                klee_function_out_dir = os.path.join(klee_out_dir, klee_func_name) if klee_func_name else None
                validate_res = hf_validate_one_inner(klee_function_out_dir, cuda_func, py_func, index, model_id, config_file, profile_dir)
                res[cuda_func][index][model_id] = validate_res

    with open(result_path, "w") as wf:
        json.dump(res, wf, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate HuggingFace results."
    )

    parser.add_argument(
        "--profile-out-dir", type=str, required=False, help="out directory of profiling backend"
    )
    parser.add_argument(
        "--cuklee-out-dir", type=str, required=False, help="out directory of cuKLEE"
    )
    parser.add_argument(
        "--kernel-name", type=str, required=False, help="target kernel name"
    )
    parser.add_argument(
        "--py-func", type=str, required=False, help="python function name corresponding to the kernel"
    )
    parser.add_argument(
        "--index", type=int, required=False, default=0, help="index in the input file"
    )
    parser.add_argument(
        "--model-id", type=str, required=False, help="target model ID"
    )
    parser.add_argument(
        "--config-file", type=str, required=False, help="target model config file"
    )
    parser.add_argument(
        "--dir", action=argparse.BooleanOptionalAction, default=False, help="Whether to run on directory of cuKLEE output or on a single kernel"
    )

    args = parser.parse_args()
    if args.dir:
        hf_benchmark_validate(args.cuklee_out_dir, args.profile_out_dir)
    elif args.kernel_name and args.model_id and args.config_file:
        hf_validate_one(args.cuklee_out_dir, args.kernel_name, args.py_func, args.index, args.model_id, args.config_file, args.profile_out_dir)
