import z3
import os, re, json, time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_dir = os.path.dirname(root_dir)

def eval_expr(expr, symvals):
    """Evaluate symbolic expression safely."""
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, str):
        try:
            return eval(expr, {}, symvals)
        except NameError:
            return expr  # unresolved symbol like 'u0'
    return expr

def in_range(symbol, value, sym_ranges):
    """Check if value falls within symbol range."""
    if symbol not in sym_ranges:
        return False
    low, high = sym_ranges[symbol]
    return low <= value <= high

def compare_shapes(shape1, shape2, symvals, sym_ranges):
    if len(shape1) != len(shape2):
        return False

    for d1, d2 in zip(shape1, shape2):
        v1 = eval_expr(d1, symvals)

        if isinstance(v1, str):
            # unresolved symbol → use range
            if not in_range(v1, d2, sym_ranges):
                return False
        else:
            if int(v1) != int(d2):
                return False

    return True

def compare_tensor(t1, t2, symvals):
    if t1["type"] != t2["type"]:
        return False
    
    if "dtype" in t1 and "dtype" in t2 and t1["dtype"] != t2["dtype"]:
        if not ("float16" in t1["dtype"] and "float16" in t2["dtype"]):
            return False

    sym_ranges = t1.get("symRanges", {})

    # Compare shapes
    if "shape" in t1 and "shape" in t2:
        if not compare_shapes(t1["shape"], t2["shape"], symvals, sym_ranges):
            return False
    
    if "value" in t1 and "value" in t2:
        if t1["value"] in sym_ranges:
            if not in_range(t1["value"], t2["value"], sym_ranges):
                return False
        else:   
            v1 = eval_expr(t1["value"], symvals)
            v2 = t2["value"]
            if isinstance(v1, int):
                if int(v1) != int(v2):
                    return False
            if isinstance(v1, float) and float(v1) != float(v2):
                return False
            if v1 != v2:
                return False

    # Optional: compare min/max
    for key in ["maxV", "minV"]:
        if key in t1 and key in t2:
            if not isinstance(t1[key], str) and not isinstance(t2[key], str):
                continue
            
            v1 = eval_expr(t1[key], symvals)
            v2 = t2[key]
            if isinstance(v1, str):
                return False
            if int(v1) != int(v2):
                return False

    return True

def compare_json_arrays(arr1, arr2, symvals):
    if len(arr1) != len(arr2):
        return False

    for t1, t2 in zip(arr1, arr2):
        if not compare_tensor(t1, t2, symvals):
            return False

    return True

processed = {}
def add_smt2_constraints(solver, content):
    if hasattr(z3, "parse_smt2_string"):
        solver.add(z3.parse_smt2_string(content))
        return True

    if hasattr(solver, "from_string"):
        solver.from_string(content)
        return True

    z3_path = getattr(z3, "__file__", "<unknown>")
    print(
        "The installed z3 Python module does not support parsing SMT-LIB "
        f"strings. Imported z3 from {z3_path}."
    )
    return False

def _strip_solver_commands(content):
    content = content.replace("(check-sat)", "")
    content = content.replace("(get-model)", "")
    content = content.replace("(reset)", "")
    return content

def _find_smt_symbol(content, base_name, sort_pattern):
    pattern = re.compile(
        r"\(declare-fun\s+(" + re.escape(base_name) + r"(?:_\d+)?)\s+\(\)\s+"
        + sort_pattern + r"\)"
    )
    match = pattern.search(content)
    return match.group(1) if match else None

def _bv_array_value(name, byte_width=8):
    array = z3.Array(name, z3.BitVecSort(32), z3.BitVecSort(8))
    value = z3.Select(array, z3.BitVecVal(byte_width - 1, 32))
    for index in range(byte_width - 2, -1, -1):
        value = z3.Concat(value, z3.Select(array, z3.BitVecVal(index, 32)))
    return value

def _has_int_declaration(content):
    return re.search(r"\(declare-fun\s+\S+\s+\(\)\s+Int\)", content) is not None

def _declared_symbolic_int(content, base_name):
    int_name = _find_smt_symbol(content, base_name, r"Int")
    if int_name:
        expr = z3.Int(int_name)
        return expr, expr, False

    array_name = _find_smt_symbol(
        content, base_name, r"\(Array\s+\(_\s+BitVec\s+32\)\s+\(_\s+BitVec\s+8\)\)"
    )
    if array_name:
        bv_expr = _bv_array_value(array_name)
        return bv_expr, bv_expr, True

    return None, None, None

def _fresh_symbolic_int(base_name, use_bitvec_array):
    if use_bitvec_array:
        bv_expr = _bv_array_value(base_name)
        return bv_expr, bv_expr, True

    expr = z3.Int(base_name)
    return expr, expr, False

def _find_symbolic_inputs(content):
    batch_size, batch_size_model_expr, batch_size_is_bv = _declared_symbolic_int(
        content, "batch_size"
    )
    seq_len, seq_len_model_expr, seq_len_is_bv = _declared_symbolic_int(
        content, "seq_len"
    )

    if batch_size is not None and seq_len is not None:
        return (
            batch_size,
            batch_size_model_expr,
            batch_size_is_bv,
            seq_len,
            seq_len_model_expr,
            seq_len_is_bv,
        )

    use_bitvec_array = not _has_int_declaration(content)
    if batch_size is None:
        batch_size, batch_size_model_expr, batch_size_is_bv = _fresh_symbolic_int(
            "batch_size", use_bitvec_array
        )
    if seq_len is None:
        seq_len, seq_len_model_expr, seq_len_is_bv = _fresh_symbolic_int(
            "seq_len", use_bitvec_array
        )

    return (
        batch_size,
        batch_size_model_expr,
        batch_size_is_bv,
        seq_len,
        seq_len_model_expr,
        seq_len_is_bv,
    )

def _model_expr_as_int(model, expr):
    value = model.evaluate(expr, model_completion=True)
    if z3.is_bv_value(value):
        return value.as_long()
    if z3.is_int_value(value):
        return value.as_long()
    simplified = z3.simplify(value)
    if z3.is_bv_value(simplified) or z3.is_int_value(simplified):
        return simplified.as_long()
    return None

def _add_lower_bound(solver, expr, limit, is_bv):
    if is_bv:
        solver.add(z3.UGE(expr, z3.BitVecVal(limit, expr.size())))
    else:
        solver.add(expr >= limit)

def _add_upper_bound(solver, expr, limit, is_bv):
    if is_bv:
        width = expr.size()
        if limit < (1 << width) - 1:
            solver.add(z3.ULE(expr, z3.BitVecVal(limit, width)))
    else:
        solver.add(expr <= limit)

def _add_equal_value(solver, expr, value, is_bv):
    if is_bv:
        solver.add(expr == z3.BitVecVal(value, expr.size()))
    else:
        solver.add(expr == value)

def _add_positive_input_bounds(
    solver, batch_size, batch_size_is_bv, seq_len, seq_len_is_bv
):
    _add_lower_bound(solver, batch_size, 1, batch_size_is_bv)
    _add_lower_bound(solver, seq_len, 1, seq_len_is_bv)

def _add_token_bound(
    solver, batch_size, batch_size_is_bv, seq_len, seq_len_is_bv, num_tokens
):
    if batch_size_is_bv and seq_len_is_bv:
        width = max(batch_size.size(), seq_len.size())
        product_width = width * 2
        batch_ext = z3.ZeroExt(product_width - batch_size.size(), batch_size)
        seq_ext = z3.ZeroExt(product_width - seq_len.size(), seq_len)
        solver.add(z3.ULE(batch_ext * seq_ext, z3.BitVecVal(num_tokens, product_width)))
    else:
        solver.add(batch_size * seq_len <= num_tokens)

def solve_with_bounds(smt_file, N1, N2):
    global processed
    
    if smt_file in processed:
        if (N1, N2) in processed[smt_file]:
            return processed[smt_file][(N1, N2)]
    else:
        processed[smt_file] = {}
        
    with open(smt_file, "r") as f:
        content = f.read()
    content = _strip_solver_commands(content)
    
    s = z3.Solver()
    s.set(timeout=30000)
    if not add_smt2_constraints(s, content):
        return -1, -1

    (
        batch_size,
        batch_size_model_expr,
        batch_size_is_bv,
        seq_len,
        seq_len_model_expr,
        seq_len_is_bv,
    ) = (
        _find_symbolic_inputs(content)
    )

    # Add new constraints
    _add_positive_input_bounds(
        s, batch_size, batch_size_is_bv, seq_len, seq_len_is_bv
    )
    _add_upper_bound(s, batch_size, N1, batch_size_is_bv)
    _add_upper_bound(s, seq_len, N2, seq_len_is_bv)
    result = s.check()

    if result == z3.sat:
        model = s.model()
        bs_val = _model_expr_as_int(model, batch_size_model_expr)
        sl_val = _model_expr_as_int(model, seq_len_model_expr)
        processed[smt_file][(N1, N2)] = (bs_val, sl_val)
        return bs_val, sl_val
    elif result == z3.unknown:
        processed[smt_file][(N1, N2)] = (None, None)
        print(f"Solver returned UNKNOWN. {s.reason_unknown()}")
        return None, None
    else:
        processed[smt_file][(N1, N2)] = (None, None)
        return None, None

def solve_with_bounds(smt_file, num_tokens, max_model_len=None, preferred_pair=None):
    with open(smt_file, "r") as f:
        content = f.read()
    content = _strip_solver_commands(content)
    
    s = z3.Solver()
    s.set(timeout=30000)
    if not add_smt2_constraints(s, content):
        return -1, -1

    (
        batch_size,
        batch_size_model_expr,
        batch_size_is_bv,
        seq_len,
        seq_len_model_expr,
        seq_len_is_bv,
    ) = (
        _find_symbolic_inputs(content)
    )

    # Add new constraints
    _add_positive_input_bounds(
        s, batch_size, batch_size_is_bv, seq_len, seq_len_is_bv
    )
    if num_tokens:
        _add_token_bound(
            s, batch_size, batch_size_is_bv, seq_len, seq_len_is_bv, num_tokens
        )
    if max_model_len:
        _add_upper_bound(s, seq_len, max_model_len, seq_len_is_bv)

    if preferred_pair is not None:
        preferred_batch_size, preferred_seq_len = preferred_pair
        s.push()
        _add_equal_value(
            s, batch_size, int(preferred_batch_size), batch_size_is_bv
        )
        _add_equal_value(s, seq_len, int(preferred_seq_len), seq_len_is_bv)
        result = s.check()
        s.pop()
        if result == z3.sat:
            return int(preferred_batch_size), int(preferred_seq_len)
        if result == z3.unknown:
            print(f"Solver returned UNKNOWN for preferred pair. {s.reason_unknown()}")

    result = s.check()

    if result == z3.sat:
        model = s.model()
        bs_val = _model_expr_as_int(model, batch_size_model_expr)
        sl_val = _model_expr_as_int(model, seq_len_model_expr)
        return bs_val, sl_val
    elif result == z3.unknown:
        return None, None
    else:
        return None, None

def _read_itanium_name_component(mangled, index):
    if index >= len(mangled) or not mangled[index].isdigit():
        return None, index

    end = index
    while end < len(mangled) and mangled[end].isdigit():
        end += 1

    length = int(mangled[index:end])
    name_start = end
    name_end = name_start + length
    if name_end > len(mangled):
        return None, index

    return mangled[name_start:name_end], name_end

def getFuncName(str):
    start = str.find("_Z")
    if start < 0:
        return None

    index = start + 2

    if index < len(str) and str[index] == "N":
        index += 1
        names = []
        while index < len(str) and str[index] != "E":
            name, index = _read_itanium_name_component(str, index)
            if name is None:
                break
            names.append(name)

        if names:
            return names[-1]
    else:
        name, _ = _read_itanium_name_component(str, index)
        if name is not None:
            return name
        
    return None

vllm_ignored = {"maxV", "minV", "symRanges"}

def remove_ignored(obj, ignored=None):
    if ignored is None:
        ignored = vllm_ignored
    # Case 1: Dict → remove keys and recurse into values
    if isinstance(obj, dict):
        return {
            k: remove_ignored(v, ignored)
            for k, v in obj.items()
            if k not in ignored
        }

    # Case 2: List / tuple → recurse each element
    if isinstance(obj, (list, tuple)):
        return type(obj)(remove_ignored(v, ignored) for v in obj)

    # Case 3: Other (int, str, float, None…) → return as-is
    return obj

def locate_index(item, file_path, ignored=None):
    if not os.path.exists(file_path):
        return -1
    
    with open(file_path) as f:
        data = json.load(f)
    
    for (index, v) in enumerate(data):
        if v["args"] == item:
            return index
        
        if ignored is not None:
            if remove_ignored(v["args"], ignored) == remove_ignored(item, ignored):
                return index
    
    return -2

def scan_one_out_file(res, kernel_map, model_id, model_file, input_dir, config_dir, ignored=None):
    if "mgalkin" in model_file and "ultra" in model_file:
        return
    
    with open(model_file) as rf:
        data = json.load(rf)
    
    config_file = None
    if not model_file.endswith(model_id + ".json"):
        config_file = config_dir + "/" + os.path.basename(model_file)[:-5] + ".json"
    
    if isinstance(data, dict):
        for key in data:
            op_name = key.split(".")[-1]
            if op_name not in kernel_map:
                continue
            
            cuda_func = kernel_map[op_name]["func_name"]            
            if cuda_func not in res:
                res[cuda_func] = {}
            res[cuda_func]["py_func"] = op_name

            for item in data[key]:
                index = locate_index(item, input_dir+"/"+cuda_func+".json", ignored)
                if index >= 0:
                    if index not in res[cuda_func]:
                        res[cuda_func][index] = {}
                    if model_id not in res[cuda_func][index]:
                        res[cuda_func][index][model_id] = {}
                    if config_file is not None:
                        res[cuda_func][index][model_id]["config"] = config_file

def init_vllm_input_check(profile_out_dir, kernel_map_path, model_arch_map_path, outpath, ignored=vllm_ignored):
    if not kernel_map_path:
        kernel_map_path = f"{project_dir}/evaluation/section-6-1-bug-detection/intermediate_results/kernel_map/kernel_map_vllm.json"
    with open(kernel_map_path) as rf:
        kernel_map = json.load(rf)
    
    if not model_arch_map_path:
        model_arch_map_path = f"{project_dir}/evaluation/section-6-1-bug-detection/benchmarks/vllm/vllm_models.json"
    with open(model_arch_map_path) as mf:
        model_arch_map = json.load(mf)
    
    res = {}
    read_dir = os.path.join(profile_out_dir, "out")
    input_dir = os.path.join(profile_out_dir, "input")
    for filename in os.listdir(read_dir):
        if "seq_con.json" in filename:
            continue

        model_id = filename[:-5] if filename.endswith(".json") else filename
        config_dir = os.path.join(profile_out_dir, "config", model_arch_map[model_id.replace("_", "/", 1)])
        
        if filename.endswith(".json"):
            scan_one_out_file(res, kernel_map, model_id, read_dir+"/"+filename, input_dir, config_dir, ignored)
        else:
            for subfile in os.listdir(read_dir+"/"+model_id):
                if "seq_con.json" in subfile:
                    continue
                scan_one_out_file(res, kernel_map, model_id, read_dir+"/"+model_id+"/"+subfile, input_dir, config_dir, ignored)
    
    with open(outpath, "w") as wf:
        json.dump(res, wf, indent=4)
    
    return res

def init_hf_input_check(profile_out_dir, kernel_map_dir, outpath):
    if not kernel_map_dir:
        kernel_map_dir = f"{project_dir}/evaluation/section-6-1-bug-detection/intermediate_results/kernel_map"
    
    res = {}
    read_dir = os.path.join(profile_out_dir, "out")
    input_dir = os.path.join(profile_out_dir, "input")

    for model_id in os.listdir(read_dir):
        real_model_id = model_id[:-5] if model_id.endswith(".json") else model_id
        config_dir = os.path.join(profile_out_dir, "config", model_id)

        with open(f"{kernel_map_dir}/kernel_map_{real_model_id}.json") as rf:
            kernel_map = json.load(rf)
        
        if model_id.endswith(".json"):
            scan_one_out_file(res, kernel_map, real_model_id, read_dir+"/"+model_id, input_dir, config_dir)
        else:
            for file in os.listdir(read_dir+"/"+real_model_id):
                scan_one_out_file(res, kernel_map, real_model_id, read_dir+"/"+real_model_id+"/"+file, input_dir, config_dir)
    
    with open(outpath, "w") as wf:
        json.dump(res, wf, indent=4)
    
    return res
