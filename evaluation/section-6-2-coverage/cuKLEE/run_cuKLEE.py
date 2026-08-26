from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import subprocess
from pathlib import Path
import sys


CUDA_PATH = "/usr/local/cuda"  # Path to CUDA installation
COMBINED_SUFFIX = "_combined.bc"  # Suffix for the combined .bc files
project_path = Path(__file__).resolve().parent.parent.parent.parent
compile_include_path = os.path.join(project_path, "cuKLEE", "include")
current_section_dir = Path(__file__).resolve().parent.parent
signed_clang_path = os.getenv("SIGNED_CLANG_PATH", "clang++-13")

# Manually define PyTorch include paths
try:
    import torch
    TORCH_INCLUDE = os.path.join(os.path.dirname(torch.__file__), "include")
    print(TORCH_INCLUDE)
    if not os.path.exists(TORCH_INCLUDE):
        raise FileNotFoundError(f"PyTorch include directory not found at {TORCH_INCLUDE}")
except ImportError:
    print("PyTorch is not installed. Please install it and try again.")
    sys.exit(1)

OPENMP_INCLUDE_DIRS = [
    Path("/usr/lib/gcc/x86_64-linux-gnu/11/include"),
    Path("/usr/lib/gcc/x86_64-linux-gnu/12/include"),
    Path("/usr/lib/gcc/x86_64-linux-gnu/13/include"),
]

def _resolve_first_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None

OPENMP_INCLUDE_DIR = _resolve_first_existing_path(OPENMP_INCLUDE_DIRS)


# Utility function to run shell commands
def run_command(command, cwd=None):
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        # print(f"Error: Command failed with return code {result.returncode}")
        # print(f"Stdout: {result.stdout}")
        # print(f"Stderr: {result.stderr}")
        return False
    else:
        return True


def compile_cu_file(cu_file, root_path):
    print(f"Compiling {cu_file}...")
    if not root_path:
        root_path = os.path.dirname(cu_file)

    clang_command = [
        signed_clang_path,
        "-g",
        "-x", "cuda", 
        "--cuda-gpu-arch=sm_80",  
        "-std=c++17",
        "-Xclang",
        "-fcuda-allow-variadic-functions",
        "-D__CUDA_ARCH__=800",
        "-I", f"{CUDA_PATH}/include",
        "-I", TORCH_INCLUDE,
        "-I", TORCH_INCLUDE + "/torch/csrc/api/include",
        *(["-I", str(OPENMP_INCLUDE_DIR)] if OPENMP_INCLUDE_DIR is not None else []),
        "-I", root_path,
        "-I", "/usr/include/python3.10",
        "-I", f"{compile_include_path}/cutlass/include",
        "-I", compile_include_path,
        "-I", f"{compile_include_path}/cutlass/examples",
    ]

    if "cutlass" in cu_file:
        clang_command.append("-DENABLE_SCALED_MM_SM90=1")
        clang_command.append("-DENABLE_SCALED_MM_SM100=1")
    
    clang_command.extend(["-emit-llvm",
        "-c",
        cu_file])
    
    if "rocm" in cu_file:
        return False  # Skip compilation for ROCm files
    
    return run_command(clang_command)

def compile_original():
    original_dir = os.getcwd()
    root_dir = os.path.join(current_section_dir, "benchmarks", "original")
    for dname in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dname)
        cu_dir = dir_path
        if dname == "5169":
            cu_dir = os.path.join(dir_path, "punica")
        if dname == "6649":
            cu_dir = os.path.join(dir_path, "quantization", "fp8")
        if dname == "9391":
            cu_dir = os.path.join(dir_path, "quantization", "compressed_tensors")
        if dname == "9425":
            cu_dir = os.path.join(dir_path, "quantization", "fp8")
        if dname == "12413":
            cu_dir = os.path.join(dir_path, "moe")

        os.chdir(dir_path)
        for file in os.listdir(cu_dir):
            if file.endswith(".cu"):
                output_prefix = dir_path+"/"+file[:file.find(".cu")]
                host_bc_file = f"{output_prefix}.bc"
                cuda_bc_file = f"{output_prefix}-cuda-nvptx64-nvidia-cuda-sm_80.bc"
                combined_bc_file = f"{output_prefix}{COMBINED_SUFFIX}"
                if not os.path.exists(cuda_bc_file) or not os.path.exists(host_bc_file):
                    success = compile_cu_file(os.path.join(cu_dir, file), root_path=cu_dir)
                    if not success:
                        print(f"Failed to compile {file} in {dir_path}")
                        continue

                if os.path.exists(combined_bc_file):
                    continue

                run_command([
                    "llvm-link-13",
                    "-o", combined_bc_file,
                    host_bc_file,
                    cuda_bc_file
                ])
                run_command([
                    "llvm-dis-13",
                    combined_bc_file
                ])

    os.chdir(original_dir)

def compile_simplified():
    original_dir = os.getcwd()
    root_dir = os.path.join(current_section_dir, "benchmarks", "cuKLEE-simplified")
    os.chdir(root_dir)
    for file in os.listdir(root_dir):
        if file.endswith(".cu"):
            output_prefix = root_dir+"/"+file[:file.find(".cu")]
            host_bc_file = f"{output_prefix}.bc"
            cuda_bc_file = f"{output_prefix}-cuda-nvptx64-nvidia-cuda-sm_80.bc"
            combined_bc_file = f"{output_prefix}{COMBINED_SUFFIX}"
            if not os.path.exists(cuda_bc_file) or not os.path.exists(host_bc_file):
                success = compile_cu_file(os.path.join(root_dir, file), root_path=root_dir)
                if not success:
                    print(f"Failed to compile {file}")
                    continue

            if os.path.exists(combined_bc_file):
                continue

            run_command([
                "llvm-link-13",
                "-o", combined_bc_file,
                host_bc_file,
                cuda_bc_file
            ])
            run_command([
                "llvm-dis-13",
                combined_bc_file
            ])

    os.chdir(original_dir)


import subprocess
import logging

TIMEOUT_LIMIT = 6*60*60

def run_klee_on_json_file(json_file, logDir, outputdir, useDirName=False):
    one_timeout = 3600
    try:
        # Ensure output directory exists
        os.makedirs(logDir, exist_ok=True)
        
        # Create a log file for this specific KLEE run (output and error)
        if useDirName:
            dir_name = os.path.basename(os.path.dirname(json_file))
            log_file = os.path.join(logDir, dir_name + '_cuklee_output.log')
            outputdir = os.path.join(outputdir, dir_name)
            os.makedirs(outputdir, exist_ok=True)
        else:
            log_file = os.path.join(logDir, os.path.splitext(os.path.basename(json_file))[0] + '_cuklee_output.log')

        if os.path.exists(log_file):
            print(f"Log file {log_file} already exists. Skipping run for {json_file}.")
            return True         
        
        print(f"Running cuKLEE on {json_file}")
        # Run KLEE and capture its output and error in the log file
        with open(log_file, 'w') as output_file:
            cmd = ['cuKLEE', f"--timeout={one_timeout}", f"--cuklee-out-dir={outputdir}", "--use-merge=false"]
            if "9838" in json_file:
                cmd.append("--loop-max=4")
            cmd.append(json_file)
            subprocess.run(cmd, stdout=output_file, stderr=output_file, timeout=TIMEOUT_LIMIT, check=True)
        
        print(f"Output saved to {log_file}")
    
    except Exception as e:
        pass
    except subprocess.TimeoutExpired:
        pass
    except subprocess.CalledProcessError as e:
        pass
    return True

def run_klee_on_bc_file(bc_file, logDir, outputdir):
    one_timeout = 3600
    try:
        os.makedirs(logDir, exist_ok=True)

        log_file = os.path.join(logDir, os.path.splitext(os.path.basename(bc_file))[0] + '_cuklee_output.log')
        if os.path.exists(log_file):
            print(f"Log file {log_file} already exists. Skipping run for {bc_file}.")
            return True         

        outputdir = os.path.join(outputdir, os.path.splitext(os.path.basename(bc_file))[0])
        os.makedirs(outputdir, exist_ok=True)
        
        print(f"Running cuKLEE on {bc_file}")
        # Run KLEE and capture its output and error in the log file
        with open(log_file, 'w') as output_file:
            cmd = ['cuKLEE', f"--timeout={one_timeout}", f"--cuklee-out-dir={outputdir}", "--use-merge=false"]
            if "9838" in bc_file:
                cmd.append("--loop-max=4")
            cmd.append(bc_file)
            subprocess.run(cmd, stdout=output_file, stderr=output_file, check=True)
        
        print(f"Output saved to {log_file}")
    
    except Exception as e:
        pass
    except subprocess.TimeoutExpired:
        pass
    except subprocess.CalledProcessError as e:
        pass
    return True

def main(input_files, logDir, outputdir, isJson=True, max_processes=5, useDirName=False):
    
    with ProcessPoolExecutor(max_processes) as executor:
        if isJson:
            future_to_file = {executor.submit(run_klee_on_json_file, json_file, logDir, outputdir, useDirName): json_file for json_file in input_files} # if json_file not in filter_files
        else:
            future_to_file = {executor.submit(run_klee_on_bc_file, bc_file, logDir, outputdir): bc_file for bc_file in input_files} # if bc_file not in filter_files
        
        for future in as_completed(future_to_file):
            json_file = future_to_file[future]
            try:
                success = future.result()
            except Exception as e:
                pass

def run_original():
    compile_original()
    input_dir = os.path.join(current_section_dir, "benchmarks", "original")
    logDir = os.path.join(current_section_dir, "cuKLEE/log-original")
    outputdir = os.path.join(current_section_dir, "cuKLEE/output-original")
    input_files = []

    for dname in os.listdir(input_dir):
        dir_path = os.path.join(input_dir, dname)
        for file in os.listdir(dir_path):
            if file.endswith(".json"):
                json_file = os.path.join(dir_path, file)
                input_files.append(json_file)

    main(input_files, logDir, outputdir, isJson=True, useDirName=True)

def run_simplified():
    compile_simplified()
    input_dir = os.path.join(current_section_dir, "benchmarks", "cuKLEE-simplified")
    logDir = os.path.join(current_section_dir, "cuKLEE/log-simplified")
    outputdir = os.path.join(current_section_dir, "cuKLEE/output-simplified")
    processed = []

    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            json_file = os.path.join(input_dir, file)
            run_klee_on_json_file(json_file, logDir, outputdir)
            processed.append(file.split('.')[0])
        elif file.endswith(COMBINED_SUFFIX):
            bc_file = os.path.join(input_dir, file)
            if file.split('_combined')[0][4:] not in processed:
                run_klee_on_bc_file(bc_file, logDir, outputdir)


if __name__ == "__main__":
    run_original()
    run_simplified()
