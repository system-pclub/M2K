import argparse
import json
import os
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

TIMEOUT_LIMIT = 6*60*60
MAX_ENTRIES = 4

def run_klee_on_json_file(json_file, logDir, outputdir, useDirName=False):
    if "swap_blocks" in json_file:
        return True
    
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
            subprocess.run(['cuKLEE', f"--timeout={one_timeout}", f"--cuklee-out-dir={outputdir}", json_file], stdout=output_file, stderr=output_file, timeout=TIMEOUT_LIMIT, check=True)
        
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
            subprocess.run(['cuKLEE', f"--timeout={one_timeout}", f"--cuklee-out-dir={outputdir}", bc_file], stdout=output_file, stderr=output_file, check=True) 
        
        print(f"Output saved to {log_file}")
    
    except Exception as e:
        pass
    except subprocess.TimeoutExpired:
        pass
    except subprocess.CalledProcessError as e:
        pass
    return True

def main(directory, logDir, outputdir, isJson=True, max_processes=5, json_files=None, useDirName=False):
    failed_files = []
    if json_files is None:
        if isJson:
            json_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.json')]
        else:
            json_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('_combined.bc')]
    
    run_func = run_klee_on_json_file if isJson else run_klee_on_bc_file
    with ProcessPoolExecutor(max_processes) as executor:
        if isJson:
            future_to_file = {executor.submit(run_func, json_file, logDir, outputdir, useDirName): json_file for json_file in json_files}
        else:
            future_to_file = {executor.submit(run_func, json_file, logDir, outputdir): json_file for json_file in json_files}
        
        for future in as_completed(future_to_file):
            json_file = future_to_file[future]
            try:
                success = future.result()
            except Exception as e:
                pass

def run_small(directory, logDir, outputdir, max_processes=5):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(
        project_dir,
        "evaluation",
        "section-6-1-bug-detection",
        "small_dataset.json",
    )

    with open(dataset_path, encoding="utf-8") as dataset_file:
        small_dataset = json.load(dataset_file)

    selected_names = set(small_dataset)
    json_files = sorted(
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".json")
        and os.path.splitext(filename)[0] in selected_names
    )

    main(
        directory,
        logDir,
        outputdir,
        isJson=True,
        max_processes=max_processes,
        json_files=json_files,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run cuKLEE"
    )

    parser.add_argument(
        "--input-dir", type=str, required=True, help="Input directory containing .json files or .bc files"
    )
    parser.add_argument(
        "--cuklee-out-dir", type=str, required=False, help="Output directory"
    )
    parser.add_argument(
        "--log-dir", type=str, required=False, help="Log directory"
    )
    parser.add_argument(
        "--is-json", action=argparse.BooleanOptionalAction, default=True, help="Whether the input files are JSON"
    )
    parser.add_argument(
        "--threads", type=int, required=False, default=5, help="Number of threads to use"
    )
    parser.add_argument(
        "--run-small", action=argparse.BooleanOptionalAction, default=False, help="run the small dataset"
    )
    args = parser.parse_args()

    if args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"Input directory {args.input_dir} does not exist.")
            exit(1)
        
        current_dirpath = os.path.dirname(os.path.abspath(__file__))
        output_directory = args.cuklee_out_dir or os.path.join(current_dirpath, "output")
        log_directory = args.log_dir or os.path.join(current_dirpath, "vllm_log")
        os.makedirs(output_directory, exist_ok=True)
        threads = args.threads if args.threads is not None else 5
        if args.run_small:
            run_small(args.input_dir, log_directory, output_directory, threads)
        else:
            main(args.input_dir, log_directory, output_directory, args.is_json, threads)
