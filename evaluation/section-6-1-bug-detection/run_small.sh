cd "$(dirname "$0")/../.."

python3 evaluation/section-6-1-bug-detection/run_small_HFProbe.py
python3 -m HFProbe.backend.config_agent_hf --run-small --use-existent-config --profile-out-dir=evaluation/section-6-1-bug-detection/small_dataset_results/huggingface
python3 -m HFProbe.backend.config_agent_rs --run-small --profile-out-dir=evaluation/section-6-1-bug-detection/small_dataset_results/papers

python3 HFProbe/input_generate.py --vllm-benchmark --add-seq-len-limit --add-memory-max-num-tokens --profile-out-dir=evaluation/section-6-1-bug-detection/small_dataset_results/vllm
python3 HFProbe/input_generate.py --hf-benchmark --add-memory-max-num-tokens --profile-out-dir=evaluation/section-6-1-bug-detection/small_dataset_results/huggingface
python3 HFProbe/input_generate.py --research-paper --profile-out-dir=evaluation/section-6-1-bug-detection/small_dataset_results/papers

python3 cuKLEE/run.py --run-small --input-dir=evaluation/section-6-1-bug-detection/small_dataset_results/vllm/input --cuklee-out-dir=cuKLEE/results-small/vllm/out --log-dir=cuKLEE/results-small/vllm/log
python3 cuKLEE/run.py --input-dir=evaluation/section-6-1-bug-detection/small_dataset_results/huggingface/input --cuklee-out-dir=cuKLEE/results-small/huggingface/out --log-dir=cuKLEE/results-small/huggingface/log
python3 cuKLEE/run.py --input-dir=evaluation/section-6-1-bug-detection/small_dataset_results/papers/input --cuklee-out-dir=cuKLEE/results-small/papers/out --log-dir=cuKLEE/results-small/papers/log

python3 HFProbe/validation/run_vllm_validation.py --dir --cuklee-out-dir=cuKLEE/results-small/vllm/out --profile-out-dir=evaluation/section-6-1-bug-detection/small_dataset_results/vllm
python3 HFProbe/validation/run_hf_validation.py --dir --cuklee-out-dir=cuKLEE/results-small/huggingface/out --profile-out-dir=evaluation/section-6-1-bug-detection/small_dataset_results/huggingface
python3 HFProbe/validation/run_rs_validation.py --dir --cuklee-out-dir=cuKLEE/results-small/papers/out --profile-out-dir=evaluation/section-6-1-bug-detection/small_dataset_results/papers
