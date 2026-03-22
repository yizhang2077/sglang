## Usage
### When will it can accelerate
```
normal_decode_time * step / accept_rate + deterministic_prefill_verify_time < deterministic_decode_time * step
```
### Test Correctness
```
# test1

USE_CHAIN_SPECULATIVE_SAMPLING=1 USE_DECODE_TOPK_RENORM=1  python3 -m sglang.launch_server --model Qwen/Qwen3-30B-A3B --speculative-algorithm DECODE_VERIFY_ROLLBACK --speculative-num-steps 15 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16  --max-running-requests 256  --decode-log-interval 1  --enable-deterministic-inference --disable-overlap-schedule --mem-frac 0.65 --rl-on-policy-target fsdp --tp 2

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 32 --random-input 1024 --random-output 6000 --random-range-ratio 1
```
accept rate should be also 1 except when a request is finished

```
# test2

USE_CHAIN_SPECULATIVE_SAMPLING=1 python3 -m sglang.launch_server --model Qwen/Qwen3-30B-A3B --speculative-algorithm DECODE_VERIFY_ROLLBACK --speculative-num-steps 15 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16  --max-running-requests 256  --decode-log-interval 1  --enable-prefill-only-deterministic-inference --disable-overlap-schedule --mem-frac 0.65 --rl-on-policy-target fsdp --tp 2

python3 dvr_test.py
```

### Inference
```
USE_CHAIN_SPECULATIVE_SAMPLING=1 python3 -m sglang.launch_server --model Qwen/Qwen3-30B-A3B --speculative-algorithm DECODE_VERIFY_ROLLBACK --speculative-num-steps 15 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16  --max-running-requests 256  --decode-log-interval 1  --enable-prefill-only-deterministic-inference --disable-overlap-schedule --mem-frac 0.65 --rl-on-policy-target fsdp --tp 2
```

You could change **--speculative-num-steps** or **--speculative-num-draft-tokens** according to accept rate. It can be faster than deterministic inference especially in small batch size

### Performance Test
```
USE_CHAIN_SPECULATIVE_SAMPLING=1 python3 -m sglang.launch_server --model Qwen/Qwen3-30B-A3B --speculative-algorithm DECODE_VERIFY_ROLLBACK --speculative-num-steps 15 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16  --max-running-requests 256  --decode-log-interval 1  --enable-prefill-only-deterministic-inference --disable-overlap-schedule --mem-frac 0.65 --rl-on-policy-target fsdp --tp 2

python3 -m sglang.launch_server --model Qwen/Qwen3-30B-A3B --max-running-requests 256 --enable-deterministic-inference --disable-overlap-schedule --mem-frac 0.65 --rl-on-policy-target fsdp --tp 2

python3 -m sglang.bench_serving --backend sglang --dataset-name random --num-prompt 16 --random-input 1024 --random-output 12000 --random-range-ratio 1
```

### Reproduce Command for qwen3next
```
SGLANG_GDN_PREFILL_TRUNCATION_ALIGN_SIZE=128 python -m sglang.launch_server --model-path Qwen/Qwen3-Next-80B-A3B-Instruct --tp 2  --speculative-num-steps 15 --speculative-eagle-topk 1 --speculative-num-draft-tokens 16 --speculative-algo DECODE_VERIFY_ROLLBACK  --mamba-scheduler-strategy extra_buffer --disable-overlap-schedule  --mamba-track-interval 64  --mem-frac 0.8   --max-running-requests 48
```
