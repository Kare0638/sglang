#!/usr/bin/env bash
# n=1 性能回退 A/B：同一台机器、同一份 Python，只切换 rust/ 源码（base = 分支基点，head = 本分支）。
# 顺序 base,head,base,head 交替跑，用来看出机器本身的漂移。
# 用法: DATA=/data bash e2e_tmp/bench_ab.sh
set -euo pipefail
DATA=${DATA:-/data}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/.." && pwd)
BASE=1da8ac10e17baa394c44dafa173382f9ab03d8e9
OUT=$DATA/bench_ab
export HF_HOME=$DATA/huggingface CARGO_HOME=$DATA/cargo RUSTUP_HOME=$DATA/rustup
export PATH="$SRC/.venv/bin:$CARGO_HOME/bin:$PATH"
[ -d /usr/local/cuda-13.0 ] && export CUDA_HOME=/usr/local/cuda-13.0 PATH="/usr/local/cuda-13.0/bin:$PATH"
MODEL=meta-llama/Llama-3.2-1B-Instruct
mkdir -p "$OUT"
cd "$SRC"
git cat-file -e "$BASE^{commit}" 2>/dev/null || git fetch -q --deepen 10 origin
git cat-file -e "$BASE^{commit}"

# 两种负载：
#   throughput: 常规吞吐（in 256 / out 128, 并发 64）
#   frontend:   前端压力（in 32 / out 4, 并发 256）——输出极短，GPU 时间少，前端开销占比最大
bench() {  # $1=variant $2=round
  for cfg in "throughput 1000 256 128 64" "frontend 3000 32 4 256"; do
    set -- "$1" "$2" $cfg
    python -m sglang.benchmark.serving --backend sglang --port 30000 --model $MODEL \
      --dataset-name random-ids --num-prompts $4 --random-input-len $5 --random-output-len $6 \
      --random-range-ratio 1.0 --max-concurrency $7 --seed 1 --disable-tqdm \
      --output-file "$OUT/$3.jsonl" > "$OUT/${3}_$1_r$2.log" 2>&1
    python - "$OUT/$3.jsonl" "$1" "$2" <<'EOF'
import json, sys
p, variant, rnd = sys.argv[1:]
lines = open(p).read().strip().splitlines()
d = json.loads(lines[-1]); d["variant"] = variant; d["round"] = int(rnd)
lines[-1] = json.dumps(d); open(p, "w").write("\n".join(lines) + "\n")
EOF
  done
}

run_variant() {  # $1=variant $2=round
  if [ "$1" = base ]; then git checkout -q "$BASE" -- rust/; else git checkout -q HEAD -- rust/; fi
  python -c "from sglang.srt.rust_extensions.loader import load_rust_extension as l; l('sglang.srt.rust_extensions._server')" 2>&1 | grep -E "Finished|error" || true
  SGLANG_RUST_SERVER=1 setsid python -m sglang.launch_server --model $MODEL --port 30000 > "$OUT/server_$1_r$2.log" 2>&1 &
  local pid=$!
  trap "kill -- -$pid 2>/dev/null || true; git -C '$SRC' checkout -q HEAD -- rust/" EXIT
  for _ in $(seq 1 300); do grep -q "fired up and ready" "$OUT/server_$1_r$2.log" && break; kill -0 $pid || { tail -20 "$OUT/server_$1_r$2.log"; exit 1; }; sleep 2; done
  echo "== $1 round $2 ($(git -C "$SRC" diff --quiet HEAD -- rust/ && echo 'rust/=HEAD' || echo 'rust/=BASE'))"
  bench "$1" "$2"
  kill -- -$pid; wait $pid 2>/dev/null || true; trap - EXIT; sleep 10
}

rm -f "$OUT"/*.jsonl
run_variant base 1; run_variant head 1; run_variant base 2; run_variant head 2
git checkout -q HEAD -- rust/

python - "$OUT" <<'EOF'
import json, sys, statistics as st
out = sys.argv[1]
keys = [("request_throughput", "req/s", 1), ("output_throughput", "out tok/s", 1),
        ("median_ttft_ms", "median TTFT ms", -1), ("p99_ttft_ms", "p99 TTFT ms", -1),
        ("median_e2e_latency_ms", "median E2E ms", -1)]
for cfg in ("throughput", "frontend"):
    rows = [json.loads(l) for l in open(f"{out}/{cfg}.jsonl")]
    print(f"\n[{cfg}]  {'metric':<16}{'base r1':>10}{'base r2':>10}{'head r1':>10}{'head r2':>10}{'head vs base':>14}")
    for k, name, better in keys:
        b = [r[k] for r in rows if r["variant"] == "base"]
        h = [r[k] for r in rows if r["variant"] == "head"]
        delta = (st.mean(h) - st.mean(b)) / st.mean(b) * 100
        print(f"{'':10}{name:<16}" + "".join(f"{v:>10.1f}" for v in b + h) + f"{delta:>+13.1f}%")
EOF
