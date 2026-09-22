#!/usr/bin/env bash
# 租机一键脚本：装环境 → 编译 → Rust 单测 → e2e(A 段) → 完整测试类(B 段)
# 用法（先在数据盘 clone 本分支，再在仓库根目录执行）:
#   git clone -b parallel-sampling-e2e --depth 1 https://github.com/Kare0638/sglang.git /数据盘/sglang
#   DATA=/数据盘 bash /数据盘/sglang/e2e_tmp/run_remote.sh [setup|rust|a|b|all]
# 更新代码: cd /数据盘/sglang && git pull
# 每一步都会把日志写到 $DATA/logs/，出错时把对应日志发给 Claude。
set -euo pipefail

DATA=${DATA:-}
if [ -z "$DATA" ]; then
  for d in /root/autodl-tmp /data /mnt/data /root/data; do [ -d "$d" ] && DATA=$d && break; done
fi
[ -n "$DATA" ] || { echo "✗ 找不到数据盘，请用 DATA=/路径 bash run_remote.sh 指定"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/.." && pwd)
LOG=$DATA/logs
export HF_HOME=$DATA/huggingface UV_CACHE_DIR=$DATA/uv-cache CARGO_HOME=$DATA/cargo RUSTUP_HOME=$DATA/rustup
export PATH="$SRC/.venv/bin:$CARGO_HOME/bin:$HOME/.local/bin:$PATH"
mkdir -p "$LOG" "$HF_HOME"
STEP=${1:-all}
MODEL=meta-llama/Llama-3.2-1B-Instruct

setup() {
  echo "== 0. 检查 GPU / CUDA 驱动"
  SMI=$(nvidia-smi)  # 不用 `nvidia-smi | head`：pipefail 下 head 提前关管道会让整条命令失败
  echo "$SMI" | sed -n 1,12p
  grep -q "CUDA Version: 1[3-9]" <<<"$SMI" || { echo "✗ 驱动不支持 CUDA 13，换镜像/机器"; exit 1; }

  echo "== 1. 系统依赖"
  if command -v apt-get >/dev/null; then
    SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git curl build-essential pkg-config libssl-dev python3-dev >/dev/null
  fi

  echo "== 2. 代码: $SRC @ $(git -C "$SRC" log --oneline -1)"

  echo "== 3. Rust 1.92"
  command -v rustup >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92 --profile minimal -c clippy -c rustfmt
  rustup default 1.92

  echo "== 4. Python 环境 + 编译 sglang（含 Rust server，约 20-40 分钟）"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  cd "$SRC"
  [ -d .venv ] || uv venv -p 3.12 .venv
  VIRTUAL_ENV=$SRC/.venv SGLANG_BUILD_RUST_EXTS=server uv pip install -e python/ 2>&1 | tee "$LOG/install.log" | tail -5
  python -c "import sglang, flashinfer; print('sglang', sglang.__version__, '| flashinfer', flashinfer.__version__)"

  echo "== 5. HuggingFace 登录检查（token 请你自己 huggingface-cli login，不要发给任何人）"
  python - <<EOF
import sys
from huggingface_hub import whoami, hf_hub_download
try:
    print("已登录:", whoami()["name"])
except Exception:
    sys.exit("✗ 未登录：先执行  $SRC/.venv/bin/hf auth login（旧版叫 huggingface-cli login）再重跑 setup")
hf_hub_download("$MODEL", "config.json")
print("✓ 有 $MODEL 访问权限")
EOF
  python -c "from huggingface_hub import snapshot_download as s; s('$MODEL', allow_patterns=['*.json','*.safetensors','tokenizer*'])" >/dev/null
  echo "✓ setup 完成"
}

rust_tests() {
  echo "== Rust: fmt / clippy / 单测"
  cd "$SRC/rust"
  cargo fmt --all -- --check
  cargo clippy -p sglang-server --all-targets -- -D warnings 2>&1 | tail -3
  cargo test -p sglang-server 2>&1 | tee "$LOG/cargo_test.log" | grep -E "^test result|FAILED|panicked" || true
}

wait_ready() {  # $1=pid $2=log
  for _ in $(seq 1 300); do
    grep -q "fired up and ready" "$2" && return 0
    kill -0 "$1" 2>/dev/null || { echo "✗ 服务器退出了，看 $2"; tail -30 "$2"; return 1; }
    sleep 2
  done
  echo "✗ 10 分钟没起来"; return 1
}

phase_a() {
  echo "== A 段：默认后端(flashinfer) 起 Rust server，跑 12 项 e2e 检查"
  cd "$SRC"
  SGLANG_RUST_SERVER=1 setsid python -m sglang.launch_server --model $MODEL --port 30000 > "$LOG/e2e_server.log" 2>&1 &
  PID=$!
  trap "kill -- -$PID 2>/dev/null || true" EXIT
  wait_ready $PID "$LOG/e2e_server.log"
  python "$HERE/e2e_review.py" 2>&1 | tee "$LOG/e2e_review.log" || true
  kill -- -$PID; wait $PID 2>/dev/null || true; trap - EXIT
  sleep 10  # 等显存释放，B 段要自己起服务器
}

phase_b() {
  echo "== B 段：完整 TestRustServerEndpoint（测试自己起服务器）"
  pgrep -f "sglang.launch_server" && { echo "✗ 还有服务器在跑，先停掉"; exit 1; }
  cd "$SRC/test/registered/core"
  python -m unittest -v test_srt_endpoint.TestRustServerEndpoint 2>&1 | tee "$LOG/test_rust_endpoint.log" | grep -E "\.\.\. (ok|FAIL|ERROR|skipped)|^Ran |^OK|^FAILED" || true
}

case $STEP in
  setup) setup ;;
  rust) rust_tests ;;
  a) phase_a ;;
  b) phase_b ;;
  all) setup; rust_tests; phase_a; phase_b ;;
  *) echo "用法: bash run_remote.sh [setup|rust|a|b|all]"; exit 1 ;;
esac
echo "日志都在 $LOG/"
