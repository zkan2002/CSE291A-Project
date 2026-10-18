#!/usr/bin/env bash
# One-time setup of a RunPod pod for the harness (docs/environment.md, "Running on RunPod").
#
#   bash scripts/runpod-setup.sh
#
# Builds the pinned llama.cpp from source and fetches the pinned model into
# $AGENTSERVE_ARTIFACTS (default /workspace/artifacts, the network volume), reading every
# version from configs/pins.toml. Idempotent: on a new pod with the same volume it only
# reinstalls system packages and uv, then re-checks the pins.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AGENTSERVE_ARTIFACTS="${AGENTSERVE_ARTIFACTS:-/workspace/artifacts}"
UV_VERSION="0.13.0"

step() { printf '\n== %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

step "System packages"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO=sudo
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq build-essential cmake git tmux curl ca-certificates \
    libcurl4-openssl-dev libssl-dev >/dev/null
command -v nvcc >/dev/null || die "nvcc not found: redeploy the pod with a CUDA *devel* template"
nvidia-smi -L >/dev/null || die "no GPU visible to this pod"

step "Python environment (uv $UV_VERSION)"
if ! command -v uv >/dev/null || [ "$(uv --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"
uv sync -q

pin() {  # pin llama_cpp.commit -> the value in configs/pins.toml (lists joined by spaces)
    uv run -q python - "$1" <<'PY'
import sys, tomllib
value = tomllib.load(open("configs/pins.toml", "rb"))
for key in sys.argv[1].split("."):
    value = value[key]
print(" ".join(value) if isinstance(value, list) else value)
PY
}

TAG="$(pin llama_cpp.release_tag)"
COMMIT="$(pin llama_cpp.commit)"
BIN_DIR="$AGENTSERVE_ARTIFACTS/llama.cpp/$TAG"

step "llama.cpp $TAG ($COMMIT)"
if [ -x "$BIN_DIR/llama-server" ]; then
    echo "already built: $BIN_DIR/llama-server"
else
    SRC="$AGENTSERVE_ARTIFACTS/src/llama.cpp"
    # blobless, not shallow: the build number comes from the commit count
    [ -d "$SRC/.git" ] || git clone -q --filter=blob:none --no-checkout \
        https://github.com/ggml-org/llama.cpp.git "$SRC"
    git -C "$SRC" cat-file -e "$COMMIT^{commit}" 2>/dev/null || git -C "$SRC" fetch -q origin
    git -C "$SRC" checkout -q --detach "$COMMIT"
    read -r -a FLAGS <<< "$(pin llama_cpp.linux.cmake_flags)"
    cmake -S "$SRC" -B "$SRC/build" "${FLAGS[@]}"
    cmake --build "$SRC/build" --target llama-server -j "$(nproc)"
    mkdir -p "$BIN_DIR"
    cp "$SRC/build/bin/llama-server" "$BIN_DIR/"
fi

MODEL_FILE="$(pin model.file)"
MODEL_PATH="$AGENTSERVE_ARTIFACTS/models/$MODEL_FILE"
step "Model $MODEL_FILE"
if [ ! -f "$MODEL_PATH" ]; then
    mkdir -p "$(dirname "$MODEL_PATH")"
    curl -fL --retry 3 -o "$MODEL_PATH.part" \
        "https://huggingface.co/$(pin model.hf_repo)/resolve/$(pin model.hf_revision)/$MODEL_FILE"
    mv "$MODEL_PATH.part" "$MODEL_PATH"
fi
echo "$(pin model.sha256)  $MODEL_PATH" | sha256sum -c --quiet - || die "model SHA256 does not match the pin"

step "Check against the pins"
VERSION="$("$BIN_DIR/llama-server" --version 2>&1)"
echo "$VERSION"
grep -q "build ${TAG#b}, commit ${COMMIT:0:9}" <<< "$VERSION" \
    || die "llama-server reports a different build than $TAG / ${COMMIT:0:9}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvcc --version | tail -n 1
echo "model SHA256 OK"

cat <<EOF

Setup complete. Artifacts: $AGENTSERVE_ARTIFACTS
Next (inside tmux):
  uv run pytest && uv run pytest -m real_runtime
EOF
