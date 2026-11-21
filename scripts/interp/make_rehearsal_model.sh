#!/bin/bash
# make_rehearsal_model.sh: (re)create what rehearse_interp.sh needs, outside the repo. Free; no pod.
#   /tmp/si-interp-venv   venv: torch 2.13.0, transformers 5.17.0, xgrammar (the pod's versions)
#   /tmp/si-tiny-31b      tiny random-weight Gemma4ForConditionalGeneration with the 31B's config shape
#                         and the pinned 31B tokenizer files (checked against config/manifests)
#   /tmp/si-tiny-31b.tsv  its manifest
# Usage: bash scripts/interp/make_rehearsal_model.sh [--force]      (skips what already exists)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
VENV=/tmp/si-interp-venv; SRC=/tmp/si-tiny-31b-src; DST=/tmp/si-tiny-31b; MAN=/tmp/si-tiny-31b.tsv
REV=$(python3 -c "import json; print(next(m['revision'] for m in json.load(open('$ROOT/config/pins.json'))['models'] if m['name']=='gemma-4-31b-it'))")
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
if [ $FORCE = 1 ] || [ ! -x "$VENV/bin/python" ]; then
  rm -rf "$VENV"; python3 -m venv "$VENV" || exit 1
  perl -e 'alarm 1200; exec @ARGV' "$VENV/bin/pip" install -q "torch==2.13.0" "transformers==5.17.0" "xgrammar>=0.2.1,<1.0.0" || exit 1
fi
mkdir -p "$SRC"
for f in config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
  if [ $FORCE = 1 ] || [ ! -s "$SRC/$f" ]; then
    perl -e 'alarm 300; exec @ARGV' curl -sSfL -o "$SRC/$f" "https://huggingface.co/google/gemma-4-31B-it/resolve/$REV/$f" || { echo "download failed: $f"; exit 1; }
  fi
  want=$(awk -F'\t' -v p="$f" '$1==p {print $3}' "$ROOT/config/manifests/gemma-4-31b-it.tsv")
  got=$(shasum -a 256 "$SRC/$f" | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "MISMATCH against the manifest: $f"; exit 1; }
done
if [ $FORCE = 1 ] || [ ! -s "$DST/model.safetensors" ]; then
  perl -e 'alarm 600; exec @ARGV' "$VENV/bin/python" - "$SRC" "$DST" "$MAN" <<'EOF' || exit 1
import json, shutil, os, sys, hashlib, torch, transformers
src, dst, man = sys.argv[1:4]
shutil.rmtree(dst, ignore_errors=True)
cfg = json.load(open(f"{src}/config.json"))
cfg["text_config"].update(num_hidden_layers=6, layer_types=["sliding_attention"] * 5 + ["full_attention"], hidden_size=64,
                          intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
                          num_global_key_value_heads=1, head_dim=32, global_head_dim=64)
cfg["vision_config"].update(num_hidden_layers=1, hidden_size=32, intermediate_size=64, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=16, global_head_dim=16)
torch.manual_seed(0)
model = transformers.Gemma4ForConditionalGeneration(transformers.Gemma4Config.from_dict(cfg)).to(torch.bfloat16)
model.save_pretrained(dst)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
    shutil.copy(f"{src}/{f}", f"{dst}/{f}")
rows = [f"{f}\t{os.path.getsize(os.path.join(dst, f))}\t{hashlib.sha256(open(os.path.join(dst, f), 'rb').read()).hexdigest()}\tcomputed"
        for f in sorted(os.listdir(dst))]
open(man, "w").write("# tiny rehearsal model\n" + "\n".join(rows) + "\n")
print("tiny model:", sorted(os.listdir(dst)))
EOF
fi
"$VENV/bin/python" -c "import torch, transformers; print('ready: torch', torch.__version__, '| transformers', transformers.__version__)"
