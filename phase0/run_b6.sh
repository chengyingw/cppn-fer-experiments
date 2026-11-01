#!/bin/sh
# Fixed butterfly batch b6 (seeds 3-8), then measurement and verdicts. Failed jobs are reported, not retried.
OUT=runs/b6
SEEDS="3 4 5 6 7 8"
mkdir -p "$OUT"
python train_sgd.py --name butterfly --seeds $SEEDS --iters 50000 --res 128 --out "$OUT" || echo "FAILED G0"
python train_early.py --name butterfly --seeds $SEEDS --out "$OUT" || echo "FAILED G0e"
for group in G1b G1a G4; do
  python train_control.py --name butterfly --group "$group" --seeds $SEEDS --out "$OUT" || echo "FAILED $group"
done
python eval_controls.py && cd ../kb && python verify_kb.py --settle
