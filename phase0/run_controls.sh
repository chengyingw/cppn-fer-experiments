#!/bin/sh
# Sparse SGD controls, one job at a time, then measurement and verdicts.
for job in "skull G1a" "skull G1b" "skull G4" "butterfly G1b" "butterfly G4" "apple G1b" "apple G4"; do
  set -- $job
  python train_control.py --name "$1" --group "$2" || echo "FAILED $job"
done
python eval_controls.py && cd ../kb && python verify_kb.py --settle
