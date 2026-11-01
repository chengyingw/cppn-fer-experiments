#!/bin/sh
# Fetches the Picbreeder genomes used by phase0/ from the FER paper's repository, at the commit these results used.
set -e
COMMIT=67c270c197d8243435d6a9ad036e26c9c5e3712d
mkdir -p refs
git -c core.longpaths=true clone --filter=blob:none --no-checkout https://github.com/akarshkumar0101/fer.git refs/fer_repo
git -c core.longpaths=true -C refs/fer_repo sparse-checkout set picbreeder_genomes
git -c core.longpaths=true -C refs/fer_repo checkout "$COMMIT"
