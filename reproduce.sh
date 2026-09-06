#!/usr/bin/env bash
# Regenerate every number reported in METHODS.md and README.md.
# Requires data/maude_raw.jsonl - run `python3 run.py fetch` first,
# from a machine with access to api.fda.gov.
set -euo pipefail
cd "$(dirname "$0")"
echo "python: $(python3 --version)"
mkdir -p results
if [ ! -f data/maude_raw.jsonl ]; then
  echo "data/maude_raw.jsonl missing - run: python3 run.py fetch" >&2; exit 1;
fi
echo
echo ">>> run.py setup --source synthetic"
python3 run.py setup --source synthetic
echo
echo ">>> run.py capa-risk --source synthetic"
python3 run.py capa-risk --source synthetic
echo
echo ">>> run.py evaluate --source synthetic"
python3 run.py evaluate --source synthetic
echo
echo ">>> run.py experiments --source synthetic --seeds 10"
python3 run.py experiments --source synthetic --seeds 10
echo
echo ">>> run.py experiments --source synthetic --ablation --seeds 10"
python3 run.py experiments --source synthetic --ablation --seeds 10
echo
echo ">>> run.py setup --source maude"
python3 run.py setup --source maude
echo
echo ">>> run.py capa-risk --source maude --by action_class"
python3 run.py capa-risk --source maude --by action_class
echo
echo ">>> run.py evaluate --source maude"
python3 run.py evaluate --source maude
echo
echo ">>> run.py experiments --source maude --n-boot 200"
python3 run.py experiments --source maude --n-boot 200
echo
echo ">>> run.py experiments --source maude --adjusted --n-boot 400"
python3 run.py experiments --source maude --adjusted --n-boot 400
echo
echo ">>> run.py experiments --source maude --stratified --n-boot 400"
python3 run.py experiments --source maude --stratified --n-boot 400
echo
echo ">>> run.py experiments --source maude --ablation"
python3 run.py experiments --source maude --ablation
echo
echo "Done. Reported numbers are in results/."
