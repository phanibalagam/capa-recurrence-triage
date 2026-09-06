#!/usr/bin/env bash
# Regenerate every number reported in METHODS.md and README.md.
#
# The analysis corpus ships with this repository as data/deviations_maude.jsonl
# (635 filing events, openFDA, CC0). No network access is needed to reproduce
# any reported figure.
#
# To rebuild that corpus from scratch instead, run `python3 run.py fetch` from a
# machine with access to api.fda.gov, then `python3 run.py setup --source maude`.
# Note that openFDA is a live database: a later fetch will not return the same
# records, so the shipped corpus is the one the reported numbers come from.
set -euo pipefail
cd "$(dirname "$0")"
echo "python: $(python3 --version)"
mkdir -p results
if [ ! -f data/deviations_maude.jsonl ]; then
  echo "data/deviations_maude.jsonl missing. It ships with the repository;" >&2
  echo "to rebuild it: python3 run.py fetch && python3 run.py setup --source maude" >&2
  exit 1
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
echo ">>> run.py capa-risk --source maude --by action_class"
python3 run.py capa-risk --source maude --by action_class
echo
echo ">>> run.py evaluate --source maude"
python3 run.py evaluate --source maude
echo
echo ">>> run.py experiments --source maude --n-boot 200"
python3 run.py experiments --source maude --n-boot 200
echo
echo ">>> run.py experiments --source maude --adjusted --n-boot 800"
python3 run.py experiments --source maude --adjusted --n-boot 800
echo
echo ">>> run.py experiments --source maude --stratified --n-boot 800"
python3 run.py experiments --source maude --stratified --n-boot 800
echo
echo ">>> run.py experiments --source maude --ablation"
python3 run.py experiments --source maude --logo
echo
echo ">>> run.py experiments --source maude --ablation"
python3 run.py experiments --source maude --ablation
echo
echo "Done. Reported numbers are in results/."
