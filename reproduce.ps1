# Regenerate every number reported in METHODS.md and README.md.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python --version
New-Item -ItemType Directory -Force -Path results | Out-Null
if (-not (Test-Path data/maude_raw.jsonl)) {
  Write-Error "data/maude_raw.jsonl missing - run: python run.py fetch"; exit 1 }
Write-Host ""; Write-Host ">>> run.py setup --source synthetic"
python run.py setup --source synthetic
Write-Host ""; Write-Host ">>> run.py capa-risk --source synthetic"
python run.py capa-risk --source synthetic
Write-Host ""; Write-Host ">>> run.py evaluate --source synthetic"
python run.py evaluate --source synthetic
Write-Host ""; Write-Host ">>> run.py experiments --source synthetic --seeds 10"
python run.py experiments --source synthetic --seeds 10
Write-Host ""; Write-Host ">>> run.py experiments --source synthetic --ablation --seeds 10"
python run.py experiments --source synthetic --ablation --seeds 10
Write-Host ""; Write-Host ">>> run.py setup --source maude"
python run.py setup --source maude
Write-Host ""; Write-Host ">>> run.py capa-risk --source maude --by action_class"
python run.py capa-risk --source maude --by action_class
Write-Host ""; Write-Host ">>> run.py evaluate --source maude"
python run.py evaluate --source maude
Write-Host ""; Write-Host ">>> run.py experiments --source maude --n-boot 200"
python run.py experiments --source maude --n-boot 200
Write-Host ""; Write-Host ">>> run.py experiments --source maude --adjusted --n-boot 400"
python run.py experiments --source maude --adjusted --n-boot 400
Write-Host ""; Write-Host ">>> run.py experiments --source maude --ablation"
python run.py experiments --source maude --ablation
Write-Host ""; Write-Host "Done. Reported numbers are in results/."
