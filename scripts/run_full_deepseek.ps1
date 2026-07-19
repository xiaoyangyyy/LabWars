param(
    [int]$Seeds = 1,
    [int]$Rounds = 60,
    [int]$Parallel = 1,
    [switch]$ThirtySeedMatrix,
    [switch]$SkipConnectivityCheck
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($ThirtySeedMatrix) {
    $Seeds = 30
}

if (-not $env:DEEPSEEK_API_KEY) {
    $env:DEEPSEEK_API_KEY = Read-Host "Paste DEEPSEEK_API_KEY"
}

$env:LABWARS_LLM_CONFIG = "config/llm.deepseek.yaml"

if (-not $SkipConnectivityCheck) {
    python -c "from src.engine.llm_adapter import get_adapter; llm=get_adapter(); print(llm.complete_json('Return JSON only.', 'Return exactly: {`\"ok`\": true}'))"
}

python -c "from src.experiments.batch import run_batch; [run_batch(e, seeds=$Seeds, parallel=$Parallel, max_rounds=$Rounds) for e in ['A','B','C','D']]"
python -c "from src.experiments.batch import run_batch; run_batch('V', seeds=$Seeds, condition_ids=['V1','V2','V3','V6'], parallel=$Parallel, max_rounds=$Rounds)"
python -c "from src.experiments.aggregate import write_aggregate_report; [print(write_aggregate_report(e)) for e in ['A','B','C','D','V']]"
python -c "from src.experiments.report import generate_report; print(generate_report(experiment_id='A', condition_id='A2', seed=0))"

Write-Host "LabWars DeepSeek-compatible full run complete. Outputs:" -ForegroundColor Green
Write-Host "  output/runs/"
Write-Host "  output/reports/"
