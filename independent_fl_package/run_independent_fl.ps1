param(
    [ValidateSet('Prepare','Train','All')]
    [string]$Mode = 'All',
    [ValidateSet('UNSW_NB15','CICIoT2023','EdgeIIoTset','all')]
    [string]$Dataset = 'all',
    [int]$Rounds = 10,
    [int]$LocalSteps = 100,
    [int]$BatchSize = 128,
    [double]$LearningRate = 0.0001,
    [switch]$RebuildData
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { $Python = (Get-Command python -ErrorAction Stop).Source }

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host 'INDEPENDENT DATASET FEDERATED LEARNING' -ForegroundColor Cyan
Write-Host "Project: $ProjectRoot"
Write-Host "Python : $Python"
Write-Host "Mode   : $Mode | Dataset: $Dataset"
Write-Host '============================================================'

if ($Mode -in @('Prepare','All')) {
    $PrepareArgs = @('-u','src/independent_fl/prepare_independent_fl_data.py')
    if ($RebuildData) { $PrepareArgs += '--overwrite' }
    Write-Host "`n[1/2] Preparing separate 20% samples for each dataset..." -ForegroundColor Yellow
    & $Python @PrepareArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed with exit code $LASTEXITCODE" }
}

if ($Mode -in @('Train','All')) {
    $PreparedRoot = Join-Path $ProjectRoot 'data\independent_fl_v1'
    if (-not (Test-Path (Join-Path $PreparedRoot 'global_summary.json'))) {
        throw "Prepared datasets not found. Run: .\run_independent_fl.ps1 -Mode Prepare"
    }
    Write-Host "`n[2/2] Training independent model(s)..." -ForegroundColor Yellow
    & $Python -u 'src/independent_fl/train_independent_fl.py' `
        --dataset $Dataset `
        --rounds $Rounds `
        --local-steps $LocalSteps `
        --batch-size $BatchSize `
        --learning-rate $LearningRate
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
}

Write-Host "`nCompleted requested mode. Check results\independent_fl_v1\ for metrics." -ForegroundColor Green
