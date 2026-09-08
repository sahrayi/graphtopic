param(
    [string]$Python = "3.14"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environment = Join-Path $repository ".release-venv"
$pythonExe = Join-Path $environment "Scripts\python.exe"

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Command)
    Write-Host "`n==> $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Set-Location $repository
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Invoke-Checked "Create Python $Python release environment" { py "-$Python" -m venv $environment }
}
Invoke-Checked "Upgrade packaging tools" { & $pythonExe -m pip install --upgrade pip }
Invoke-Checked "Install release candidate and development tools" { & $pythonExe -m pip install -e ".[dev,embedding]" }
Invoke-Checked "Ruff lint" { & $pythonExe -m ruff check . }
Invoke-Checked "Ruff formatting" { & $pythonExe -m ruff format --check . }
Invoke-Checked "Full test suite with branch coverage" {
    & $pythonExe -m pytest --cov=graphtopic --cov-branch --cov-report=term-missing
}
Invoke-Checked "Offline precomputed example" { & $pythonExe examples\precomputed.py }
Invoke-Checked "Offline custom-components example" { & $pythonExe examples\custom_components.py }
Invoke-Checked "Build wheel and source distribution" { & $pythonExe -m build }
Invoke-Checked "Validate distribution metadata" { & $pythonExe -m twine check dist\* }
Invoke-Checked "Git whitespace check" { git diff --check }
Write-Host "`nAll local release checks passed."
