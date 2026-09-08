param(
    [string]$Python = "3.14"
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environment = Join-Path $repository ".wheel-smoke-venv"
$pythonExe = Join-Path $environment "Scripts\python.exe"
$wheels = @(Get-ChildItem (Join-Path $repository "dist\graphtopic-*.whl"))
if ($wheels.Count -ne 1) {
    throw "Expected exactly one GraphTopic wheel in dist; found $($wheels.Count)"
}

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Command)
    Write-Host "`n==> $Description"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Invoke-Checked "Create isolated wheel environment" { py "-$Python" -m venv $environment }
}
Invoke-Checked "Install the built wheel" { & $pythonExe -m pip install $wheels[0].FullName }
Invoke-Checked "Verify installed version" {
    & $pythonExe -I -c "import graphtopic; assert graphtopic.__version__ == '0.1.0'"
}
Invoke-Checked "Run installed-wheel precomputed example" {
    & $pythonExe -I (Join-Path $repository "examples\precomputed.py")
}
Invoke-Checked "Run installed-wheel custom-components example" {
    & $pythonExe -I (Join-Path $repository "examples\custom_components.py")
}
Write-Host "`nInstalled-wheel smoke checks passed."
