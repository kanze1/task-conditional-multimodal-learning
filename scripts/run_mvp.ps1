[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('smoke', 'pilot')]
    [string]$Profile = 'smoke',

    [Parameter(Mandatory = $false)]
    [ValidateSet('data', 'train', 'probe', 'aggregate', 'all')]
    [string]$Stage = 'all'
)

$ErrorActionPreference = 'Stop'
$project_root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project_root '.venv\Scripts\python.exe'
$config = Join-Path $project_root "configs\mvp_$Profile.yaml"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "未找到项目 Python 环境：$python"
}
Push-Location $project_root
try {
    if ($Stage -in @('data', 'all')) {
        & $python -m tcmi generate --config $config
        & $python -m tcmi audit --config $config
    }
    if ($Stage -in @('train', 'all')) {
        & $python -m tcmi matrix --config $config --execute --stage train
    }
    if ($Stage -in @('probe', 'all')) {
        & $python -m tcmi matrix --config $config --execute --stage probe
    }
    if ($Stage -in @('aggregate', 'all')) {
        & $python -m tcmi aggregate --config $config
        & $python -m tcmi decide --config $config
        & $python -m tcmi status --config $config
    }
}
finally {
    Pop-Location
}
