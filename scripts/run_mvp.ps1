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
    $setup_commands = @(
        "Project virtual environment was not found: $python"
        "Create it with a Python 3.11 or 3.12 interpreter:"
        "  python --version"
        "  python -m venv .venv"
        "  .\.venv\Scripts\python.exe -m pip install --upgrade pip"
        "  .\.venv\Scripts\python.exe -m pip install -e '.[dev]'"
    ) -join [Environment]::NewLine
    throw $setup_commands
}

if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "MVP config was not found: $config"
}

function Invoke-ProjectPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$CommandArguments
    )

    & $python @CommandArguments
    $exit_code = $LASTEXITCODE
    if ($exit_code -ne 0) {
        $rendered_command = $CommandArguments -join ' '
        throw "Python command failed with exit code ${exit_code}: python $rendered_command"
    }
}

Push-Location $project_root
try {
    if ($Stage -in @('data', 'all')) {
        Invoke-ProjectPython -CommandArguments @('-m', 'tcmi', 'generate', '--config', $config)
        Invoke-ProjectPython -CommandArguments @('-m', 'tcmi', 'audit', '--config', $config)
    }
    if ($Stage -in @('train', 'all')) {
        Invoke-ProjectPython -CommandArguments @(
            '-m', 'tcmi', 'matrix', '--config', $config, '--execute', '--stage', 'train'
        )
    }
    if ($Stage -in @('probe', 'all')) {
        Invoke-ProjectPython -CommandArguments @(
            '-m', 'tcmi', 'matrix', '--config', $config, '--execute', '--stage', 'probe'
        )
    }
    if ($Stage -in @('aggregate', 'all')) {
        Invoke-ProjectPython -CommandArguments @('-m', 'tcmi', 'aggregate', '--config', $config)
        Invoke-ProjectPython -CommandArguments @('-m', 'tcmi', 'decide', '--config', $config)
        Invoke-ProjectPython -CommandArguments @('-m', 'tcmi', 'status', '--config', $config)
    }
}
finally {
    Pop-Location
}
