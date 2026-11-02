Push-Location (Join-Path $PSScriptRoot "..")
try {
    $pythonExe = if ($env:PYTHON) { $env:PYTHON } else { "python" }
    & $pythonExe -m ui.model_service @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
