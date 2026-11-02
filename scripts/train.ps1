Push-Location (Join-Path $PSScriptRoot "..")
try {
    $pythonExe = if ($env:PYTHON) { $env:PYTHON } else { "python" }
    & $pythonExe -m src.training.train @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
