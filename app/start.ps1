# set the parent of the script as the current location.
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "Loading azd .env file from current environment"
Write-Host ""

foreach ($line in (& azd env get-values)) {
    if ($line -match "([^=]+)=(.*)") {
        $key = $matches[1]
        $value = $matches[2] -replace '^"|"$'
        Set-Item -Path "env:\$key" -Value $value
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to load environment variables from azd environment"
    exit $LASTEXITCODE
}


Write-Host 'Creating python virtual environment ".venv"'
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
  # fallback to python3 if python not found
  $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}
Start-Process -FilePath ($pythonCmd).Source -ArgumentList "-m venv .venv" -Wait -NoNewWindow

Write-Host ""
Write-Host "Restoring python packages"
Write-Host ""

$directory = Get-Location
$venvPythonPath = "$directory/.venv/scripts/python.exe"
if (Test-Path -Path "/usr") {
  # fallback to Linux venv path
  $venvPythonPath = "$directory/.venv/bin/python"
}

Start-Process -FilePath $venvPythonPath -ArgumentList "-m pip install -r requirements.txt" -Wait -NoNewWindow
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to restore backend python packages"
    exit $LASTEXITCODE
}


Write-Host ""
Write-Host "Starting app"
Write-Host ""
Set-Location ../app

$port = 8000
$hostname = "localhost"
Start-Process -FilePath $venvPythonPath -ArgumentList "-m streamlit run app.py --server.port $port" -Wait -NoNewWindow

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start app"
    exit $LASTEXITCODE
}
