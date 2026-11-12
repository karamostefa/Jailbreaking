# Starts a local Ollama server (portable install, models on D:) if it is not already running.
$exe = "C:\Users\DELL\tools\ollama\ollama.exe"
$env:OLLAMA_MODELS = "D:\ollama_models"
$env:OLLAMA_HOST = "127.0.0.1:11434"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "3"
New-Item -ItemType Directory -Force $env:OLLAMA_MODELS | Out-Null
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null
    Write-Output "ollama already running"
} catch {
    $log = "D:\jail\experiments\results\ollama_server.log"
    New-Item -ItemType Directory -Force "D:\jail\experiments\results" | Out-Null
    Start-Process -FilePath $exe -ArgumentList "serve" -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    Start-Sleep -Seconds 5
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 10
}
