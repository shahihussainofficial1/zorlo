# Start the Zorlo chat server and open the chat page in the browser.
#   .\tools\zorlo-chat.ps1 [-Pack models\qwen3-30b-a3b.zorlo] [-Port 8080] [extra zorlo serve options, e.g. --mode fast]
param([string]$Pack = "models\qwen3-30b-a3b.zorlo", [int]$Port = 8080)
$zorlo = Join-Path $PSScriptRoot "..\build\bin\Release\zorlo.exe"
Start-Job { param($p) Start-Sleep 8; Start-Process "http://127.0.0.1:$p/" } -ArgumentList $Port | Out-Null
& $zorlo serve $Pack --port $Port @args
