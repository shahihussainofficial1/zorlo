# Downloads Microsoft's WebView2 SDK (headers + static loader) into third_party\webview2 for Zorlo Desktop.
# The runtime itself is part of Windows 10/11; only the SDK is needed to build. Run once before configuring.
param([string]$Version = "1.0.2903.40")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dst = Join-Path $root "third_party\webview2"
if (Test-Path (Join-Path $dst "build\native\include\WebView2.h")) { "WebView2 SDK already in $dst"; return }
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("webview2-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force $tmp | Out-Null
$pkg = Join-Path $tmp "webview2.zip"
Invoke-WebRequest -UseBasicParsing "https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/$Version" -OutFile $pkg
New-Item -ItemType Directory -Force $dst | Out-Null
Expand-Archive $pkg -DestinationPath $dst -Force
Remove-Item -Recurse -Force $tmp
"WebView2 SDK $Version extracted to $dst"
