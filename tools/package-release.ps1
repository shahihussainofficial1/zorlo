# Builds the Windows release zip from build-dl (GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS, Release).
#   cmake -S . -B build-dl -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON
#   .\tools\get-webview2.ps1   (once, for Zorlo Desktop)
#   cmake --build build-dl --config Release --parallel --target zorlo zorlo-desktop ggml-cpu-haswell ggml-cpu-cascadelake
#   .\tools\package-release.ps1 [-Version 0.1]
param([string]$Version = "0.1")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$bin = Join-Path $root "build-dl\bin\Release"
$name = "zorlo-v$Version-windows-x64"
$out = Join-Path $root "build-dl\dist\$name"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force $out | Out-Null

# AVX2 (haswell) for every supported CPU, AVX-512 VNNI (cascadelake) for faster prefill where present
$files = "Zorlo Desktop.exe", "zorlo.exe", "llama.dll", "llama-common.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu-haswell.dll", "ggml-cpu-cascadelake.dll"
foreach ($f in $files) { Copy-Item -LiteralPath (Join-Path $bin $f) $out }

# Microsoft C++ runtime, deployed next to the exe (app-local), so nothing needs installing
$redist = Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\VC\Redist\MSVC\*\x64" -Directory |
    Sort-Object FullName | Select-Object -Last 1
foreach ($f in "Microsoft.VC143.CRT\msvcp140.dll", "Microsoft.VC143.CRT\vcruntime140.dll",
               "Microsoft.VC143.CRT\vcruntime140_1.dll", "Microsoft.VC143.OpenMP\vcomp140.dll") {
    Copy-Item (Join-Path $redist.FullName $f) $out
}

Copy-Item (Join-Path $root "release\README.txt") $out
Copy-Item (Join-Path $root "LICENSE") (Join-Path $out "LICENSE.txt")     # Zorlo itself: Apache-2.0
Copy-Item (Join-Path $root "NOTICE") (Join-Path $out "NOTICE.txt")
Copy-Item -LiteralPath (Join-Path $root "release\Start Zorlo.bat") $out   # double-click: pull the model, serve, open the chat
Copy-Item -Recurse (Join-Path $root "ui") (Join-Path $out "ui")            # the agent app served at "/"

$lc = Join-Path $root "third_party\llama.cpp"
$parts = @(
    @("llama.cpp and ggml (MIT)", "$lc\LICENSE"),
    @("cpp-httplib (MIT)", "$lc\vendor\cpp-httplib\LICENSE"),
    @("nlohmann/json (MIT)", "$lc\licenses\LICENSE-jsonhpp"),
    @("Microsoft Edge WebView2 SDK loader, linked into Zorlo Desktop.exe (BSD-3-Clause)", "$root\third_party\webview2\LICENSE.txt")
)
$text = "Third-party code included in Zorlo v$Version`r`n`r`n"
foreach ($p in $parts) {
    $text += ("=" * 72) + "`r`n" + $p[0] + "`r`n" + ("=" * 72) + "`r`n" + (Get-Content -Raw $p[1]) + "`r`n"
}
$text += ("=" * 72) + "`r`nsheredom/subprocess.h (Unlicense, public domain)`r`n" + ("=" * 72) + "`r`n" +
    "https://github.com/sheredom/subprocess.h - released into the public domain (http://unlicense.org/).`r`n`r`n"
$text += ("=" * 72) + "`r`ncommon/base64.hpp in llama-common.dll (Unlicense, public domain)`r`n" + ("=" * 72) + "`r`n" +
    "Released into the public domain (http://unlicense.org/).`r`n`r`n"
# Parts of ggml-cpu that carry their own copyright line inside llama.cpp's MIT-licensed tree
$mit = Get-Content -Raw "$lc\LICENSE"
$mit = $mit.Substring($mit.IndexOf("Permission is hereby granted"))
$text += ("=" * 72) + "`r`nllamafile sgemm, in the ggml-cpu DLLs (MIT)`r`n" + ("=" * 72) + "`r`nCopyright 2024 Mozilla Foundation`r`n`r`n" + $mit + "`r`n"
$text += ("=" * 72) + "`r`nYaRN rotary embedding, in the ggml-cpu DLLs (MIT)`r`n" + ("=" * 72) + "`r`n" +
    "Based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn`r`nCopyright (c) 2023 Jeffrey Quesnelle and Bowen Peng`r`n`r`n" + $mit + "`r`n"
$text += ("=" * 72) + "`r`nMicrosoft Visual C++ runtime`r`n" + ("=" * 72) + "`r`n" +
    "msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll and vcomp140.dll are Microsoft Visual C++`r`n" +
    "Redistributable files, distributed under the Visual Studio license terms for redistributable code.`r`n"
Set-Content -Path (Join-Path $out "THIRD_PARTY_LICENSES.txt") -Value $text -Encoding utf8

$zip = "$out.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $out -DestinationPath $zip
Get-Item $zip | Select-Object FullName, Length
