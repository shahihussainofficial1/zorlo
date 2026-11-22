Zorlo v0.1 Beta for Windows (x64)

Runs Qwen3-30B-A3B on a normal PC: the model's experts stay on the SSD and are read as needed,
so it works with little RAM. Output is identical to llama.cpp (greedy, AVX2 float order).

v0.1 supports one model: Qwen3-30B-A3B, Q4_K_M GGUF. The requirements and speeds below are for this
model only, measured on a reference PC with 8 GB of memory and an NVMe SSD. Other models will be added in
later versions with their own requirements.

EASIEST WAY: double-click "Zorlo Desktop".
  Zorlo opens in its own window. The first time, click the model to download (Qwen3 30B is about
  18.6 GB; if it stops, open Zorlo Desktop again and it continues where it left off). Then it loads
  the model and shows the chat. Next time it starts straight away.
  When you close the window Zorlo asks whether to keep running in the background (the model stays
  loaded, the Zorlo icon sits next to the clock) or to stop and free the memory. Right-click that icon
  for: Open Zorlo, Open in browser, Switch model, Start with Windows, and Stop Zorlo and quit.
  A model is downloaded only once: Zorlo keeps it in the "models" folder here, and a copy you
  downloaded earlier into another folder next to Zorlo is moved there instead of downloaded again.
  Models you downloaded are kept up to date automatically, and new models appear in the list when
  they are released.

  No app window? Double-click "Start Zorlo" instead: the same thing in a black window, and the chat
  opens in your browser. Keep that window open while you chat; close it to stop Zorlo.

  Zorlo is a chat that can also do work for you: pick a project folder (top of the page)
  and ask it to build a website, write and run a script, or look something up. It asks before writing
  files or running commands; Settings (bottom left) sets what it may do on its own, plus web search,
  connectors (MCP) and skills. The plain chat page is at http://127.0.0.1:8080/simple
  Attach PDF, Word, Excel and PowerPoint files, pictures and code to a message: Zorlo reads their text
  (the words in pictures and scans with Windows text recognition; the model cannot see the pictures).
  Settings > Model lets the agent use a cloud model instead (OpenAI, Anthropic, Gemini, DeepSeek,
  OpenRouter and others) with your own API key; conversations then go to that provider.
  Already have the model on this PC? Zorlo Desktop finds it on your drives, or choose its folder with
  "Already have this model?" on the first screen.
  If Windows says "Windows protected your PC", click "More info", then "Run anyway".

From a terminal (the first time, open PowerShell in this folder: click the address bar of this
folder in File Explorer, type powershell and press Enter):

  1. .\zorlo start      (the same as Zorlo Desktop, as a menu; it opens the chat in your browser)
     After the first start, Zorlo adds this folder to your PATH, so in any NEW PowerShell window,
     from any folder, you can simply type:  zorlo start
     (zorlo path off undoes that.) If the browser does not open: start http://127.0.0.1:8080
  2. To stop: press Ctrl+C in that window (or close it). This also frees the model's memory.
     Closing only the browser leaves Zorlo running. To stop every copy: Stop-Process -Name zorlo
  3. "'.\zorlo' is not recognized": PowerShell is in another folder. Use zorlo start (without .\)
     after the first start, or open PowerShell in this folder as described above.
  Zorlo Desktop also adds itself to the Start menu: press the Windows key and type Zorlo.

  Download by hand into the same folder Start Zorlo uses:
     .\zorlo pull Haidershahi1/qwen3-30b-a3b-zorlo models\qwen3-30b-a3b

Add --lowbit to the pull command for the optional faster low-bit copy (+9.5 GB, used by --mode fast).
If the download stops, run the same command again: it continues where it stopped.

Or make the pack yourself from the GGUF:
  1. Download Qwen3-30B-A3B-Q4_K_M.gguf (Hugging Face, ~18.6 GB) next to zorlo.exe
  2. .\zorlo pack Qwen3-30B-A3B-Q4_K_M.gguf models\qwen3-30b-a3b
This runs once and needs about 19 GB more disk; delete the .gguf afterwards if you like.

The same server is an OpenAI-compatible API at http://127.0.0.1:8080/v1 (any API key) and an
Anthropic Messages API at /v1/messages. `.\zorlo run models\qwen3-30b-a3b -p "Hello"` answers once in the terminal.

Needs:
  - Windows 10 or 11, 64-bit. Nothing to install: the Microsoft C++ runtime is included, and the app
    window uses Microsoft Edge WebView2, which comes with Windows.
  - A CPU with AVX2 (Intel Haswell 2013 or newer, AMD Zen or newer). CPUs with AVX-512 VNNI use a
    faster prefill kernel automatically.
  - The pack on an SSD. Speed is set mostly by the SSD: NVMe gives ~2-5 tokens/s, SATA much less,
    a hard disk is too slow.
  - 8 GB RAM recommended (works from 4 GB, slower). The cache sizes itself from free RAM.

If zorlo says "no ggml CPU backend found", the CPU has no AVX2 and is not supported.

Files: Zorlo Desktop.exe (the app window; it runs zorlo.exe), zorlo.exe, the llama/ggml DLLs (one ggml-cpu DLL per CPU type, picked at start), and the
Microsoft C++ runtime DLLs. Licenses of included third-party code: THIRD_PARTY_LICENSES.txt.

License: Zorlo is under the Apache License 2.0 (see LICENSE.txt and NOTICE.txt). Free to use, change
and share for any purpose, including commercial use, as long as the copyright and NOTICE are kept.

Qwen3 is made by the Qwen team (Alibaba Cloud) and licensed under Apache 2.0, not under Zorlo's license.
Zorlo changes how its weights are laid out on disk, and for the optional fast mode
makes a smaller 2-bit copy of its experts.
