# Contributing to Zorlo

Thanks for your interest. Zorlo's rule for every change is that it must be **measured**. Speed claims need
numbers, and the default mode must stay identical to llama.cpp.

## Reporting a bug

Open an issue with:
- your CPU, RAM, disk type (NVMe/SATA/HDD) and Windows version
- the exact command, and the model and pack you used
- the console output (run with the same flags again and copy everything)

## Sharing benchmarks

Results from other hardware are very welcome. Please include `python tools/diskbench.py .bench` output and
the tok/s that `zorlo run` prints, with the `--cache-mb` you used.

## Pull requests

1. Keep a change focused: one fix or feature per pull request.
2. Run the tests before you send it:
   ```
   python -m unittest discover -s tests
   zorlo golden <pack> golden\qwen3-30b-a3b.jsonl
   ```
   A change to the engine must keep `zorlo golden` at 20/20 prompts identical in the default mode.
3. A speed change needs before/after numbers in the description, measured on the same machine.
4. Follow the style of the code around your change (C++17, 4-space indent, short comments that say why).

## Adding a model architecture

A new architecture needs its forward pass in `src/engine/moe_model.cpp`, its tensors mapped in
`src/pack/pack.cpp`, and a golden file made with `zorlo-golden` (llama.cpp) for the same model. It is
supported once `zorlo golden` matches llama.cpp exactly.

By contributing, you agree that your contribution is licensed under the Apache License 2.0.
