# third_party/

External code we depend on at *reference* level (not vendored as source).

## llama2.c

```bash
bash scripts/fetch_karpathy_llama2c.sh
```

Clones https://github.com/karpathy/llama2.c into `third_party/llama2.c`.
We use it for:

- the `run.c` reference inference loop (we ported the architecture
  here in `src/llama/llama_forward_cpu.cpp`),
- the `export*.py` scripts if you want to convert a HuggingFace or
  Meta Llama2 model into the same flat checkpoint format,
- the README's download URLs for the stories15M / stories42M /
  stories110M test checkpoints.

The submodule is *.gitignore'd*. Re-run `fetch_karpathy_llama2c.sh` on
each machine that needs it.
