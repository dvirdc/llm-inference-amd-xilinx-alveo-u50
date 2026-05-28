# models/

This directory holds the runtime artifacts the host needs at inference
time: a checkpoint and a tokenizer. They are NOT vendored into git --
they're multi-MB binaries and the upstream URLs are stable.

## What goes here

- `stories15M.bin`  -- 15M-parameter Llama2-architecture model trained
  by Karpathy on the TinyStories dataset. Good first target: small
  enough to load in seconds, big enough to produce coherent stories.
- `tokenizer.bin`   -- the matching SentencePiece tokenizer (32k vocab).
- (optional) other Karpathy checkpoints: `stories42M.bin`,
  `stories110M.bin`, your own fine-tunes.

## How to fetch

Step 1 -- clone llama2.c (it has the URLs in its README and useful
export scripts):

```bash
bash scripts/fetch_karpathy_llama2c.sh
```

Step 2 -- download stories15M and the tokenizer. As of writing the
URLs are:

```bash
cd models
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```

The `tokenizer.bin` shipped with llama2.c works against *all* of the
tinyllamas checkpoints (they share the same SentencePiece vocab).

## Verifying

```bash
make host
./build/inspect_model --checkpoint models/stories15M.bin
```

You should see:

```
Model config:
  dim         : 288
  hidden_dim  : 768
  n_layers    : 6
  n_heads     : 6
  n_kv_heads  : 6
  vocab_size  : 32000
  seq_len     : 256
  shared_cls  : yes
```

If you see the warning "checkpoint has N trailing bytes after the modern
v0 layout", you may have downloaded the legacy format (which includes
precomputed RoPE tables) -- re-download the modern variant.
