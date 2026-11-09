# Colab GPU runs (CLI only)

All commands run from the repo root on a laptop. The CLI login is a one-time browser OAuth consent on the
first `colab` call; `colab whoami` shows the account.

```sh
# 1. Package the working tree (tracked + new, untracked-but-not-ignored files; no PNGs)
git ls-files -co --exclude-standard | grep -v '\.png$' | COPYFILE_DISABLE=1 tar -czf /tmp/pagp.tar.gz -T -

# 2. Provision a free T4 and upload
colab new --gpu T4 -s pagp
colab upload -s pagp /tmp/pagp.tar.gz /content/pagp.tar.gz

# 3. Build + ctest (CUDA only), then the benchmark sweep
colab exec -s pagp --env PAGP_STEPS=build -f colab/run_gpu.py
colab exec -s pagp --timeout 2900 --env PAGP_STEPS=sweep -f colab/run_gpu.py

# 4. Download results right away (sessions can be preempted), save the log, release the VM
colab ls -s pagp /content/pagp/bench/results
colab download -s pagp /content/pagp/bench/results/<name>.csv bench/results/<name>.csv
colab download -s pagp /content/pagp/bench/results/<name>.cmd bench/results/<name>.cmd
colab log -s pagp -o colab/logs/<date>.md
colab stop -s pagp
```

The tar warnings about `LIBARCHIVE.xattr.com.apple.provenance` on the VM are harmless (macOS metadata).

`bench/run_gpu.sh` CSV convention: `median_ms`/`min_ms` are kernel time only; `h2d_ms`/`d2h_ms` are the
median transfer times; `workers` is the thread-block shape. End-to-end time = h2d + kernel + d2h.
