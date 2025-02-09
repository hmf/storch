# Frequently Asked Questions

## Q: I want to run operations on the GPU, but Storch seems to hang?

Depending on your hardware, the CUDA version and capability settings, CUDA might need to do
[just-in-time compilation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation)
of your kernels, which can take a few minutes. The result is cached, so it should load faster on subsequent runs.

If you're unsure, you can watch the size of the cache:

```bash
watch -d du -sm ~/.nv/ComputeCache
```
If it's still growing, it's very likely that CUDA is doing just-in-time compilation.

You can also increase the cache size to up to 4GB, to avoid recomputation:

```bash
export CUDA_CACHE_MAXSIZE=4294967296
```


## Q: What about GPU support on my Mac?

Recent PyTorch versions provide a new backend based on Apple’s Metal Performance Shaders (MPS).
The MPS backend enables GPU-accelerated training on the M1/M2 architecture.
Right now, there's no ARM build of PyTorch in JavaCPP and MPS ist not enabled.
If you have an M1/M2 machine and want to help, check the umbrella [issue for macosx-aarch64 support](https://github.com/bytedeco/javacpp-presets/issues/1069).