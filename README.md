# Storch - GPU Accelerated Deep Learning for Scala 3

Storch is a Scala library for fast tensor computations and deep learning, based on PyTorch.

Like PyTorch, Storch provides
* A NumPy like API for working with tensors
* GPU support
* Automatic differentiation
* A neural network API for building and training neural networks.

Storch aims to close to the Python API to make porting existing models and the life of people already familiar with PyTorch easier.

For documentation, see https://storch.dev

## Example:

```scala
val data = Seq(0,1,2,3)
// data: Seq[Int] = List(0, 1, 2, 3)
val t1 = torch.Tensor(data)
// t1: Tensor[Int32] = dtype=int32, shape=[4], device=CPU 
// [0, 1, 2, 3]
t1.equal(torch.arange(0,4))
// res0: Boolean = true
val t2 = t1.to(dtype=float32)
// t2: Tensor[Float32] = dtype=float32, shape=[4], device=CPU 
// [0,0000, 1,0000, 2,0000, 3,0000]
val t3 = t1 + t2
// t3: Tensor[Float32] = dtype=float32, shape=[4], device=CPU 
// [0,0000, 2,0000, 4,0000, 6,0000]

val shape = Seq(2l,3l)
// shape: Seq[Long] = List(2, 3)
val randTensor = torch.rand(shape)
// randTensor: Tensor[Float32] = dtype=float32, shape=[2, 3], device=CPU 
// [[0,4341, 0,9738, 0,9305],
//  [0,8987, 0,1122, 0,3912]]
val zerosTensor = torch.zeros(shape, dtype=torch.int64)
// zerosTensor: Tensor[Int64] = dtype=int64, shape=[2, 3], device=CPU 
// [[0, 0, 0],
//  [0, 0, 0]]

val x = torch.ones(Seq(5))
// x: Tensor[Float32] = dtype=float32, shape=[5], device=CPU 
// [1,0000, 1,0000, 1,0000, 1,0000, 1,0000]
val w = torch.randn(Seq(5, 3), requiresGrad=true)
// w: Tensor[Float32] = dtype=float32, shape=[5, 3], device=CPU 
// [[0,8975, 0,5484, 0,2307],
//  [0,2689, 0,7430, 0,6446],
//  [0,9503, 0,6342, 0,7523],
//  [0,5332, 0,7497, 0,3665],
//  [0,3376, 0,6040, 0,5033]]
val b = torch.randn(Seq(3), requiresGrad=true)
// b: Tensor[Float32] = dtype=float32, shape=[3], device=CPU 
// [0,2638, 0,9697, 0,3664]
val z = (x matmul w) + b
// z: Tensor[Float32] = dtype=float32, shape=[3], device=CPU 
// [3,2513, 4,2490, 2,8640]
```



vscode ➜ /workspaces/storch (explore_1) $ ./mill examples.runMain LeNetApp2
[109/116] examples.compile 
[info] compiling 1 Scala source to /workspaces/storch/out/examples/compile.dest/classes ...
[info] done compiling
[116/116] examples.runMain 
[W interface.cpp:47] Warning: Loading nvfuser library failed with: Error in dlopen: libtorch.so: cannot open shared object file: No such file or directory (function LoadingNvfuserLibrary)
Using device: Device(CUDA,-1)
Exception in thread "main" java.lang.ExceptionInInitializerError
        at LeNetApp2.main(LeNet2.scala)
Caused by: java.lang.RuntimeException: CUDA error: the provided PTX was compiled with an unsupported toolchain.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Exception raised from c10_cuda_check_implementation at /home/runner/work/javacpp-presets/javacpp-presets/pytorch/cppbuild/linux-x86_64-gpu/pytorch/c10/cuda/CUDAException.cpp:44 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x6c (0x7fc0e40b1d8c in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xfa (0x7fc0e4077f6a in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10.so)
frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x3cc (0x7fc0e468e34c in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10_cuda.so)
frame #3: void at::native::gpu_kernel_impl<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&) + 0xb45 (0x7fbf5d7f5b85 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #4: void at::native::gpu_kernel<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&) + 0x35b (0x7fbf5d7f64fb in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #5: <unknown function> + 0x3005120 (0x7fbf5d7bf120 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #6: at::native::add_kernel(at::TensorIteratorBase&, c10::Scalar const&) + 0x34 (0x7fbf5d7c03c4 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #7: <unknown function> + 0x2e44703 (0x7fbf5d5fe703 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #8: at::_ops::add__Tensor::call(at::Tensor&, at::Tensor const&, c10::Scalar const&) + 0x15b (0x7fbf7d74164b in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #9: at::native::_convolution(at::Tensor const&, at::Tensor const&, c10::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long, bool, bool, bool, bool) + 0x1bf2 (0x7fbf7cb18982 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #10: <unknown function> + 0x28741ea (0x7fbf7dce51ea in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #11: <unknown function> + 0x28742b9 (0x7fbf7dce52b9 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #12: at::_ops::_convolution::call(at::Tensor const&, at::Tensor const&, c10::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<long>, bool, c10::ArrayRef<c10::SymInt>, long, bool, bool, bool, bool) + 0x29b (0x7fbf7d410f3b in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #13: at::native::convolution(at::Tensor const&, at::Tensor const&, c10::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, bool, c10::ArrayRef<long>, long) + 0x15a (0x7fbf7cb0b93a in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #14: <unknown function> + 0x2873a50 (0x7fbf7dce4a50 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #15: <unknown function> + 0x2873ad7 (0x7fbf7dce4ad7 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #16: at::_ops::convolution::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&, c10::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<long>, bool, c10::ArrayRef<c10::SymInt>, long) + 0x14a (0x7fbf7d3d229a in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #17: <unknown function> + 0x3cf52b2 (0x7fbf7f1662b2 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #18: <unknown function> + 0x3cf624b (0x7fbf7f16724b in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #19: at::_ops::convolution::call(at::Tensor const&, at::Tensor const&, c10::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<long>, bool, c10::ArrayRef<c10::SymInt>, long) + 0x240 (0x7fbf7d410200 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #20: at::native::conv2d(at::Tensor const&, at::Tensor const&, c10::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long) + 0x251 (0x7fbf7cb0f221 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #21: <unknown function> + 0x2a6bbc5 (0x7fbf7dedcbc5 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #22: at::_ops::conv2d::call(at::Tensor const&, at::Tensor const&, c10::optional<at::Tensor> const&, c10::ArrayRef<long>, c10::ArrayRef<long>, c10::ArrayRef<long>, long) + 0x21f (0x7fbf7da146af in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #23: torch::nn::Conv2dImpl::_conv_forward(at::Tensor const&, at::Tensor const&) + 0x47b (0x7fbf808a0b2b in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #24: torch::nn::Conv2dImpl::forward(at::Tensor const&) + 0x29 (0x7fbf808a0fc9 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #25: Java_org_bytedeco_pytorch_Conv2dImpl_forward + 0xcc (0x7fbf59545bbc in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libjnitorch.so)
frame #26: [0x7fc2349a053a]

        at org.bytedeco.pytorch.Conv2dImpl.forward(Native Method)
        at torch.nn.modules.conv.Conv2d.apply(Conv2d.scala:62)
        at LeNet2.apply(LeNet2.scala:50)
        at LeNetApp2$.$init$$$anonfun$1$$anonfun$1$$anonfun$1(LeNet2.scala:101)
        at scala.runtime.function.JProcedure1.apply(JProcedure1.java:15)
        at scala.runtime.function.JProcedure1.apply(JProcedure1.java:10)
        at scala.util.Using$.resource(Using.scala:261)
        at LeNetApp2$.$init$$$anonfun$1$$anonfun$1(LeNet2.scala:114)
        at scala.runtime.function.JProcedure1.apply(JProcedure1.java:15)
        at scala.runtime.function.JProcedure1.apply(JProcedure1.java:10)
        at scala.collection.IterableOnceOps.foreach(IterableOnce.scala:575)
        at scala.collection.IterableOnceOps.foreach$(IterableOnce.scala:573)
        at scala.collection.AbstractIterator.foreach(Iterator.scala:1300)
        at LeNetApp2$.$init$$$anonfun$1(LeNet2.scala:114)
        at scala.runtime.java8.JFunction1$mcVI$sp.apply(JFunction1$mcVI$sp.scala:18)
        at scala.collection.immutable.Range.foreach(Range.scala:190)
        at LeNetApp2$.<clinit>(LeNet2.scala:117)
        ... 1 more
1 targets failed
examples.runMain subprocess failed
vscode ➜ /workspaces/storch (explore_1) $ 



vscode ➜ /workspaces/storch (explore_1) $ sudo apt-get update
vscode ➜ /workspaces/storch (explore_1) $ sudo apt install locate
vscode ➜ /workspaces/storch (explore_1) $ sudo updatedb 
vscode ➜ /workspaces/storch (explore_1) $ locate -i libtorch.so
/home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20231025.021128-64-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch.so


https://docs.nvidia.com/deploy/cuda-compatibility/

CUDA Toolkit
CUDA 12.x 	>=525.60.13

https://docs.nvidia.com/deploy/cuda-compatibility/#use-the-right-compat-package

CUDA Forward Compatible Upgrade
12-3 	Ok with 535.54.03+ (CUDA 12.2)
12-3  not Ok 545.23.06+ (CUDA 12.3) 

Forward Compatibility is applicable only for systems with NVIDIA Data Center GPUs or select NGC Server Ready SKUs of RTX cards. It’s mainly intended to support applications built on newer CUDA Toolkits to run on systems installed with an older NVIDIA Linux GPU driver from different major release families. This new forward-compatible upgrade path requires the use of a special package called “CUDA compat package”. 

https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
cuDNN 8.9.5 for CUDA 12.x 	12.2, >=525.60.13

https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
