package gpt

/*
import torch
import torch.nn as nn
from torch.nn import functional as F
*/

// cSpell: ignore gpt, hyperparameters, logits, softmax
// cSpell: ignore CUDA, torchvision
// cSpell: ignore dtype
// cSpell: ignore stoi, itos
// cSpell: ignore nn, probs, numel, itemsize, nbytes
// cSpell: ignore xbow, xprev, isinstance, idx, tok_emb

import java.nio.file.Paths
import java.nio.file.Files
import java.net.URL
import java.net.URI

import scala.annotation.targetName
import scala.util.Random
import scala.util.Using
import scala.collection.immutable.SortedSet

import org.bytedeco.pytorch.OutputArchive
import org.bytedeco.javacpp.PointerScope

import torch.*
import torch.Device.CUDA
import torch.Device.CPU
import torch.nn.functional as F
import torch.nn.modules.Module
import torch.nn.modules.HasParams
import torch.{---, Slice}
import torch.optim.Adam
import torch.DType.float32
import org.bytedeco.javacpp.Pointer
import torch.nn.modules.HasWeight
import org.bytedeco.pytorch.cuda.Stat
import org.bytedeco.pytorch.cuda.CheckpointDelta
import org.bytedeco.pytorch.cuda.SnapshotInfo
import org.bytedeco.pytorch.cuda.CUDAAllocator
import org.bytedeco.pytorch.cuda.SegmentInfo
import org.bytedeco.pytorch.cuda.BlockInfo
import org.bytedeco.pytorch.cuda.DeviceStats
import org.bytedeco.javacpp.BoolPointer
import org.bytedeco.pytorch.global.torch_cuda



/**
  * ./mill -i examples.runMain gpt.V2
  * nohup ./mill -i examples.runMain gpt.V2 > v2_0.txt 2>&1 &
  * 
  * @see https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
  */
object V2:


  // TODO: memory profiling
  // https://docs.scala-lang.org/scala3/reference/experimental/cc.html
  // https://stackoverflow.com/questions/43944949/is-it-possible-to-reference-count-call-location
  // https://www.toptal.com/software/eliminating-garbage-collector
  // https://verdagon.dev/blog/single-ownership-without-borrow-checking-rc-gc
  // https://www.eddywm.com/modern-approaches-to-automatic-memory-management/
  // https://news.ycombinator.com/item?id=31139610
  // https://stackoverflow.com/questions/12429091/smart-pointers-and-ref-counting-in-java
  // https://sites.cs.ucsb.edu/~ckrintz/racelab/gc/papers/levanoni-on-the-fly-rc.pdf
  // https://www.baeldung.com/java-destructor

  // https://saturncloud.io/blog/how-to-monitor-gpu-memory-usage-in-pytorch/
  // Retrieve maximum GPU memory allocated by PyTorch
  // max_memory_allocated = torch.cuda.max_memory_allocated()
  //
  // https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
  // # Retrieve GPU memory statistics
  // memory_stats = torch.cuda.memory_stats()
  // 
  // # Calculate available GPU memory
  // total_memory = torch.cuda.get_device_properties(0).total_memory
  // available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
  // 
  // # Print the result
  // print(f"Available GPU memory: {available_memory / 1024**3:.2f} GB")    

  /**
    * 
    *
    * @param availablePhysicalBytes
    * @param totalPhysicalBytes
    * @param maxPhysicalBytes
    * @param maxBytes
    * @param physicalBytes
    * @param totalBytes
    * @param totalCount
    * 
    * @see org.bytedeco.javacpp.Pointer
    */
  case class PointerInfo(
    availablePhysicalBytes: BigInt,
    totalPhysicalBytes: BigInt,
    maxPhysicalBytes: BigInt,
    maxBytes: BigInt,
    physicalBytes: BigInt,
    totalBytes: BigInt,
    totalCount: BigInt,
  ):
    def -(o: PointerInfo): PointerInfo =
      PointerInfo(
        availablePhysicalBytes - o.availablePhysicalBytes,
        totalPhysicalBytes     - o.totalPhysicalBytes,
        maxPhysicalBytes       - o.maxPhysicalBytes,
        maxBytes               - o.maxBytes,
        physicalBytes          - o.physicalBytes,
        totalBytes             - o.totalBytes,
        totalCount             - o.totalCount,
      )
  
    def +(o: PointerInfo): PointerInfo =
      PointerInfo(
        availablePhysicalBytes + o.availablePhysicalBytes,
        totalPhysicalBytes     + o.totalPhysicalBytes,
        maxPhysicalBytes       + o.maxPhysicalBytes,
        maxBytes               + o.maxBytes,
        physicalBytes          + o.physicalBytes,
        totalBytes             + o.totalBytes,
        totalCount             + o.totalCount,
      )

  object PointerInfo:
    def apply(): PointerInfo =
      val z = BigInt(0)
      PointerInfo(z, z, z, z, z, z, z)

  // nvidia-smi -l 1
  // No parity between pointer and nvidia-smi values
  // Do no seem to worK
  // x.native.referenceCount()  // 1 
  // y.native.referenceCount()  // 1
  // x.native.deallocate()
  // y.native.deallocate()

  def getMemoryInfo() =
    PointerInfo(
      Pointer.availablePhysicalBytes(),
      Pointer.totalPhysicalBytes(),
      Pointer.maxPhysicalBytes(),
      Pointer.maxBytes(),
      Pointer.physicalBytes(),
      Pointer.totalBytes(),
      Pointer.totalCount()
    )

  def printAllMemoryInfo(info: PointerInfo, inUseOnly : Boolean = false) =
    if !inUseOnly
    then
      println(s"PointerInfo.availablePhysicalBytes ${humanReadableSize(info.availablePhysicalBytes)}")
      println(s"PointerInfo.totalPhysicalBytes ${humanReadableSize(info.totalPhysicalBytes)}")
      println(s"PointerInfo.maxPhysicalBytes ${humanReadableSize(info.maxPhysicalBytes)}")
      println(s"PointerInfo.maxBytes ${humanReadableSize(info.maxBytes)}")
    println(s"PointerInfo.physicalBytes ${humanReadableSize(info.physicalBytes)}")
    println(s"PointerInfo.totalBytes ${humanReadableSize(info.totalBytes)}")
    println(s"PointerInfo.totalCount ${humanReadableSize(info.totalCount)}")
    
  def printMemoryInfo(info: PointerInfo) =
    printAllMemoryInfo(info, inUseOnly = true)

  
  def statToDict(stat: Stat, name: String, dict: scala.collection.mutable.Map[String, Long]) =
    dict(s"$name.current")   = stat.current()
    dict(s"$name.peak")      = stat.peak()
    dict(s"$name.allocated") = stat.allocated()
    dict(s"$name.freed")     = stat.freed()

  def statArrayToDict(
          statArray: Stat, 
          name: String, 
          dict: scala.collection.mutable.Map[String, Long]) =

    val statTypeNames = Array("all", "small_pool", "large_pool")
    for i <- 0 until statTypeNames.length
    do
      statToDict(statArray.position(i), s"$name.${statTypeNames(i)}", dict)


  /**
    * Equivalent to PyTorch memory_stats. 
    * Returns a dictionary of CUDA memory allocator statistics for a given device.
    * The return value of this function is a dictionary of statistics, each of which is a non-negative integer.
    * 
    * Reference implementation
    * @see https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
    * @see https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/Module.cpp#L1243
    * @see https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py#L165
    * 
    * JavaCPP references
    * @see https://github.com/bytedeco/javacpp-presets/blob/master/pytorch/src/gen/java/org/bytedeco/pytorch/cuda/DeviceStats.java#L26
    * @see https://github.com/bytedeco/javacpp-presets/blob/master/pytorch/src/gen/java/org/bytedeco/pytorch/global/torch_cuda.java#L766
    * @see https://github.com/bytedeco/javacpp-presets/issues/1422
    * 
    * @param device
    * @return
    */
  def memoryStats(device: Int): scala.collection.mutable.Map[String, Long] =
    // get the device
    val cudaAllocator = torch_cuda.getAllocator()
    println( cudaAllocator.initialized() )
    val stats = cudaAllocator.getDeviceStats(device)

    // Collect the statistics
    val result = scala.collection.mutable.Map[String, Long]()
    result("num_alloc_retries") = stats.num_alloc_retries
    result("num_ooms") = stats.num_ooms
    result("max_split_size") = stats.max_split_size

    // Stat(stats.allocation) casts a Pointer to a Stats pointer. Can be an array 
    statArrayToDict(Stat(stats.allocation),           "allocation",           result)
    statArrayToDict(Stat(stats.segment),              "segment",              result)
    statArrayToDict(Stat(stats.active),               "active",               result)
    statArrayToDict(Stat(stats.inactive_split),       "inactive_split",       result)
    statArrayToDict(Stat(stats.allocated_bytes),      "allocated_bytes",      result)
    statArrayToDict(Stat(stats.reserved_bytes),       "reserved_bytes",       result)
    statArrayToDict(Stat(stats.active_bytes),         "active_bytes",         result)
    statArrayToDict(Stat(stats.inactive_split_bytes), "inactive_split_bytes", result)
    statArrayToDict(Stat(stats.requested_bytes),      "requested_bytes",      result)

    statToDict(stats.oversize_allocations, "oversize_allocations", result)
    statToDict(stats.oversize_segments,    "oversize_segments",    result)
    result

  // TODO
  // https://pytorch.org/docs/stable/generated/torch.cuda.memory_summary.html
  // https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py#L469
  // https://discuss.pytorch.org/t/how-to-check-the-gpu-memory-being-used/131220
  def memory_summary(device: Int, abbreviated: Boolean = false) =
    val result = memoryStats(device)
    val l = result.toList.sortBy(_._1)
    l.map((a,b) => s"$a : ${humanReadableSize(b)}")
     .mkString("\n")
    

  // https://discuss.pytorch.org/t/memory-cached-and-memory-allocated-does-not-nvidia-smi-result/28420
  // https://discuss.pytorch.org/t/pytorchs-torch-cuda-max-memory-allocated-showing-different-results-from-nvidia-smi/165706
  // torch.cuda.max_memory_reserved
  // https://github.com/pytorch/pytorch/issues/101159
  // https://github.com/pytorch/pytorch/issues/37250
    
  // memory_stats
  // https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
  def printMemoryInfo(device: Int) =
    println(memory_summary(device))
//     // CUDACachingAllocator
//     // https://github.com/bytedeco/javacpp-presets/issues/1422
//     // https://github.com/bytedeco/javacpp-presets/tree/master/pytorch
//     // https://github.com/bytedeco/javacpp-presets/blob/7629ea48be0a3a368c74fe43ef70577bee091010/pytorch/src/gen/java/org/bytedeco/pytorch/cuda/DeviceStats.java#L26
// 
//     val cudaAllocator = torch_cuda.getAllocator()
//     println( cudaAllocator.initialized() )
//     val stats = cudaAllocator.getDeviceStats(0)
// 
//     val result = scala.collection.mutable.Map[String, Long]()
//     result("num_alloc_retries") = stats.num_alloc_retries
//     result("num_ooms") = stats.num_ooms
//     result("max_split_size") = stats.max_split_size
// 
//     statArrayToDict(Stat(stats.allocation),           "allocation",           result)
//     statArrayToDict(Stat(stats.segment),              "segment",              result)
//     statArrayToDict(Stat(stats.active),               "active",               result)
//     statArrayToDict(Stat(stats.inactive_split),       "inactive_split",       result)
//     statArrayToDict(Stat(stats.allocated_bytes),      "allocated_bytes",      result)
//     statArrayToDict(Stat(stats.reserved_bytes),       "reserved_bytes",       result)
//     statArrayToDict(Stat(stats.active_bytes),         "active_bytes",         result)
//     statArrayToDict(Stat(stats.inactive_split_bytes), "inactive_split_bytes", result)
//     statArrayToDict(Stat(stats.requested_bytes),      "requested_bytes",      result)
// 
//     statToDict(stats.oversize_allocations, "oversize_allocations", result)
//     statToDict(stats.oversize_segments,    "oversize_segments",    result)
// 
//     // val devAllocation: BoolPointer = stats.allocation()
//     // println(devAllocation.capacity())
//     // val statArray = Stat( devAllocation )
//     // println(statArray.capacity())
//     // println(statArray.sizeof())
//     // val c = statArray.position(4).current()
//     // val z = statArray.position(4)
//     // println(c)
//     println(result.mkString(",\n"))
//    ???

    // TODO: get device properties
    // https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/Module.cpp#L1243
    // https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch

    // TODO: memory_summary 
    // https://discuss.pytorch.org/t/how-to-check-the-gpu-memory-being-used/131220
    // https://pytorch.org/docs/stable/generated/torch.cuda.memory_summary.html

    // https://discuss.pytorch.org/t/how-to-check-the-gpu-memory-being-used/131220/3
    // https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    // https://pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html#torch.cuda.mem_get_info
    // https://discuss.pytorch.org/t/how-to-calculate-the-gpu-memory-that-a-model-uses/157486/4

    // https://discuss.pytorch.org/t/how-can-i-use-the-cudacachingallocator-api/173258
    // https://pytorch.org/docs/stable/notes/cuda.html
    // val s = Stat()
    // val allocated = s.allocated()
    // val current = s.current()
    // val peak = s.peak()
    // val freed = s.freed()
    // val capacity = s.capacity()
    // val limit = s.limit()
    // println(humanReadableSize(allocated))
    // println(humanReadableSize(current))
    // println(humanReadableSize(peak))
    // println(humanReadableSize(freed))
    // println(humanReadableSize(capacity))
    // println(humanReadableSize(limit))

    // TODO: how do we get array of stat?
    // https://github.com/pytorch/pytorch/blob/af5a3bda4518be89239623a5490ca3392cc0fcbd/torch/csrc/cuda/Module.cpp#L587    
    // https://github.com/pytorch/pytorch/blob/main/torch/csrc/cuda/Module.cpp#L587
    // val ds = DeviceStats()
    // val allocation = ds.allocation()
    // val devS = CUDAAllocator(Pointer()).getDeviceStats(1)
    // val devAllocation: BoolPointer = devS.allocation()
    // //val p = devAllocation.getPointer[Stat]()
    // val oversize_allocation = devS.oversize_allocations()

    //val memory_stats = torch.cuda.memory_stats()
    // val cp = CheckpointDelta()
    // val snap = SnapshotInfo()
    // val segment = SegmentInfo()
    // val block = BlockInfo()
    // val cudaS = CUDAAllocator()
    // https://discuss.pytorch.org/t/obtaining-memory-stats-in-libtorch/105119
    // val alloc = CUDAAllocator(Pointer())
    // val stats = alloc.getDeviceStats(1)
    // val gpuAllocated = stats.allocated_bytes()
    // println(humanReadableSize(allocated))
    // https://discuss.pytorch.org/t/libtorch-equivalent-of-torch-cuda-memory-reserved/165995
    // https://github.com/pytorch/pytorch/issues/85436
    // https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    // https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
    // https://github.com/pytorch/pytorch/blob/af5a3bda4518be89239623a5490ca3392cc0fcbd/torch/csrc/cuda/Module.cpp#L1435
    ()
    


  
  // Utility functions


  def len[T <: torch.DType](t: Tensor[T]): Int = 
    // t.size.sum
    t.shape.sum

  def register_i[M1 <: Module, M2 <: Module](parent: M1, child: M2, i: Int, n: String = "")(using name: sourcecode.Name): M2 =
    val name_ = if n.trim().isEmpty() then name.value else n.trim()
    val name_i = s"${name_}_$i"
    parent.register(child, name_i)

  def totalNuParameters(m: Module): String =
    val nuParams = m.parameters.map(_.numel).sum
    if nuParams < 1e5
    then 
      s"${nuParams} parameters"
    else if nuParams < 1e6
    then 
      s"${nuParams/1e3}K parameters"
    else 
      s"${nuParams/1e6}M parameters"

  val SI = (BigInt(1000), Vector("B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"))
  val BINARY = (BigInt(1024), Vector("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"))

  /**
    * Converts a number of bytes into a human-readable string
    * such as `2.2 MB` or `8.0 EiB`.
    *
    * @param bytes the number of bytes we want to convert
    * @param si    if true, we use base 10 SI units where 1000 bytes are 1 kB.
    *              If false, we use base 2 IEC units where 1024 bytes are 1 KiB.
    * @return the bytes as a human-readable string
    * 
    * @see https://stackoverflow.com/questions/45885151/bytes-in-human-readable-format-with-idiomatic-scala
    * @see https://stackoverflow.com/questions/3758606/how-can-i-convert-byte-size-into-a-human-readable-format-in-java
    */
  def humanReadableSize(bytes: BigInt, si: Boolean = false): String = 
    // See https://en.wikipedia.org/wiki/Byte
    val (baseValue, unitStrings) =
      if (si)
        SI
      else
        BINARY

    def getExponent(curBytes: BigInt, baseValue: BigInt, curExponent: Int = 0): Int =
      if (curBytes < baseValue) 
      then
        curExponent
      else
        val newExponent = 1 + curExponent
        // getExponent(curBytes / (baseValue * newExponent), baseValue, newExponent)
        getExponent(curBytes / baseValue, baseValue, newExponent)

    val exponent = getExponent(bytes, baseValue)
    val divisor = baseValue.pow( exponent)
    val unitString = unitStrings(exponent)

    // Divide the bytes and show one digit after the decimal point
    f"${bytes.toDouble / divisor.toDouble}%.1f $unitString"

  // https://users.scala-lang.org/t/timing-a-computation/5361/4
  inline def elapsed[R](inline block: => R): (Long, R) = {
    val t0 = System.nanoTime()
    val r = block
    val t1 = System.nanoTime()
    val elapsed = t1 - t0
    // elapsed time in nanoseconds
    (elapsed, r)
  }

  inline def elapsedOnly[R](inline block: => R): Long = elapsed(block)._1
    

  // https://stackoverflow.com/questions/625433/how-to-convert-milliseconds-to-x-mins-x-seconds-in-java
  // https://www.skptricks.com/2018/09/convert-milliseconds-into-days-hours-minutes-seconds-in-java.html
  def durationParts(nanoSeconds: Long) =
    val duration = java.time.Duration.ofNanos(nanoSeconds)
    val d = duration.toDaysPart()
    val hh = duration.toHoursPart()
    val mm = duration.toMinutesPart()
    val ss = duration.toSecondsPart()
    val ms = duration.toMillisPart()
    val ns = duration.toNanosPart()
    (d, hh, mm, ss, ms,ns)
  
  def humanReadableDuration(nanoSeconds: Long) =
    val (d, hh, mm, ss, ms, ns) = durationParts(nanoSeconds)
    String.format("%02d %02d:%02d:%02d.%03d", d, hh, mm, ss, ms)

  // Code starts here

  /*
  # hyperparameters
  batch_size = 64 # how many independent sequences will we process in parallel?
  block_size = 256 # what is the maximum context length for predictions?
  max_iters = 5000
  eval_interval = 500
  learning_rate = 3e-4
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  eval_iters = 200
  n_embd = 384
  n_head = 6
  n_layer = 6
  dropout = 0.2
  # ------------
  */

  val batch_size = 64 // how many independent sequences will we process in parallel?
  val block_size = 256 // what is the maximum context length for predictions?
  val max_iters = 5000  
  val eval_interval = 500
  val learning_rate = 3e-4
  val device = if torch.cuda.isAvailable then CUDA else CPU
  val eval_iters = 200
  val n_embed = 384
  val n_head = 6
  val n_layer = 6
  val dropout = 0.2

  // Initialize weights
  val init_mean= 0.0 // 0,0
  val init_std= 0.2 // 0.02

  /*
  torch.manual_seed(1337)
  */
  torch.manualSeed(1337)

  /*
  # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  with open('input.txt', 'r', encoding='utf-8') as f:
      text = f.read()
  */
  val DATA = "data"
  val INPUT = "input.txt"
  val URL_ = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

  val dataDir = os.pwd / DATA
  if ! os.exists(dataDir)
  then
      println(s"Creating folder: $dataDir")
      os.makeDir(dataDir) 

  val dataFile = dataDir / "input.txt"
  if ! os.exists(dataFile)
  then
      // Assume the URL is valid
      val uri = URI.create(URL_)
      val url = uri.toURL() // URL(URL_)
      println(s"Downloading from $url to $dataFile")
      val nuBytes = Using.resource(url.openStream()) { inputStream =>
      //os.write(dataFile, inputStream)
      Files.copy(inputStream, dataFile.toNIO)
      }
      println(s"Read ${nuBytes} bytes")
  else 
      println(s"File $dataFile already exists.")

  val text = os.read(dataFile)

  /*
  # here are all the unique characters that occur in this text
  chars = sorted(list(set(text)))
  vocab_size = len(chars)
  # create a mapping from characters to integers
  stoi = { ch:i for i,ch in enumerate(chars) }
  itos = { i:ch for i,ch in enumerate(chars) }
  encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
  decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
  */

  // here are all the unique characters that occur in this text
  val chars = SortedSet(text:_*)
  println(s"chars = ${chars.mkString(", ")}")
  val vocab_size = chars.size
  println(s"vocab_size = $vocab_size")

  // create a mapping from characters to integers
  val stoi = chars.zipWithIndex.map((ch, i) => ch -> i.toLong).toMap
  val itos = stoi.map((ch,i) => i -> ch)
  def encode(s: String) = s.map( c => stoi(c) )
  def decode(s: Seq[Long]) = s.map( i => itos(i) ).mkString

  println(s""""BiGram!" = "${decode(encode("BiGram!"))}"""")

  /*
  # Train and test splits
  data = torch.tensor(encode(text), dtype=torch.long)
  n = int(0.9*len(data)) # first 90% will be train, rest val
  train_data = data[:n]
  val_data = data[n:]
  */

  // Train and test splits
  val data = torch.Tensor(encode(text)).long
  val n = (0.9 * len(data)).toInt // first 90% will be train, rest val
  val train_data = data(Slice(None, n))
  val val_data = data(Slice(n, None))

  /*
  # data loading
  def get_batch(split):
      # generate a small batch of data of inputs x and targets y
      data = train_data if split == 'train' else val_data
      ix = torch.randint(len(data) - block_size, (batch_size,))
      x = torch.stack([data[i:i+block_size] for i in ix])
      y = torch.stack([data[i+1:i+block_size+1] for i in ix])
      x, y = x.to(device), y.to(device)
      return x, y
  */

  // data loading
  def getBatch(split: String) = 
    // generate a small batch of data of inputs x and targets y
    val data = if split == "train" then train_data else val_data
    val ix = torch.randint(0, len(data) - block_size, Seq(batch_size)).to(dtype = int32)
    val stacks_x = ix.toSeq.map(i => data(Slice(i, i+block_size)))
    val x = torch.stack(stacks_x)
    val stacks_y = ix.toSeq.map(i => data(Slice(i+1, i+block_size+1)))
    val y = torch.stack(stacks_y)
    ( x.to(device), y.to(device) )

  val (xb, yb) = getBatch("train")
  println("xb:")
  println(decode( xb(0).toSeq ))
  println("yb:")
  println(decode( yb(0).toSeq ))

  trait BigramLanguageModel extends nn.Module:
    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None): (Tensor[Float32], Tensor[Float32])
    def generate(idx: Tensor[Int64], max_new_tokens: Int): Tensor[Int64]
    def apply(x: Tensor[Int64], y: Tensor[Int64]): (Tensor[Float32], Tensor[Float32])
    def apply(x: Tensor[Int64]): (Tensor[Float32], Tensor[Float32])


  /*
  @torch.no_grad()
  def estimate_loss():
      out = {}
      model.eval()
      for split in ['train', 'val']:
          losses = torch.zeros(eval_iters)
          for k in range(eval_iters):
              X, Y = get_batch(split)
              logits, loss = model(X, Y)
              losses[k] = loss.item()
          out[split] = losses.mean()
      model.train()
      return out
  */

  def estimateLoss(model: BiGram.BigramLanguageModel) = 
    val out = scala.collection.mutable.Map[String, Float]()
    model.eval()

    for 
      split <- List("train", "val")
    do
      // println(s"Estimate '$split' loss")
      val losses: Tensor[Float32] = torch.zeros(eval_iters, device=device)
      for 
        k <- 0 until eval_iters
      do
        // Need to release memory
        Using.resource(new PointerScope()) { p =>
          val (x, y) = getBatch(split)
          val (logits, loss) = model(x, y)
          losses(Seq(k)) = loss.item
        }

      out(split) = losses.mean.item

    model.train()
    out

  /*
  class Head(nn.Module):
      """ one head of self-attention """

      def __init__(self, head_size):
          super().__init__()
          self.key = nn.Linear(n_embd, head_size, bias=False)
          self.query = nn.Linear(n_embd, head_size, bias=False)
          self.value = nn.Linear(n_embd, head_size, bias=False)
          self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          # input of size (batch, time-step, channels)
          # output of size (batch, time-step, head size)
          B,T,C = x.shape
          k = self.key(x)   # (B,T,hs)
          q = self.query(x) # (B,T,hs)
          # compute attention scores ("affinities")
          wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
          wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
          wei = F.softmax(wei, dim=-1) # (B, T, T)
          wei = self.dropout(wei)
          # perform the weighted aggregation of the values
          v = self.value(x) # (B,T,hs)
          out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
          return out
  */

  class Head[D <: FloatNN: Default](
          n_embed: Int, 
          head_size: Int, 
          block_size: Int,
          dropout: Double
          ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    val key = register( nn.Linear[D](n_embed, head_size, addBias=false) )
    val query = register( nn.Linear[D](n_embed, head_size, addBias=false) )
    val value = register( nn.Linear[D](n_embed, head_size, addBias=false) )
    val ones = torch.ones[D](Seq(block_size, block_size), dtype=key.paramType)
    val tril = registerBuffer(torch.tril(ones), "tril")
    val drop = register( nn.Dropout( dropout ) )


    def forward(x: Tensor[D]): Tensor[D] =
      // input of size (batch, time-step, channels) = (B,T,C)
      // output of size (batch, time-step, head size) = (B,T,H)
      // batch, number of time steps, channels
      val Seq(b,t,c) = x.shape
      assert(block_size == t, "Block size must be equal to time step")
      val k = key(x)   // (B,T,C) @ (C,H) -> (B,T,H)
      val q = query(x) // (B,T,H)
      // compute attention scores ("affinities")
      // hs should be the head size
      val hs = k.size.last
      assert(head_size == hs, "Head size does not match k")
      val qk = q `@` k.transpose(-2, -1) * Tensor(hs).pow(-0.5).to(dtype=q.dtype)  // (B, T, H) @ (B, H, T) -> (B, T, T)
      val mask = qk.maskedFill(tril(Slice(None,n), Slice(None,n)) == 0, Float.NegativeInfinity).to(dtype=q.dtype) // (B, T, T)
      val soft = F.softmax(mask, dim= -1) // (B, T, T)
      val wei = drop( soft )
      // perform the weighted aggregation of the values
      val v = value(x) // (B,T,H)
      val out = wei `@` v // (B, T, T) @ (B, T, H) -> (B, T, H)
      out


    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(n_embed=$n_embed, head_size=$head_size, block_size=$block_size)"
  end Head

  class Head_2[D <: FloatNN: Default](
          nEmbed: Int, 
          headSize: Int, 
          blockSize: Int
          //drop: Double
          ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    val key = register( nn.Linear[D](nEmbed, headSize, addBias=false) )
    val query = register( nn.Linear[D](nEmbed, headSize, addBias=false) )
    val value = register( nn.Linear[D](nEmbed, headSize, addBias=false) )
    val ones = torch.ones[D](Seq(blockSize, blockSize), dtype=key.paramType)
    val tril = registerBuffer(torch.tril(ones), "tril")
    val drop = register( nn.Dropout( dropout ) )

    def forward(x: Tensor[D]): Tensor[D] =
      // input of size (batch, time-step, channels) = (B,T,C)
      // output of size (batch, time-step, head size) = (B,T,H)
      // batch, number of time steps, channels
      val Seq(b,t,c) = x.shape
      // fails on generate ?
      // assert(blockSize == t, "Block size must be equal to time step")

      // key = Linear(inFeatures=C, outFeatures=T, bias=false)
      val k = key(x)   // (B,T,C) @ (C,H) -> (B,T,H)
      val q = query(x) // (B,T,H)
      // compute attention scores ("affinities")
      // c should be the head size
      val hs = k.size.last
      assert(headSize == hs, "Head size does not match k")
      val qk = q `@` k.transpose(-2, -1) * Tensor(hs).pow(-0.5).to(dtype=q.dtype)  // (B, T, H) @ (B, H, T) -> (B, T, T)
      // val mask = qk.maskedFill(tril == 0, Float.NegativeInfinity) // (B, T, T)
      val mask = qk.maskedFill(tril((º`:`n), (º`:`n)) == 0, Float.NegativeInfinity).to(dtype=q.dtype) // (B, T, T)
      // val softmax = F.softmax(mask, dim= -1) // (B, T, T)
      // val wei = dropout(softmax)
      val soft = F.softmax(mask, dim= -1) // (B, T, T)
      val wei = drop( soft )
      // perform the weighted aggregation of the values
      val v = value(x) // (B,T,H)
      val out = wei `@` v // (B, T, T) @ (B, T, H) -> (B, T, H)
      out

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(n_embed=$nEmbed, head_size=$headSize, block_size=$blockSize)"
  end Head_2


  /*
  class MultiHeadAttention(nn.Module):
      """ multiple heads of self-attention in parallel """

      def __init__(self, num_heads, head_size):
          super().__init__()
          self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
          self.proj = nn.Linear(head_size * num_heads, n_embd)
          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          out = torch.cat([h(x) for h in self.heads], dim=-1)
          out = self.dropout(self.proj(out))
          return out
  */

  class MultiHeadAttention[D <: FloatNN: Default](
                            numHeads: Int, 
                            nEmbed: Int, 
                            headSize: Int, 
                            blockSize: Int,
                            dropout: Double
                          ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    // Cannot register with the same name
    // val hs = 0 until numHeads map{ _ => register(Head_2(nEmbed, headSize, blockSize)) }
    // val hs = 0 until numHeads map{ i => register_i(this, BiGram.Head_2(nEmbed, headSize, blockSize), i) }
    // TODO: out of mem
    // println( s"$nEmbed, $headSize, $blockSize" )
    val hs = 0 until numHeads map{ i => register_i(this, Head_2(nEmbed, headSize, blockSize), i) }
    // val hs = 0 until numHeads map{ i => register_i(this, Head(nEmbed, headSize, blockSize, dropout), i) }
    val heads = register( nn.ModuleList( hs:_* ) )
    // TODO: BUG - self.proj = nn.Linear(head_size * num_heads, n_embd)
    val proj = register( nn.Linear(headSize * numHeads, nEmbed) )
    // val proj = register( nn.Linear(nEmbed, nEmbed) )
    val drop = register( nn.Dropout(dropout) )

    def forward(x: Tensor[D]): Tensor[D] =
        //torch.cat(heads.modules.map( h => h(x) ), dim=1)
        val out = torch.cat(heads.map( h => h(x) ).toSeq, dim= -1)
        drop( proj(out) )

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass().getSimpleName()}(numHeads=$numHeads, nEmbed=$nEmbed, headSize=$headSize, blockSize=$blockSize)"


  end MultiHeadAttention

  /*
  class FeedFoward(nn.Module):
      """ a simple linear layer followed by a non-linearity """

      def __init__(self, n_embd):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(n_embd, 4 * n_embd),
              nn.ReLU(),
              nn.Linear(4 * n_embd, n_embd),
              nn.Dropout(dropout),
          )

      def forward(self, x):
          return self.net(x)
  */
  class FeedForward[D <: FloatNN: Default](
                          nEmbed: Int 
                        ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:
    val net = register( nn.Sequential(
                    // Increase output dimension by 4
                    nn.Linear(nEmbed, 4L * nEmbed),
                    nn.ReLU(),
                    // Decrease output dimension by 4
                    nn.Linear(4L*nEmbed, nEmbed), // residual network implemented using projection - why not on x directly?
                    nn.Dropout(dropout),
              )
            )

    def forward(x: Tensor[D]): Tensor[D] = net(x)

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass().getSimpleName()}(nEmbed = $nEmbed)"

  end FeedForward


  /*
  class Block(nn.Module):
      """ Transformer block: communication followed by computation """

      def __init__(self, n_embd, n_head):
          # n_embd: embedding dimension, n_head: the number of heads we'd like
          super().__init__()
          head_size = n_embd // n_head
          self.sa = MultiHeadAttention(n_head, head_size)
          self.ffwd = FeedFoward(n_embd)
          self.ln1 = nn.LayerNorm(n_embd)
          self.ln2 = nn.LayerNorm(n_embd)

      def forward(self, x):
          x = x + self.sa(self.ln1(x))
          x = x + self.ffwd(self.ln2(x))
          return x
  */
  class Block[D <: FloatNN: Default](
                          nEmbed: Int, 
                          nHead: Int,
                          blockSize: Int,
                          vocabSize: Int,
                          dropout: Double
                        ) extends torch.nn.modules.TensorModule[D]: 

    // n_embd: embedding dimension, n_head: the number of heads we'd like
    val headSize = nEmbed / nHead
    val sa = register( MultiHeadAttention(
                                            numHeads = nHead, 
                                            nEmbed = nEmbed, 
                                            headSize = headSize,
                                            blockSize = blockSize,
                                            dropout = dropout) 
                                            )
    val ffwd = register( FeedForward(nEmbed) )
    // val lm_head = register( nn.Linear(nEmbed, vocabSize) )
    val ln1 = register( nn.LayerNorm(Seq(nEmbed)) )
    val ln2 = register( nn.LayerNorm(Seq(nEmbed)) )

    def forward(x: Tensor[D]): Tensor[D] = 
      val x1 = x + sa( ln1(x) )
      // TODO: BUG found
      x1 + ffwd( ln2(x1) )

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(nEmbed = $nEmbed)"

  end Block

  /*
  class GPTLanguageModel(nn.Module):

      def __init__(self):
          super().__init__()
          # each token directly reads off the logits for the next token from a lookup table
          self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
          self.position_embedding_table = nn.Embedding(block_size, n_embd)
          self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
          self.ln_f = nn.LayerNorm(n_embd) # final layer norm
          self.lm_head = nn.Linear(n_embd, vocab_size)

          # better init, not covered in the original GPT video, but important, will cover in followup video
          self.apply(self._init_weights)

      def _init_weights(self, module):
          if isinstance(module, nn.Linear):
              torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
              if module.bias is not None:
                  torch.nn.init.zeros_(module.bias)
          elif isinstance(module, nn.Embedding):
              torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

      def forward(self, idx, targets=None):
          B, T = idx.shape

          # idx and targets are both (B,T) tensor of integers
          tok_emb = self.token_embedding_table(idx) # (B,T,C)
          pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
          x = tok_emb + pos_emb # (B,T,C)
          x = self.blocks(x) # (B,T,C)
          x = self.ln_f(x) # (B,T,C)
          logits = self.lm_head(x) # (B,T,vocab_size)

          if targets is None:
              loss = None
          else:
              B, T, C = logits.shape
              logits = logits.view(B*T, C)
              targets = targets.view(B*T)
              loss = F.cross_entropy(logits, targets)

          return logits, loss

      def generate(self, idx, max_new_tokens):
          # idx is (B, T) array of indices in the current context
          for _ in range(max_new_tokens):
              # crop idx to the last block_size tokens
              idx_cond = idx[:, -block_size:]
              # get the predictions
              logits, loss = self(idx_cond)
              # focus only on the last time step
              logits = logits[:, -1, :] # becomes (B, C)
              # apply softmax to get probabilities
              probs = F.softmax(logits, dim=-1) # (B, C)
              # sample from the distribution
              idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
              # append sampled index to the running sequence
              idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
          return idx
  */

  // https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
  class GPTLanguageModel(
    vocabSize: Int, 
    blockSize:Int, 
    nEmbed: Int,
    nBlocks: Int,
    nHead: Int,
    dropout: Double
    ) extends BiGram.BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    val blocks_i = 0 until nBlocks map { i => Block(nEmbed, nHead, blockSize, vocabSize, dropout) }
    // val blocks_i = 0 until nBlocks map { i => BiGram.Block_4(nEmbed, nHead, blockSize, vocabSize) }
    val blocks = register(  nn.Sequential( blocks_i:_* ) )
    val ln_f = register( nn.LayerNorm(Seq(nEmbed)) )
    val lm_head = register( nn.Linear(nEmbed, vocabSize) )

    // better init, not covered in the original GPT video, but important, will cover in followup video
    // self.apply(self._init_weights)
    modules.foreach(init_weights)
// 
//     def _init_weights(self, module):
//         if isinstance(module, nn.Linear):
//             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
//             if module.bias is not None:
//                 torch.nn.init.zeros_(module.bias)
//         elif isinstance(module, nn.Embedding):
//             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    // private def init_weights[D <: FloatNN | ComplexNN](m: Module with HasWeight[D]): Unit = 
    private def init_weights(m: Module): Unit = 
      m match
        case lm : nn.Linear[_] => 
          torch.nn.init.normal_(lm.weight, mean=init_mean, std=init_std)
          if lm.hasBias()
          then
            torch.nn.init.zeros_(lm.bias)
        case em : nn.Embedding[_] => 
          torch.nn.init.normal_(em.weight, mean=init_mean, std=init_std)
        case _ => ()


    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =
      val Seq(b,t) = idx.shape

      // idx and targets are both (B,T) tensor of integers
      // idx is (B,T)
      val token_embed = token_embedding_table( idx ) // (B,T,C) where C is nEmbed
      // positions of tokens
      val pos = torch.arange(0L,t, device=device) // (T) were T is the block size?
      val pos_embed = position_embedding_table( pos ) // (T,C)
      // Add the position embeddings
      val x0 = token_embed + pos_embed // (B,T,C)
      val x1 = blocks(x0) // apply blocks of self-attention (B,T,C) - C = nEmbed
      val x2 = ln_f(x1) // (B,T,C) layer norm
      val logits = lm_head( x2 ) // (B,T,C) @ (C,vocabSize) -> (B,T,vocabSize)

      if targets.isEmpty
      then
        val zero = torch.Tensor(0.0f) 
        (logits, zero)
      else
        val Seq(b,t,c) = logits.shape
        val logitsV = logits.view(b*t, c)  // batch size x number of classes
        val targetsV = targets.get.view(b*t) 
        // NOTE: we are using all of the time-steps (symbols) to calculate error
        // Why do we not use only the last one, the prediction?
        val loss = F.crossEntropy(logitsV, targetsV)
        (logitsV, loss)

    def generate(idx: Tensor[Int64], max_new_tokens: Int) =
      var idx_ = idx.copy_(idx)
      printMemoryInfo(0)

      // idx is (B, T) array of indices in the current context
      for i <- 0 until max_new_tokens
      do
        // crop idx to the last block_size tokens
        val idx_cond = idx_(Slice(), Slice(-blockSize,None))
        // get the predictions
        val (logits_t, loss) = apply(idx_cond)
        // focus only on the last time step
        val logits = logits_t(Slice(), -1, Slice()) // becomes (B, C)
        // apply softmax to get probabilities
        val probs = F.softmax(logits, dim = -1L) // (B, C)
        // sample from the distribution
        val idx_next = torch.multinomial(probs, numSamples=1) // (B, 1)
        // append sampled index to the running sequence
        idx_ = torch.cat(Seq(idx_, idx_next), dim=1) // (B, T+1)
        printMemoryInfo(0)
      idx_

    def apply(x: Tensor[Int64], y: Tensor[Int64]) =
      forward(x, Some(y) )

    def apply(x: Tensor[Int64]) =
      forward(x, None )


  end GPTLanguageModel

  /*
  for iter in range(max_iters):

      # every once in a while evaluate the loss on train and val sets
      if iter % eval_interval == 0 or iter == max_iters - 1:
          losses = estimate_loss()
          print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

      # sample a batch of data
      xb, yb = get_batch('train')

      # evaluate the loss
      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

  */

  def train(m : BiGram.BigramLanguageModel, learningRate: Double, maxIterations: Int): Unit =
    m.to(device)

    // print the number of parameters in the model
    val nuParams = m.parameters.map(_.numel).sum
    //println(s"${nuParams/1e6}M parameters")
    println(s"Device = ${device}")
    println(s"${nuParams} parameters")
    println(s"learningRate = ${learningRate}")
    println(s"maxIterations = ${maxIterations}")
    println(s"dropout = ${dropout}")
    // Get values from nvidia-smi -l 1
    val gpuTotal: BigInt = BigInt(24576) * BINARY._1 * BINARY._1
    println(s"GPU total = ${humanReadableSize(gpuTotal)}") // MiB
    val gpuUsed: BigInt = BigInt(7056) * BINARY._1 * BINARY._1
    println(s"GPU used = ${humanReadableSize(gpuUsed)}") // MiB
    // numel() * itemsize() = nbytes
    val nBytes = m.parameters.map(t => t.native.nbytes()).sum
    println(s"${nuParams} parameters >= ${nBytes} bytes = ${humanReadableSize(nBytes)}")
    

    m.train()
    // create a PyTorch optimizer
    val optimizer = torch.optim.AdamW(m.parameters, lr=learningRate)
    
    var delta = 0L
    var total = 0L
    for iter <- 0 until maxIterations
    do
      // make sure we deallocate intermediate tensors in time
      Using.resource(new PointerScope()) { p => 

        // every once in a while evaluate the loss on train and val sets
        if (iter % eval_interval == 0) || (iter == maxIterations - 1)
        then
          val losses = estimateLoss(m)
          val memoryBytes = humanReadableSize( Pointer.physicalBytes() )
          delta = delta / eval_interval
          val accumulated = humanReadableDuration(total)
          val perIteration = humanReadableDuration(delta)
          println(s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}, mem $memoryBytes @ ${accumulated}, mean $perIteration")
          //printMemoryInfo(device.index)
          //printMemoryInfo(0)
          delta = 0L

        val elapsed = elapsedOnly {
          // sample a batch of datas
          val (xb, yb) = getBatch("train")

          // evaluate the loss
          val (logits, loss) = m(xb, yb)
          optimizer.zeroGrad(setToNone=true)
          loss.backward()
          optimizer.step()
        }
        delta = delta + elapsed
        total = total + elapsed
      }
      
    val losses = estimateLoss(m)
    val accumulated = humanReadableDuration(total)
    val perIteration = humanReadableDuration(total / maxIterations)
    println(s"step ${maxIterations}: train loss ${losses("train")}, val loss ${losses("val")}, @ ${accumulated}, mean $perIteration")


  def main(args: Array[String]): Unit =
    println("V2")

    /*
    model = GPTLanguageModel()
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    */

    val model = GPTLanguageModel(vocabSize = vocab_size, blockSize = block_size, nEmbed = n_embed, nBlocks = n_layer, nHead = n_head, dropout= dropout)
    println(totalNuParameters(model))
    println(moduleInfoString(model))
    //printMemoryInfo(device.index)
    printMemoryInfo(0)

    /*
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    See train loop above
    */

    // train(model, learning_rate, 67000)
    // train(model, 1e-4, 41500) // No weight init
    // train(model, 2e-4, 41500)  // with weight init 
    train(model, 1e-4, 41500)  // with weight init 

    /*    
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
    */
    
    // TODO: Bug just (1,1) ?
    val context = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device)
    // val context = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device)
    val embedding = model.generate(idx = context, max_new_tokens=500)(0)
    val decoded = decode(embedding.toSeq)
    println(s"decode:'$decoded'")

    ()

  
end V2

