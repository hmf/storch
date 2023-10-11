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



/**
  * ./mill examples.runMain gpt.V2
  * nohup ./mill examples.runMain gpt.V2 > v2_0.txt 2>&1 &
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

    val key = register( nn.Linear[D](n_embed, head_size, hasBias=false) )
    val query = register( nn.Linear[D](n_embed, head_size, hasBias=false) )
    val value = register( nn.Linear[D](n_embed, head_size, hasBias=false) )
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

    val key = register( nn.Linear[D](nEmbed, headSize, hasBias=false) )
    val query = register( nn.Linear[D](nEmbed, headSize, hasBias=false) )
    val value = register( nn.Linear[D](nEmbed, headSize, hasBias=false) )
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

    // TODO: _init_weights

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

    /*
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    See train loop above
    */

    train(model, learning_rate, 67000)

    /*    
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
    */
    // TODO: Bug just (1,1) ?
    val context = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device)
    val embedding = model.generate(idx = context, max_new_tokens=500)(0)
    val decoded = decode(embedding.toSeq)
    println(s"decode:'$decoded'")

    ()

  
end V2


/*


https://devicetests.com/identify-process-using-gpu
nvidia-smi
gpustat

nvidia-smi --query-compute-apps=pid --format=csv,noheader
pid
764160

inside the container
ps ax | grep V2
pid = 310988

outside the container
ps ax | grep V2
pid = 764160

https://github.com/wookayin/gpustat
python package

#~4
batch of 64 too large - out of memory, using 32
24576MiB = 24 GibiBytes
18848MiB = 18.406 used
reported 2.2 GiB

13347905 Float32 = 13347905 * 4 = 53391620 bytes = 50.91 mebibytes

batch_size = 32
Device = Device(CUDA,-1)
13347905 parameters
learningRate = 1.0E-6
maxIterations = 450000
dropout = 0.2
step 0: train loss 4.262311, val loss 4.2712555, mem 1.3 GiB @ 00 00:00:00.000, mean 00 00:00:00.000
step 500: train loss 3.3336122, val loss 3.3660629, mem 1.7 GiB @ 00 00:00:18.685, mean 00 00:00:00.037
step 1000: train loss 3.1699007, val loss 3.2048519, mem 1.8 GiB @ 00 00:00:37.255, mean 00 00:00:00.037
step 1500: train loss 3.0736809, val loss 3.0911696, mem 1.6 GiB @ 00 00:00:55.846, mean 00 00:00:00.037
step 2000: train loss 2.9691136, val loss 2.9894543, mem 1.6 GiB @ 00 00:01:14.605, mean 00 00:00:00.037
step 2500: train loss 2.8912554, val loss 2.9229655, mem 1.6 GiB @ 00 00:01:33.313, mean 00 00:00:00.037
step 3000: train loss 2.8240547, val loss 2.8245873, mem 1.6 GiB @ 00 00:01:51.962, mean 00 00:00:00.037
step 3500: train loss 2.7625897, val loss 2.772149, mem 1.6 GiB @ 00 00:02:11.047, mean 00 00:00:00.038
step 4000: train loss 2.7203698, val loss 2.733889, mem 1.6 GiB @ 00 00:02:29.779, mean 00 00:00:00.037
step 4500: train loss 2.688148, val loss 2.6862524, mem 1.5 GiB @ 00 00:02:48.427, mean 00 00:00:00.037
step 5000: train loss 2.6386409, val loss 2.6464264, mem 1.8 GiB @ 00 00:03:07.519, mean 00 00:00:00.038
step 5500: train loss 2.6217017, val loss 2.622003, mem 2.0 GiB @ 00 00:03:26.921, mean 00 00:00:00.038
step 6000: train loss 2.581709, val loss 2.5906417, mem 2.0 GiB @ 00 00:03:45.671, mean 00 00:00:00.037
step 6500: train loss 2.5601115, val loss 2.5654008, mem 2.0 GiB @ 00 00:04:04.427, mean 00 00:00:00.037
step 7000: train loss 2.5319061, val loss 2.539647, mem 1.6 GiB @ 00 00:04:22.991, mean 00 00:00:00.037
step 7500: train loss 2.5145438, val loss 2.510933, mem 1.6 GiB @ 00 00:04:41.316, mean 00 00:00:00.036
step 8000: train loss 2.494046, val loss 2.4868686, mem 1.5 GiB @ 00 00:04:59.908, mean 00 00:00:00.037
step 8500: train loss 2.4917753, val loss 2.4922922, mem 1.5 GiB @ 00 00:05:18.406, mean 00 00:00:00.036
step 9000: train loss 2.4516144, val loss 2.46908, mem 1.5 GiB @ 00 00:05:37.050, mean 00 00:00:00.037
step 9500: train loss 2.4529207, val loss 2.4529982, mem 1.8 GiB @ 00 00:05:56.012, mean 00 00:00:00.037
step 10000: train loss 2.444607, val loss 2.4461143, mem 1.8 GiB @ 00 00:06:14.782, mean 00 00:00:00.037
step 10500: train loss 2.4204478, val loss 2.4330733, mem 1.8 GiB @ 00 00:06:33.469, mean 00 00:00:00.037
step 11000: train loss 2.4148889, val loss 2.4195092, mem 1.5 GiB @ 00 00:06:52.032, mean 00 00:00:00.037
step 11500: train loss 2.3990304, val loss 2.411274, mem 1.5 GiB @ 00 00:07:11.670, mean 00 00:00:00.039
step 12000: train loss 2.393478, val loss 2.3953273, mem 1.9 GiB @ 00 00:07:31.293, mean 00 00:00:00.039
step 12500: train loss 2.3888512, val loss 2.394262, mem 2.0 GiB @ 00 00:07:50.280, mean 00 00:00:00.037
step 13000: train loss 2.3675349, val loss 2.3934054, mem 2.1 GiB @ 00 00:08:08.785, mean 00 00:00:00.037
step 13500: train loss 2.3716033, val loss 2.367454, mem 2.1 GiB @ 00 00:08:27.692, mean 00 00:00:00.037
step 14000: train loss 2.3729062, val loss 2.3851686, mem 2.1 GiB @ 00 00:08:46.828, mean 00 00:00:00.038
step 14500: train loss 2.360136, val loss 2.3735313, mem 2.1 GiB @ 00 00:09:05.638, mean 00 00:00:00.037
step 15000: train loss 2.354177, val loss 2.3663445, mem 2.1 GiB @ 00 00:09:24.203, mean 00 00:00:00.037
step 15500: train loss 2.344297, val loss 2.3483455, mem 2.1 GiB @ 00 00:09:42.685, mean 00 00:00:00.036
step 16000: train loss 2.3344684, val loss 2.3466318, mem 2.2 GiB @ 00 00:10:01.116, mean 00 00:00:00.036
step 16500: train loss 2.3210425, val loss 2.3291836, mem 2.2 GiB @ 00 00:10:20.440, mean 00 00:00:00.038
step 17000: train loss 2.3170369, val loss 2.3303854, mem 2.2 GiB @ 00 00:10:39.706, mean 00 00:00:00.038
step 17500: train loss 2.3139145, val loss 2.331151, mem 2.2 GiB @ 00 00:10:58.239, mean 00 00:00:00.037
step 18000: train loss 2.3126464, val loss 2.3228002, mem 2.2 GiB @ 00 00:11:16.874, mean 00 00:00:00.037
step 18500: train loss 2.2990053, val loss 2.3157554, mem 2.2 GiB @ 00 00:11:36.046, mean 00 00:00:00.038
step 19000: train loss 2.292631, val loss 2.3061376, mem 2.2 GiB @ 00 00:11:55.651, mean 00 00:00:00.039
step 19500: train loss 2.2753887, val loss 2.282227, mem 2.2 GiB @ 00 00:12:15.066, mean 00 00:00:00.038
step 20000: train loss 2.2852285, val loss 2.292785, mem 2.2 GiB @ 00 00:12:34.530, mean 00 00:00:00.038
step 20500: train loss 2.2807667, val loss 2.3035812, mem 2.2 GiB @ 00 00:12:55.462, mean 00 00:00:00.041
step 21000: train loss 2.2744253, val loss 2.291565, mem 2.2 GiB @ 00 00:13:15.249, mean 00 00:00:00.039
step 21500: train loss 2.2664242, val loss 2.2650592, mem 2.2 GiB @ 00 00:13:34.447, mean 00 00:00:00.038
step 22000: train loss 2.2510986, val loss 2.276302, mem 2.2 GiB @ 00 00:13:52.864, mean 00 00:00:00.036
step 22500: train loss 2.2519617, val loss 2.270701, mem 2.2 GiB @ 00 00:14:11.453, mean 00 00:00:00.037
step 23000: train loss 2.2575552, val loss 2.262467, mem 2.2 GiB @ 00 00:14:30.091, mean 00 00:00:00.037
step 23500: train loss 2.2461667, val loss 2.2625213, mem 2.2 GiB @ 00 00:14:48.890, mean 00 00:00:00.037
step 24000: train loss 2.243675, val loss 2.2598832, mem 2.2 GiB @ 00 00:15:07.518, mean 00 00:00:00.037
step 24500: train loss 2.2335835, val loss 2.2453442, mem 2.2 GiB @ 00 00:15:26.145, mean 00 00:00:00.037
step 25000: train loss 2.2263505, val loss 2.2450097, mem 2.2 GiB @ 00 00:15:44.581, mean 00 00:00:00.036
step 25500: train loss 2.2252822, val loss 2.2469742, mem 2.2 GiB @ 00 00:16:03.121, mean 00 00:00:00.037
step 26000: train loss 2.2226584, val loss 2.2366908, mem 2.2 GiB @ 00 00:16:21.610, mean 00 00:00:00.036
step 26500: train loss 2.2242975, val loss 2.2323186, mem 2.2 GiB @ 00 00:16:40.038, mean 00 00:00:00.036
step 27000: train loss 2.220194, val loss 2.221968, mem 2.2 GiB @ 00 00:16:58.551, mean 00 00:00:00.037
step 27500: train loss 2.2117057, val loss 2.2257288, mem 2.2 GiB @ 00 00:17:16.990, mean 00 00:00:00.036
step 28000: train loss 2.20084, val loss 2.22498, mem 2.2 GiB @ 00 00:17:35.527, mean 00 00:00:00.037
step 28500: train loss 2.1949475, val loss 2.2317848, mem 2.2 GiB @ 00 00:17:54.593, mean 00 00:00:00.038
step 29000: train loss 2.2007554, val loss 2.2267685, mem 2.2 GiB @ 00 00:18:13.276, mean 00 00:00:00.037
step 29500: train loss 2.1911948, val loss 2.2143266, mem 2.2 GiB @ 00 00:18:31.804, mean 00 00:00:00.037
step 30000: train loss 2.1906161, val loss 2.2195477, mem 2.2 GiB @ 00 00:18:50.044, mean 00 00:00:00.036
step 30500: train loss 2.1858075, val loss 2.216376, mem 2.2 GiB @ 00 00:19:08.505, mean 00 00:00:00.036
step 31000: train loss 2.1878498, val loss 2.202733, mem 2.2 GiB @ 00 00:19:26.991, mean 00 00:00:00.036
step 31500: train loss 2.1831527, val loss 2.2026868, mem 2.2 GiB @ 00 00:19:45.140, mean 00 00:00:00.036
step 32000: train loss 2.1752417, val loss 2.2016854, mem 2.2 GiB @ 00 00:20:03.648, mean 00 00:00:00.037
step 32500: train loss 2.1730018, val loss 2.2012823, mem 2.2 GiB @ 00 00:20:22.103, mean 00 00:00:00.036
step 33000: train loss 2.1779869, val loss 2.1983907, mem 2.2 GiB @ 00 00:20:41.821, mean 00 00:00:00.039
step 33500: train loss 2.172403, val loss 2.1837246, mem 2.2 GiB @ 00 00:21:00.580, mean 00 00:00:00.037
step 34000: train loss 2.1567879, val loss 2.1823664, mem 2.2 GiB @ 00 00:21:19.207, mean 00 00:00:00.037
step 34500: train loss 2.1571796, val loss 2.1875587, mem 2.2 GiB @ 00 00:21:38.031, mean 00 00:00:00.037
step 35000: train loss 2.1627057, val loss 2.186361, mem 2.2 GiB @ 00 00:21:56.777, mean 00 00:00:00.037
step 35500: train loss 2.1581306, val loss 2.1795056, mem 2.2 GiB @ 00 00:22:15.425, mean 00 00:00:00.037
step 36000: train loss 2.1419187, val loss 2.1740892, mem 2.2 GiB @ 00 00:22:33.845, mean 00 00:00:00.036
step 36500: train loss 2.14437, val loss 2.175493, mem 2.2 GiB @ 00 00:22:52.427, mean 00 00:00:00.037
step 37000: train loss 2.1395135, val loss 2.1674972, mem 2.2 GiB @ 00 00:23:10.893, mean 00 00:00:00.036
step 37500: train loss 2.139334, val loss 2.1679149, mem 2.2 GiB @ 00 00:23:29.489, mean 00 00:00:00.037
step 38000: train loss 2.1408162, val loss 2.1732433, mem 2.2 GiB @ 00 00:23:48.241, mean 00 00:00:00.037
step 38500: train loss 2.1433108, val loss 2.1735113, mem 2.2 GiB @ 00 00:24:08.069, mean 00 00:00:00.039
step 39000: train loss 2.1290123, val loss 2.1678183, mem 2.2 GiB @ 00 00:24:25.908, mean 00 00:00:00.035
step 39500: train loss 2.126224, val loss 2.153663, mem 2.2 GiB @ 00 00:24:43.806, mean 00 00:00:00.035
step 40000: train loss 2.1187236, val loss 2.1631012, mem 2.2 GiB @ 00 00:25:02.649, mean 00 00:00:00.037
step 40500: train loss 2.1270425, val loss 2.1526163, mem 2.2 GiB @ 00 00:25:21.900, mean 00 00:00:00.038
step 41000: train loss 2.1229808, val loss 2.1651435, mem 2.2 GiB @ 00 00:25:42.403, mean 00 00:00:00.041
step 41500: train loss 2.1123312, val loss 2.1609418, mem 2.2 GiB @ 00 00:26:00.822, mean 00 00:00:00.036
step 42000: train loss 2.1147144, val loss 2.1640427, mem 2.2 GiB @ 00 00:26:21.663, mean 00 00:00:00.041
step 42500: train loss 2.1274238, val loss 2.147262, mem 2.2 GiB @ 00 00:26:40.112, mean 00 00:00:00.036
step 43000: train loss 2.111654, val loss 2.1549854, mem 2.2 GiB @ 00 00:26:58.581, mean 00 00:00:00.036
step 43500: train loss 2.1077032, val loss 2.1497552, mem 2.2 GiB @ 00 00:27:16.979, mean 00 00:00:00.036
step 44000: train loss 2.1086404, val loss 2.1480615, mem 2.2 GiB @ 00 00:27:35.119, mean 00 00:00:00.036
step 44500: train loss 2.09795, val loss 2.1420248, mem 2.2 GiB @ 00 00:27:54.445, mean 00 00:00:00.038
step 45000: train loss 2.0986533, val loss 2.1407397, mem 2.2 GiB @ 00 00:28:13.067, mean 00 00:00:00.037
step 45500: train loss 2.1069658, val loss 2.1372073, mem 2.2 GiB @ 00 00:28:31.186, mean 00 00:00:00.036
step 46000: train loss 2.0985174, val loss 2.1409235, mem 2.2 GiB @ 00 00:28:49.303, mean 00 00:00:00.036
step 46500: train loss 2.0934982, val loss 2.1345923, mem 2.2 GiB @ 00 00:29:07.814, mean 00 00:00:00.037
step 47000: train loss 2.091304, val loss 2.1351974, mem 2.2 GiB @ 00 00:29:26.247, mean 00 00:00:00.036
step 47500: train loss 2.093283, val loss 2.1407595, mem 2.2 GiB @ 00 00:29:44.750, mean 00 00:00:00.037
step 48000: train loss 2.0953188, val loss 2.1244977, mem 2.2 GiB @ 00 00:30:03.943, mean 00 00:00:00.038
step 48500: train loss 2.0951722, val loss 2.135266, mem 2.2 GiB @ 00 00:30:23.887, mean 00 00:00:00.039
step 49000: train loss 2.0836008, val loss 2.127759, mem 2.2 GiB @ 00 00:30:42.368, mean 00 00:00:00.036
step 49500: train loss 2.0862753, val loss 2.1219108, mem 2.2 GiB @ 00 00:31:01.075, mean 00 00:00:00.037
step 50000: train loss 2.084155, val loss 2.1226802, mem 2.2 GiB @ 00 00:31:19.135, mean 00 00:00:00.036
step 50500: train loss 2.076774, val loss 2.118638, mem 2.2 GiB @ 00 00:31:37.490, mean 00 00:00:00.036
step 51000: train loss 2.0719635, val loss 2.125747, mem 2.2 GiB @ 00 00:31:56.073, mean 00 00:00:00.037
step 51500: train loss 2.081441, val loss 2.1183457, mem 2.2 GiB @ 00 00:32:14.898, mean 00 00:00:00.037
step 52000: train loss 2.0749075, val loss 2.1206903, mem 2.2 GiB @ 00 00:32:33.431, mean 00 00:00:00.037
step 52500: train loss 2.0661206, val loss 2.1101978, mem 2.2 GiB @ 00 00:32:52.058, mean 00 00:00:00.037
step 53000: train loss 2.0709722, val loss 2.1261191, mem 2.2 GiB @ 00 00:33:10.503, mean 00 00:00:00.036
step 53500: train loss 2.0692613, val loss 2.1133883, mem 2.2 GiB @ 00 00:33:30.164, mean 00 00:00:00.039
step 54000: train loss 2.0769336, val loss 2.1180606, mem 2.2 GiB @ 00 00:33:49.912, mean 00 00:00:00.039
step 54500: train loss 2.0635202, val loss 2.115873, mem 2.2 GiB @ 00 00:34:09.422, mean 00 00:00:00.039
step 55000: train loss 2.0715654, val loss 2.1106315, mem 2.2 GiB @ 00 00:34:27.666, mean 00 00:00:00.036
step 55500: train loss 2.065869, val loss 2.1097107, mem 2.2 GiB @ 00 00:34:47.766, mean 00 00:00:00.040
step 56000: train loss 2.0606825, val loss 2.1004524, mem 2.2 GiB @ 00 00:35:07.567, mean 00 00:00:00.039
step 56500: train loss 2.0426903, val loss 2.1045156, mem 2.2 GiB @ 00 00:35:26.428, mean 00 00:00:00.037
step 57000: train loss 2.038457, val loss 2.107041, mem 2.2 GiB @ 00 00:35:45.019, mean 00 00:00:00.037
step 57500: train loss 2.0555239, val loss 2.0994198, mem 2.2 GiB @ 00 00:36:03.714, mean 00 00:00:00.037
step 58000: train loss 2.0664034, val loss 2.1035283, mem 2.2 GiB @ 00 00:36:21.384, mean 00 00:00:00.035
step 58500: train loss 2.0562973, val loss 2.1019971, mem 2.2 GiB @ 00 00:36:40.312, mean 00 00:00:00.037
step 59000: train loss 2.051991, val loss 2.1047506, mem 2.2 GiB @ 00 00:37:00.205, mean 00 00:00:00.039
step 59500: train loss 2.0538726, val loss 2.110334, mem 2.2 GiB @ 00 00:37:18.744, mean 00 00:00:00.037
step 60000: train loss 2.0355794, val loss 2.1009374, mem 2.2 GiB @ 00 00:37:37.995, mean 00 00:00:00.038
step 60500: train loss 2.055535, val loss 2.0929925, mem 2.2 GiB @ 00 00:37:56.235, mean 00 00:00:00.036
step 61000: train loss 2.0514417, val loss 2.0905874, mem 2.2 GiB @ 00 00:38:15.315, mean 00 00:00:00.038
step 61500: train loss 2.0397792, val loss 2.0950384, mem 2.2 GiB @ 00 00:38:33.514, mean 00 00:00:00.036
step 62000: train loss 2.036438, val loss 2.1025505, mem 2.2 GiB @ 00 00:38:51.956, mean 00 00:00:00.036
step 62500: train loss 2.0460563, val loss 2.0786684, mem 2.2 GiB @ 00 00:39:10.292, mean 00 00:00:00.036
step 63000: train loss 2.0414298, val loss 2.098213, mem 2.2 GiB @ 00 00:39:29.081, mean 00 00:00:00.037
step 63500: train loss 2.0340989, val loss 2.0894616, mem 2.2 GiB @ 00 00:39:47.787, mean 00 00:00:00.037
step 64000: train loss 2.0324576, val loss 2.082341, mem 2.2 GiB @ 00 00:40:06.244, mean 00 00:00:00.036
step 64500: train loss 2.0342896, val loss 2.094168, mem 2.2 GiB @ 00 00:40:25.264, mean 00 00:00:00.038
step 65000: train loss 2.0430994, val loss 2.0783777, mem 2.2 GiB @ 00 00:40:44.839, mean 00 00:00:00.039
step 65500: train loss 2.0323691, val loss 2.0764027, mem 2.2 GiB @ 00 00:41:03.992, mean 00 00:00:00.038
step 66000: train loss 2.0272613, val loss 2.0775285, mem 2.2 GiB @ 00 00:41:23.803, mean 00 00:00:00.039
step 66500: train loss 2.0064697, val loss 2.0916383, mem 2.2 GiB @ 00 00:41:42.504, mean 00 00:00:00.037
step 67000: train loss 2.0184271, val loss 2.0691679, mem 2.2 GiB @ 00 00:42:01.375, mean 00 00:00:00.037
step 67500: train loss 2.0256345, val loss 2.0835552, mem 2.2 GiB @ 00 00:42:21.358, mean 00 00:00:00.039
step 68000: train loss 2.0218549, val loss 2.0719583, mem 2.2 GiB @ 00 00:42:40.305, mean 00 00:00:00.037
step 68500: train loss 2.0178354, val loss 2.0904727, mem 2.2 GiB @ 00 00:42:58.846, mean 00 00:00:00.037
step 69000: train loss 2.0275521, val loss 2.0754037, mem 2.2 GiB @ 00 00:43:17.228, mean 00 00:00:00.036
step 69500: train loss 2.0263355, val loss 2.0891266, mem 2.2 GiB @ 00 00:43:35.880, mean 00 00:00:00.037
step 70000: train loss 2.0144417, val loss 2.0769904, mem 2.2 GiB @ 00 00:43:54.569, mean 00 00:00:00.037
step 70500: train loss 2.0153618, val loss 2.0732038, mem 2.2 GiB @ 00 00:44:12.671, mean 00 00:00:00.036
step 71000: train loss 2.0022051, val loss 2.0746324, mem 2.2 GiB @ 00 00:44:31.623, mean 00 00:00:00.037
step 71500: train loss 1.9978979, val loss 2.0684536, mem 2.2 GiB @ 00 00:44:50.095, mean 00 00:00:00.036
step 72000: train loss 2.0103166, val loss 2.0695522, mem 2.2 GiB @ 00 00:45:07.917, mean 00 00:00:00.035
step 72500: train loss 2.0142696, val loss 2.0616121, mem 2.2 GiB @ 00 00:45:26.083, mean 00 00:00:00.036
step 73000: train loss 2.0095084, val loss 2.07481, mem 2.2 GiB @ 00 00:45:44.468, mean 00 00:00:00.036
step 73500: train loss 2.0001807, val loss 2.0707889, mem 2.2 GiB @ 00 00:46:04.034, mean 00 00:00:00.039
step 74000: train loss 1.999873, val loss 2.0661273, mem 2.2 GiB @ 00 00:46:21.921, mean 00 00:00:00.035
step 74500: train loss 1.9958498, val loss 2.0752115, mem 2.2 GiB @ 00 00:46:39.563, mean 00 00:00:00.035
step 75000: train loss 1.9925823, val loss 2.077614, mem 2.2 GiB @ 00 00:46:58.757, mean 00 00:00:00.038
step 75500: train loss 1.9997668, val loss 2.0509713, mem 2.2 GiB @ 00 00:47:18.211, mean 00 00:00:00.038
step 76000: train loss 2.0052767, val loss 2.0678136, mem 2.2 GiB @ 00 00:47:36.829, mean 00 00:00:00.037
step 76500: train loss 2.0013747, val loss 2.0568671, mem 2.2 GiB @ 00 00:47:55.354, mean 00 00:00:00.037
step 77000: train loss 1.9967788, val loss 2.0618782, mem 2.2 GiB @ 00 00:48:14.804, mean 00 00:00:00.038
step 77500: train loss 1.9915736, val loss 2.0647788, mem 2.2 GiB @ 00 00:48:33.543, mean 00 00:00:00.037
step 78000: train loss 1.9888786, val loss 2.060776, mem 2.2 GiB @ 00 00:48:51.636, mean 00 00:00:00.036
step 78500: train loss 1.9888809, val loss 2.0624423, mem 2.2 GiB @ 00 00:49:10.318, mean 00 00:00:00.037
step 79000: train loss 1.9989114, val loss 2.051822, mem 2.2 GiB @ 00 00:49:29.623, mean 00 00:00:00.038
step 79500: train loss 1.9952819, val loss 2.070324, mem 2.2 GiB @ 00 00:49:49.232, mean 00 00:00:00.039
step 80000: train loss 1.9863768, val loss 2.0552497, mem 2.2 GiB @ 00 00:50:07.822, mean 00 00:00:00.037

Device = Device(CUDA,-1)
13347905 parameters
learningRate = 1.1E-5
maxIterations = 450000
dropout = 0.2
step 0: train loss 4.262311, val loss 4.2712555, mem 1.4 GiB @ 00 00:00:00.000, mean 00 00:00:00.000
step 500: train loss 2.8142638, val loss 2.8265448, mem 1.5 GiB @ 00 00:00:18.527, mean 00 00:00:00.037
step 1000: train loss 5.4893785, val loss 2.6467226, mem 1.7 GiB @ 00 00:00:37.143, mean 00 00:00:00.037
step 1500: train loss 14.086155, val loss 3.560003, mem 1.7 GiB @ 00 00:00:55.620, mean 00 00:00:00.036 <-----
step 2000: train loss 7.454455, val loss 6.835584, mem 1.7 GiB @ 00 00:01:13.964, mean 00 00:00:00.036
step 2500: train loss 4.4331555, val loss 3.8669379, mem 1.7 GiB @ 00 00:01:32.571, mean 00 00:00:00.037

#~3

Device = Device(CUDA,-1)
13347905 parameters
learningRate = 1.1E-5
maxIterations = 450000
dropout = 0.2
step 0: train loss 4.2625003, val loss 4.271374, mem 1.4 GiB @ 00 00:00:00.000, mean 00 00:00:00.000
step 500: train loss 2.8663375, val loss 2.8833148, mem 1.5 GiB @ 00 00:00:18.359, mean 00 00:00:00.036
step 1000: train loss 3.4014244, val loss 2.661168, mem 1.5 GiB @ 00 00:00:36.738, mean 00 00:00:00.036
step 1500: train loss 2.5638556, val loss 2.5608368, mem 1.7 GiB @ 00 00:00:55.182, mean 00 00:00:00.036
step 2000: train loss 9.527954, val loss 3.4294536, mem 2.0 GiB @ 00 00:01:13.560, mean 00 00:00:00.036  <----
step 2500: train loss 3.5910563, val loss 3.6771824, mem 2.1 GiB @ 00 00:01:31.920, mean 00 00:00:00.036
step 3000: train loss 5.4479795, val loss 3.8090684, mem 2.1 GiB @ 00 00:01:50.253, mean 00 00:00:00.036
step 3500: train loss 13.253169, val loss 13.91807, mem 2.1 GiB @ 00 00:02:08.667, mean 00 00:00:00.036
step 4000: train loss 54.933147, val loss 56.528515, mem 2.1 GiB @ 00 00:02:27.119, mean 00 00:00:00.036
step 4500: train loss 5.340514, val loss 5.934756, mem 2.1 GiB @ 00 00:02:45.441, mean 00 00:00:00.036

#~2

Device = Device(CUDA,-1)
95937 parameters
learningRate = 1.1E-5
maxIterations = 450000
dropout = 0.2
step 0: train loss 4.2899413, val loss 4.2902613, mem 1.3 GiB @ 00 00:00:00.000, mean 00 00:00:00.000
step 500: train loss 3.8862078, val loss 3.891418, mem 1.6 GiB @ 00 00:00:18.112, mean 00 00:00:00.036
step 1000: train loss 3.728491, val loss 3.7295227, mem 1.6 GiB @ 00 00:00:36.133, mean 00 00:00:00.036
step 1500: train loss 3.6288931, val loss 3.626859, mem 1.6 GiB @ 00 00:00:55.364, mean 00 00:00:00.038
step 2000: train loss 3.5194693, val loss 3.5491323, mem 1.6 GiB @ 00 00:01:14.251, mean 00 00:00:00.037
step 2500: train loss 3.4499502, val loss 3.4583037, mem 1.6 GiB @ 00 00:01:32.808, mean 00 00:00:00.037

#~1

1585 parameters
learningRate = 1.1E-5
maxIterations = 450000
dropout = 0.2
step 0: train loss 4.423659, val loss 4.4229326, mem 801.4 MiB @ 00 00:00:00.000, mean 00 00:00:00.000
step 500: train loss 4.1128616, val loss 4.127762, mem 1000.7 MiB @ 00 00:00:09.007, mean 00 00:00:00.018
step 1000: train loss 3.8567889, val loss 3.8830547, mem 1001.2 MiB @ 00 00:00:18.007, mean 00 00:00:00.017
step 1500: train loss 3.705913, val loss 3.73203, mem 1009.9 MiB @ 00 00:00:27.009, mean 00 00:00:00.018
step 2000: train loss 3.6056633, val loss 3.633715, mem 1010.9 MiB @ 00 00:00:35.788, mean 00 00:00:00.017
step 2500: train loss 3.5196228, val loss 3.545664, mem 1015.3 MiB @ 00 00:00:44.607, mean 00 

*/

/*

V2
59521 parameters
Device = Device(CUDA,-1)
59521 parameters
learningRate = 1.1E-5
maxIterations = 450000
dropout = 0.2
Exception in thread "main" java.lang.RuntimeException: CUDA out of memory. Tried to allocate 16.00 MiB (GPU 0; 23.69 GiB total capacity; 22.41 GiB already allocated; 7.88 MiB free; 22.47 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Exception raised from malloc at /home/runner/work/javacpp-presets/javacpp-presets/pytorch/cppbuild/linux-x86_64-gpu/pytorch/c10/cuda/CUDACachingAllocator.cpp:913 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x6c (0x7f75100b1d8c in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10.so)
frame #1: <unknown function> + 0x2a124 (0x7f751069a124 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10_cuda.so)
frame #2: <unknown function> + 0x2a376 (0x7f751069a376 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10_cuda.so)
frame #3: <unknown function> + 0x2a7be (0x7f751069a7be in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10_cuda.so)
frame #4: <unknown function> + 0x124fcc0 (0x7f72ec595cc0 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #5: at::detail::empty_generic(c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType, c10::optional<c10::MemoryFormat>) + 0x28 (0x7f72ec58f808 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #6: at::detail::empty_cuda(c10::ArrayRef<long>, c10::ScalarType, c10::optional<c10::Device>, c10::optional<c10::MemoryFormat>) + 0x9a (0x7f72f751b56a in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #7: at::detail::empty_cuda(c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) + 0x45 (0x7f72f751b6f5 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #8: at::detail::empty_cuda(c10::ArrayRef<long>, c10::TensorOptions const&) + 0x123 (0x7f72f751b883 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #9: <unknown function> + 0x2c523b9 (0x7f72f93093b9 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #10: <unknown function> + 0x2d325c6 (0x7f72f93e95c6 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #11: at::meta::structured__softmax::meta(at::Tensor const&, long, bool) + 0x34d (0x7f72ecc43dbd in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #12: <unknown function> + 0x2c5b508 (0x7f72f9312508 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #13: <unknown function> + 0x2c5b5dc (0x7f72f93125dc in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cuda.so)
frame #14: at::_ops::_softmax::redispatch(c10::DispatchKeySet, at::Tensor const&, long, bool) + 0x95 (0x7f72ed6eaf65 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #15: <unknown function> + 0x3f165fa (0x7f72ef25c5fa in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #16: <unknown function> + 0x3f16c2f (0x7f72ef25cc2f in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #17: at::_ops::_softmax::call(at::Tensor const&, long, bool) + 0x173 (0x7f72ed764bb3 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #18: at::native::softmax(at::Tensor const&, long, c10::optional<c10::ScalarType>) + 0x94 (0x7f72ecc463f4 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #19: <unknown function> + 0x2a083eb (0x7f72edd4e3eb in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #20: at::_ops::softmax_int::call(at::Tensor const&, long, c10::optional<c10::ScalarType>) + 0x174 (0x7f72ed74eb84 in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #21: Java_org_bytedeco_pytorch_global_torch_softmax__Lorg_bytedeco_pytorch_Tensor_2JLorg_bytedeco_pytorch_ScalarTypeOptional_2 + 0xde (0x7f72ea4562ee in /home/vscode/.javacpp/cache/pytorch-2.0.1-1.5.10-20230923.234018-54-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libjnitorch.so)
frame #22: [0x7f75b89a053a]

        at org.bytedeco.pytorch.global.torch.softmax(Native Method)
        at torch.nn.functional.Activations.softmax(Activations.scala:91)
        at torch.nn.functional.Activations.softmax$(Activations.scala:28)
        at torch.nn.functional.package$.softmax(package.scala:33)
        at gpt.V2$Head.forward(v2.scala:602)
        at gpt.V2$Head.apply(v2.scala:609)
        at gpt.V2$Head.apply(v2.scala:609)
        at gpt.V2$MultiHeadAttention.$anonfun$5(v2.scala:648)
        at scala.collection.Iterator$$anon$9.next(Iterator.scala:584)
        at scala.collection.immutable.List.prependedAll(List.scala:156)
        at scala.collection.immutable.List$.from(List.scala:684)
        at scala.collection.immutable.List$.from(List.scala:681)
        at scala.collection.IterableFactory$Delegate.from(Factory.scala:288)
        at scala.collection.immutable.Iterable$.from(Iterable.scala:35)
        at scala.collection.immutable.Iterable$.from(Iterable.scala:32)
        at scala.collection.IterableOps.map(Iterable.scala:682)
        at scala.collection.IterableOps.map$(Iterable.scala:682)
        at torch.nn.modules.container.ModuleList.map(ModuleList.scala:48)
        at gpt.V2$MultiHeadAttention.forward(v2.scala:648)
        at gpt.V2$MultiHeadAttention.apply(v2.scala:651)
        at gpt.V2$Block.forward(v2.scala:737)
        at gpt.V2$Block.apply(v2.scala:741)
        at gpt.V2$Block.apply(v2.scala:741)
        at torch.nn.modules.container.Sequential.apply$$anonfun$1(Sequential.scala:33)
        at scala.collection.IterableOnceOps.foldLeft(IterableOnce.scala:643)
        at scala.collection.IterableOnceOps.foldLeft$(IterableOnce.scala:669)
        at scala.collection.AbstractIterable.foldLeft(Iterable.scala:933)
        at torch.nn.modules.container.Sequential.apply(Sequential.scala:33)
        at gpt.V2$GPTLanguageModel.forward(v2.scala:838)
        at gpt.V2$GPTLanguageModel.apply(v2.scala:875)
        at gpt.V2$.estimateLoss$$anonfun$1$$anonfun$1(v2.scala:537)
        at gpt.V2$.estimateLoss$$anonfun$1$$anonfun$adapted$1(v2.scala:538)
        at scala.collection.immutable.Range.foreach(Range.scala:190)
        at gpt.V2$.estimateLoss$$anonfun$1(v2.scala:538)
        at scala.runtime.function.JProcedure1.apply(JProcedure1.java:15)
        at scala.runtime.function.JProcedure1.apply(JProcedure1.java:10)
        at scala.collection.immutable.List.foreach(List.scala:333)
        at gpt.V2$.estimateLoss(v2.scala:539)
        at gpt.V2$.train$$anonfun$1$$anonfun$1(v2.scala:925)
        at scala.runtime.function.JProcedure1.apply(JProcedure1.java:15)
        at scala.runtime.function.JProcedure1.apply(JProcedure1.java:10)
        at scala.util.Using$.resource(Using.scala:261)
        at gpt.V2$.train$$anonfun$1(v2.scala:945)
        at scala.runtime.java8.JFunction1$mcVI$sp.apply(JFunction1$mcVI$sp.scala:18)
        at scala.collection.immutable.Range.foreach(Range.scala:190)
        at gpt.V2$.train(v2.scala:945)
        at gpt.V2$.main(v2.scala:983)
        at gpt.V2.main(v2.scala)
1 targets failed
examples.runMain subprocess failed
vscode ➜ /workspaces/storch (explore_1) $ 


*/
