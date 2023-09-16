package gpt

// cSpell: ignore gpt, hyperparameters, logits, softmax
// cSpell: ignore CUDA, torchvision
// cSpell: ignore dtype
// cSpell: ignore stoi, itos
// cSpell: ignore nn, probs
// cSpell: ignore xbow, xprev

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

// 
// import caseapp.*
// import caseapp.core.argparser.{ArgParser, SimpleArgParser}
// import caseapp.core.app.CommandsEntryPoint
// import com.sksamuel.scrimage.{ImmutableImage, ScaleMethod}
// import me.tongfei.progressbar.{ProgressBar, ProgressBarBuilder}
// import org.bytedeco.javacpp.PointerScope
// import org.bytedeco.pytorch.{InputArchive, OutputArchive}
// import os.Path
// import torch.*
// import torch.Device.{CPU, CUDA}
// import torch.optim.Adam
// import torchvision.models.resnet.{ResNet, ResNetVariant}
// 
// import java.nio.file.Paths
// import scala.collection.parallel.CollectionConverters.ImmutableSeqIsParallelizable
// import scala.concurrent.ExecutionContext.Implicits.global
// import scala.concurrent.duration.DurationInt
// import scala.concurrent.{Await, Future}
// import scala.util.{Random, Try, Using}


// hyperparameters
val batch_size = 16 // 16 // how many independent sequences will we process in parallel?
val block_size = 8 // 32 // what is the maximum context length for predictions?
val max_iters = 5000  // 3000
val eval_interval = 500  // 300
val learning_rate = 1e-3 // 1e-2
//val device = 'cuda' if torch.cuda.is_available() else 'cpu'
val device = if torch.cuda.isAvailable then CUDA else CPU
//println(s"Using device: $device")
val eval_iters = 200
val n_embed = 32 // 64
val head_size = 16
val n_head = 4
val n_layer = 4
val dropout = 0.0
// ------------


// batch_size = 16 # how many independent sequences will we process in parallel?
// block_size = 32 # what is the maximum context length for predictions?
// max_iters = 5000
// eval_interval = 100
// learning_rate = 1e-3
// device = 'cuda' if torch.cuda.is_available() else 'cpu'
// eval_iters = 200
// n_embd = 64
// n_head = 4
// n_layer = 4
// dropout = 0.0

type simpleIndex = Option[Long] | Long
type SI = simpleIndex

// TODO: use inline match or macros. Issue cannot match on tuples
// See https://stackoverflow.com/questions/75873631/tuples-in-scala-3-compiler-operations-for-typeclass-derivation 
// Easier to split the matches
extension (inline a:Slice | Option[Long] | Long = None)
  @targetName("index_colon")
  inline def `:`(inline b: Slice | Option[Long] | Long = None) = 
    (a, b) match
      case (i1: SI, i2: SI) => 
        Slice(start=i1, end=i2)
      case (i1:SI, Slice(s2,e2,st2)) =>
        (s2, e2, st2) match 
          case (None, _:Some[Long], None) => 
            Slice(i1, s2, e2)
          case (_:Some[Long], _:Some[Long], None) =>
            Slice(i1, s2, e2)
          case (_, _, _:Some[Long]) => throw RuntimeException("Step already exists.")
          case (_:Some[Long], _, _) => throw RuntimeException("Start index already exists.")
          case (None, None, None) => throw RuntimeException("No end/step exists.")
      case (Slice(s1,e1,st1), i2:SI) =>
        (s1, e1, st1) match 
          case (_, None, _) => Slice(s1, i2, st1)
          case (_, _, None) => Slice(s1, e1, i2)
          case (None, _, _) => throw RuntimeException("Start index is required.")
          case (_, _, _) => throw RuntimeException("End index and/or step already exist.")
      case ( Slice(s1,e1,st1), 
             Slice(s2,e2,st2)) => 
        throw RuntimeException("Combining indexes not supported.")
  // def unary_~ = Slice(start=0, end=a)

// TODO
val º = None
def `:` = Slice()
def `::` = Slice()
def `:::` = Slice()

object SliceTests:
  val nn = 99L

  def test1 =

    val s00 = `:`
    assert(s00 == Slice(None,None,None))

    val s01 = `::`
    assert(s01 == Slice(None,None,None))

    val s02 = None `:` None
    assert(s02 == Slice(None,None,None))

    val s03 = º`:`º 
    assert(s03 == Slice(None,None,None))

    val s04 = 0L`:`nn
    assert(s04 == Slice(Some(0L),Some(99L),None))

    val s05 = º`:`nn
    assert(s05 == Slice(None,Some(99L),None))

    // Operator is right associative
    val s06 = (º `:` nn) `:` 2L
    assert(s06 == Slice(None,Some(99L),Some(2L)))

    val s07 = nn`:`º
    assert(s07 == Slice(Some(99L),None,None))

    // Operator is right associative
    val s08_ = (º `:` 3L)
    val s08 = nn `:` s08_
    assert(s08 == Slice(Some(99L),None,Some(3L)))

    // Operator is right associative
    val s09 = nn `:` º `:` 3L
    assert(s09 == Slice(Some(99L),None,Some(3L)))

    // Operator is right associative
    val s10 = º `:` nn `:` 2L
    assert(s10 == Slice(None,Some(99L),Some(2L)))

    try
      val s8 = (0L `:` º) `:` (10L `:` 15L)
    catch 
      case e: java.lang.RuntimeException => ()
      case e: Throwable => throw e


// https://stackoverflow.com/questions/75873631/tuples-in-scala-3-compiler-operations-for-typeclass-derivation    
transparent inline def op(a:Option[Long], b:Option[Long]) =
  inline (a, b) match
    case (None, None) => None
    case (_:Some[Long], None) => a
    case (None, _:Some[Long]) => b
    case (Some(aa), Some(bb)) => 
      Some(aa+bb)

transparent inline def opA[T](a:Option[T]) =
  inline a match
    case None => None
    case Some[T](aa) => aa

transparent inline def opB[T](a:Option[T], b:Option[T])(using num: scala.math.Numeric[T]) =
  import num._
  inline a match
    case None => 
      inline b match
        case None => 0
        case Some(b_) => b_
    case Some(a_) => 
      inline b match
        case None => a_
        case Some(b_) => a_ + b_

/** 
 * ./mill examples.runMain gpt.BiGram
 * 
 * $ prime-select query
 * $ sudo prime-select intel
 * $ sudo prime-select nvidia
 * The "on-demand" mode requires explicit switching. See below.
 * $ sudo prime-select nvidia
 * 
 * Vulkan: 
 *   export __NV_PRIME_RENDER_OFFLOAD=1
 * OpenGL:
 *   export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
 * 
 * ~$ glxinfo | grep vendor
 * server glx vendor string: SGI
 * client glx vendor string: Mesa Project and SGI
 * OpenGL vendor string: Intel
 * 
 * ~$ export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
 * ~$ glxinfo | grep vendor
 * server glx vendor string: NVIDIA Corporation
 * client glx vendor string: NVIDIA Corporation
 * OpenGL vendor string: NVIDIA Corporation
 * 
 * VSCodeProjects/storch$ __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia glxgears -info
 * Running synchronized to the vertical refresh.  The framerate should be
 * approximately the same as the monitor refresh rate.
 * GL_RENDERER   = Quadro P1000/PCIe/SSE2
 * GL_VERSION    = 4.6.0 NVIDIA 535.86.10
 * GL_VENDOR     = NVIDIA Corporation
 * GL_EXTENSIONS = GL_AMD_multi_draw_indirect GL_AMD_seamless_cubemap_per_texture ...
 * 
 * @see https://www.youtube.com/watch?v=kCc8FmEb1nY
 * @see https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py
 */
object BiGramX:

  // TODO: cannot use inline match due to tuple apply method (not a value)
  //val t1 = op(Some(1), Some(2))
  val t2 = opA(Some(1))
  val t3 = opB(Some(1), Some(2))
  val t4 = opB(Some(1L), Some(2L))
  SliceTests.test1

  println("BiGram")
  println(s"Using device: $device")

  def len[T <: torch.DType](t: Tensor[T]): Int = 
    t.size.sum

  val DATA = "data"
  val INPUT = "input.txt"
  val URL_ = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

  torch.manualSeed(0)

  // wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
  val wd = os.pwd
  val dataDir = wd / DATA
  if ! os.exists(dataDir)
  then
    println(s"Creating folder: $dataDir")
    os.makeDir(dataDir) 

  val dataFile = dataDir / "input.txt"
  if ! os.exists(dataFile)
  then
    // Assume the URL is vali
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

  // val text = os.read(dataFile)
  // 1115394
  // println(text.length())
  val text = "a"*1115394
  val extras = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .!,;.:\n"
  
  
  // here are all the unique characters that occur in this text
  // TODO: val chars = SortedSet(text:_*)
  val chars = SortedSet((text + extras):_*)
  println(s"chars = ${chars.mkString(", ")}")
  val vocab_size = chars.size
  println(s"vocab_size = $vocab_size")

  // create a mapping from characters to integers
  val stoi = chars.zipWithIndex.map((ch, i) => ch -> i.toLong).toMap
  val itos = stoi.map((ch,i) => i -> ch)
  def encode(s: String) = s.map( c => stoi(c) )
  def decode(s: Seq[Long]) = s.map( i => itos(i) ).mkString

  println(s""""BiGram!" = "${decode(encode("BiGram!"))}"""")
  val aId = encode("a")
  println(s"'a' = $aId")

  // Train and test splits
  val data = torch.Tensor(encode(text)).long
  val n = (0.9 * len(data)).toLong // first 90% will be train, rest val
  val train_data = data(º`:`n)
  val val_data = data(n`:`º)

  torch.manualSeed(1337)

  // data loading
  def get_batch(split: String) = 
    // generate a small batch of data of inputs x and targets y
    val data = if split == "train" then train_data else val_data
    // could have used .long
    val ix = torch.randint(0, len(data) - block_size, Seq(batch_size)).to(dtype = int64)
    val stacks_x = ix.toSeq.map(i => data(i`:`i+block_size))
    val x = torch.stack(stacks_x)
    val stacks_y = ix.toSeq.map(i => data(i+1`:`i+block_size+1))
    val y = torch.stack(stacks_y)
    ( x.to(device), y.to(device) )

  val (xb, yb) = get_batch("train")
  println("inputs:")
  println(xb.shape)
  println(xb)
  println("targets:")
  println(yb.shape)
  println(yb)
  println("----")

  println("xb:")
  println(decode( xb(0).toSeq ))
  println(".")
  println(decode( xb(1).toSeq ))
  println("yb:")
  println(decode( yb(0).toSeq ))
  println(".")
  println(decode( yb(1).toSeq ))

  for b <- 0 until batch_size // batch dimension
  do
    for t <- 0 until block_size // time dimension
    do
      val context = xb(b, º`:`t+1)
      val target = yb(b,t)
      println(s"when input is ${context.toSeq.mkString("[",", ","]")} the target: ${target.item}")

  println(xb) // our input to the transformer
  

  torch.manualSeed(1337)
  
  trait BigramLanguageModel extends nn.Module:
    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None): (Tensor[Float32], Tensor[Float32])
    def generate(idx: Tensor[Int64], max_new_tokens: Int): Tensor[Int64]
    def apply(x: Tensor[Int64], y: Tensor[Int64]): (Tensor[Float32], Tensor[Float32])
    def apply(x: Tensor[Int64]): (Tensor[Float32], Tensor[Float32])
  // class BigramLanguageModel0(vocabSize: Int) extends BigramLanguageModel: 


  val input0 = torch.randn(Seq(3, 5), requiresGrad=true)
  val target0 = torch.randint(0, 5, Seq(3), dtype=torch.int64)
  val loss0 = F.crossEntropy(input0, target0)
  loss0.backward()
  println(loss0)

  // Example of target with class probabilities
  val input1 = torch.randn(Seq(3, 5), requiresGrad=true)
  val target1 = F.softmax( input=torch.randn(Seq(3, 5)), dim=1L)
  val loss1 = F.crossEntropy(input1, target1)
  loss1.backward()
  println(loss1)

  val target2 = torch.randn(Seq(3, 5)).softmax(dim=1L)
  val loss2 = F.crossEntropy(input1, target2)
  loss2.backward()
  

  // https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
  // TODO: @torch.no_grad()
  def estimate_loss(model: BigramLanguageModel) = 
    val out = scala.collection.mutable.Map[String, Float]()
    model.eval()
    for 
      split <- List("train", "val")
    do
      // println(s"Estimate '$split' loss")
      val losses: Tensor[Float32] = torch.zeros(eval_iters)
      for 
        k <- 0 until eval_iters
      do
        val (x, y) = get_batch(split)
        val (logits, loss) = model(x, y)
        // TODO: no assignment operator available
        losses(Seq(k)) = loss.item
      out(split) = losses.mean.item
    model.train()
    out



  // The mathematical trick in self-attention
  // toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
  torch.manualSeed(42)
  val a0 = torch.tril(torch.ones(Seq(3, 3)))
  // If we use only a0, we have the sums
  val a = a0 / torch.sum(a0, 1, keepdim=true)
  val b = torch.randint(0,10,Seq(3,2)).float
  val c = a `@` b
  print("a=")
  println(a)
  println("--")
  print("b=")
  println(b)
  println("--")
  print("c=")
  println(c)  

  // consider the following toy example:

  torch.manualSeed(1337)
  val (b0,t0,c0) = (4,8,2) // batch, time, channels
  val x0 = torch.randn(Seq(b0, t0, c0))
  println(x0.shape)

  // We want x[b,t] = mean_{i<=t} x[b,i]
  val xbow = torch.zeros(Seq(b0, t0, c0))
  for b <- 0 until b0
  do
    for t <- 0 until t0
    do
      val xprev = x0(b,º`:`t+1) // (t,C)
      xbow(Seq(b,t)) = torch.mean(xprev, dim=0)

  // version 2: using matrix multiply for a weighted aggregation
  val wei0 = torch.tril(torch.ones(Seq(t0, t0)))
  println(s"wei0.shape = ${wei0.shape}")
  val wei1 = wei0 / wei0.sum(1, keepdim=true)
  val xbow2 = wei1 `@` x0 // (T, T) @ (B, T, C) (PyTorch broadcast)---> (B, T, T) @ (B, T, C) ----> (B, T, C)
  println(torch.allclose(xbow, xbow2))

  // version 3: use Softmax
  val tril1 = torch.tril(torch.ones(Seq(t0, t0)))
  val zeros1 = torch.zeros(Seq(t0,t0))
  val mask2 = zeros1.maskedFill(tril1 == 0, Float.NegativeInfinity)
  val wei2 = F.softmax(mask2, dim= -1)
  val xbow3 = wei2 `@` x0
  println(torch.allclose(xbow, xbow3))




  // version 4: self-attention!
  torch.manualSeed(1337)
  val (b1, t1, c1) = (4,8,32) // batch, time, channels
  val x = torch.randn(Seq(b1,t1,c1))
    
  // let's see a single Head perform self-attention
  val head_size_1 = 16
  val key = nn.Linear(c1, head_size_1, bias=false)
  val query = nn.Linear(c1, head_size_1, bias=false)
  val value = nn.Linear(c1, head_size_1, bias=false)
  val k = key(x)   // (B, T, 16)
  val q = query(x) // (B, T, 16)
  // TODO. https://math.stackexchange.com/questions/63074/is-there-a-3-dimensional-matrix-by-matrix-product
  // https://www.geeksforgeeks.org/numpy-3d-matrix-multiplication/
  val qk4 =  q `@` k.transpose(-2, -1) // (B, T, 16) @ (B, 16, T) ---> (B, T, T)
    
  val tril4 = torch.tril(torch.ones(Seq(t1, t1)))
  // val wei3 = torch.zeros((T,T))
  val mask4 = qk4.maskedFill(tril4 == 0, Float.NegativeInfinity)
  val wei4 = F.softmax(mask4, dim= -1)
    
  val v4 = value(x)
  val out4 = wei4 `@` v4
  // val out4 = wei4 `@` x

  // (4,8,16)
  println(out4.shape)

  println(wei4(0))
  /* Just confirming we have the same output
  tensor([
        [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1574, 0.8426, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2088, 0.1646, 0.6266, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5792, 0.1187, 0.1889, 0.1131, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0294, 0.1052, 0.0469, 0.0276, 0.7909, 0.0000, 0.0000, 0.0000],
        [0.0176, 0.2689, 0.0215, 0.0089, 0.6812, 0.0019, 0.0000, 0.0000],
        [0.1691, 0.4066, 0.0438, 0.0416, 0.1048, 0.2012, 0.0329, 0.0000],
        [0.0210, 0.0843, 0.0555, 0.2297, 0.0573, 0.0709, 0.2423, 0.2391]],
       grad_fn=<SelectBackward0>)
  */  
  val wei4_0 = wei4(0,0).toArray
  assert(wei4_0.sameElements(Array(1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000)))
  val wei4_1 = wei4(0,1)
  assert(torch.allclose(wei4_1, Tensor(Array(0.1574f, 0.8426f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f)), atol=1e-04))
  val wei4_2 = wei4(0,2)
  assert(torch.allclose(wei4_2, Tensor(Array(0.2088f, 0.1646f, 0.6266f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f)), atol=1e-04))
  val wei4_3 = wei4(0,3)
  assert(torch.allclose(wei4_3, Tensor(Array(0.5792f, 0.1187f, 0.1889f, 0.1131f, 0.0000f, 0.0000f, 0.0000f, 0.0000f)), atol=1e-04))
  val wei4_4 = wei4(0,4)
  assert(torch.allclose(wei4_4, Tensor(Array(0.0294f, 0.1052f, 0.0469f, 0.0276f, 0.7909f, 0.0000f, 0.0000f, 0.0000f)), atol=1e-04))
  val wei4_5 = wei4(0,5)
  assert(torch.allclose(wei4_5, Tensor(Array(0.0176f, 0.2689f, 0.0215f, 0.0089f, 0.6812f, 0.0019f, 0.0000f, 0.0000f)), atol=1e-04))
  val wei4_6 = wei4(0,6)
  assert(torch.allclose(wei4_6, Tensor(Array(0.1691f, 0.4066f, 0.0438f, 0.0416f, 0.1048f, 0.2012f, 0.0329f, 0.0000f)), atol=1e-04))
  val wei4_7 = wei4(0,7)
  assert(torch.allclose(wei4_7, Tensor(Array(0.0210f, 0.0843f, 0.0555f, 0.2297f, 0.0573f, 0.0709f, 0.2423f, 0.2391f)), atol=1e-04))

  // note 6: "scaled" self-attention. why divide by sqrt(head_size)
  // Scaled dot-product attention
  // "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are 
  // unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. 
  // Illustration below
  val head_size_5 = 16
  val (b5, t5, c5) = (4,8,32) // batch, time, channels
  val k5 = torch.randn(Seq(b5,t5,head_size_1))
  val q5 = torch.randn(Seq(b5,t5,head_size_1))
  //val wei5 = q5 `@` k5.transpose(-2, -1) * head_size**-0.5
  // Both k and q are a unit Gaussian (mean 0 and standard deviation 1) 
  println(s"${k5.variance}")
  println(s"${q5.variance}")
  // If we multiple these two tensors, the standard deviation will be in the order of the head size
  val wei5_0 = q5 `@` k5.transpose(-2, -1)
  println(s"${wei5_0.variance}")
  // So we can divide by an order of the head size to get same standard deviation of 1
  val wei5_1 = q5 `@` k5.transpose(-2, -1) * Math.pow(head_size, -0.5)
  println(s"${wei5_1.variance}")
  // Why is this important? Note that wei feeds into softmax that is then multiplied by the self-attention head's 
  // value `v`. It is important wei be fairly diffuse. If the the wei vectors become very peaky (takes on very 
  // positive and very negative value), softmax will convert these values and converge to a one-hot-encoding. 
  // TODO: does not compile
  val z = Tensor(Seq(0.1, -0.2, 0.3, -0.2, 0.5))
  // val one_hot_100 = torch.softmax(Tensor(Seq(0.1, -0.2, 0.3, -0.2, 0.5)), dim = -1L)
  val one_hot_0 = F.softmax(z, dim = -1L)
  println(one_hot_0)
  // If we "sharpen" these values
  val one_hot_1 = F.softmax(z*8, dim = -1) 
  // it gets too peaky and converges to one-hot (a single value will dominate all others)
  println(one_hot_1)
  // this means that in such a case, basically information will be aggregated from a single node. This can be 
  // a problem especially during the initial training


  /**
   * one head of self-attention
   */
  class Head1[D <: FloatNN: Default](
          n_embed: Int, 
          head_size: Int, 
          block_size: Int
          //drop: Double
          ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    val key = nn.Linear[D](n_embed, head_size, bias=false)
    val query = nn.Linear[D](n_embed, head_size, bias=false)
    val value = nn.Linear[D](n_embed, head_size, bias=false)
    val ones = torch.ones[D](Seq(block_size, block_size), dtype=key.paramType)
    val tril = registerBuffer(torch.tril(ones), "tril")

    // val dropout = nn.Dropout(drop)

    def forward(x: Tensor[D]): Tensor[D] =
      // input of size (batch, time-step, channels) = (B,T,C)
      // output of size (batch, time-step, head size) = (B,T,H)
      // batch, number of time steps, channels
      val Seq(b,t,c) = x.shape
      // fails on generate ?
      // assert(block_size == t, "Block size must be equal to time step")

      // key = Linear(inFeatures=C, outFeatures=T, bias=false)
      val k = key(x)   // (B,T,C) @ (C,H) -> (B,T,H)
      val q = query(x) // (B,T,H)
      // compute attention scores ("affinities")
      // c should be the head size
      val hs = k.size.last
      assert(head_size == hs, "Head size does not match k")
      val qk = q `@` k.transpose(-2, -1) * Tensor(hs).pow(-0.5).to(dtype=q.dtype)  // (B, T, H) @ (B, H, T) -> (B, T, T)
      // val mask = qk.maskedFill(tril == 0, Float.NegativeInfinity) // (B, T, T)
      val mask = qk.maskedFill(tril((º`:`n), (º`:`n)) == 0, Float.NegativeInfinity).to(dtype=q.dtype) // (B, T, T)
      // val softmax = F.softmax(mask, dim= -1) // (B, T, T)
      // val wei = dropout(softmax)
      val wei = F.softmax(mask, dim= -1) // (B, T, T)
      // perform the weighted aggregation of the values
      val v = value(x) // (B,T,H)
      val out = wei `@` v // (B, T, T) @ (B, T, H) -> (B, T, H)
      out

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

  /**
   * 
   * @param vocabSize - number o tokens
   * @param blockSize - number of tokens taken from the text input to apply to the NN
   * @param nEm
   */
  class BigramLanguageModel3(vocabSize: Int, blockSize:Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = nn.Embedding(vocabSize, nEmbed)
    val position_embedding_table = nn.Embedding(blockSize, nEmbed)
    // head_size = n_embed, block_size = T
    val sa_head = Head1(n_embed = nEmbed, head_size = head_size, block_size = blockSize) //, drop = 0.5)
    // val lm_head = nn.Linear(nEmbed, vocabSize)
    // Just to test Head1 use head_size instead on embed_size
    val lm_head = nn.Linear(head_size, vocabSize)

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =
      val Seq(b,t) = idx.shape

      // idx and targets are both (B,T) tensor of integers
      val token_embed = token_embedding_table( idx ) // (B,T,C) where C is nEmbed
      // positions of tokens
      val pos = torch.arange(0L,t, device=device) // (T) were T is the block size
      val pos_embed = position_embedding_table( pos ) // (T,C)
      // Add the position embeddings
      val x0 = token_embed + pos_embed // (B,T,C)
      val x1 = sa_head(x0) // apply one head of self-attention (B,T,C) - for test use (B,T,H)
      // val x1 = blocks(x0) // (B,T,C)
      // val x2 = ln_f(x2) // (B,T,C)
      val logits = lm_head( x1 ) //  (B,T,H) @ (H,vocabSize) -> (B,T,vocabSize)

      if targets.isEmpty
      then
        val zero = torch.Tensor(0.0f) 
        (logits, zero)
      else
        val Seq(b,t,c) = logits.shape
        val logitsV = logits.view(b*t, c)  // batch size x number of classes
        val targetsV = targets.get.view(b*t) 
        // println(s"idx.shape = ${idx.shape}")
        // println(s"targets.get.shape = ${targets.get.shape}")
        println(s"logits.shape = ${logits.shape}")
        // println(s"idx(0) = ${idx(0).toSeq}")
        // println(s"targets(0) = ${targets.get(0).toSeq}")
        println(s"logits(0) = ${logits(0)}")
        val l = logits(0)
        // println(s"l(0) = ${l}")
        // val argmax = l.indices.maxBy(l)
        // println(s"max = $argmax")
        // println(s"max = $argmax -> ${decode(Seq(argmax))}")
        val loss = F.crossEntropy(logitsV, targetsV)
        // println(s"logitsV.shape = ${logitsV.shape}")
        // println(s"targetsV.shape = ${targetsV.shape}")
        // println(s"loss.shape = ${loss.shape}")
        // println(s"loss = ${loss}")
        /*
        Estimate loss
        idx.shape = ArraySeq(16, 8)
        targets.get.shape = ArraySeq(16, 8)
        logits.shape = ArraySeq(16, 8, 65)
        idx(0) = ArraySeq(47, 60, 43, 1, 59, 52, 58, 47)
        targets(0) = ArraySeq(60, 43, 1, 59, 52, 58, 47, 50)
        targets(0) = ArraySeq(60, 43, 1, 59, 52, 58, 47, 50)
        logits(0) = tensor dtype=float32, shape=[8, 65], device=CPU 
        [[-0.0152, 0.0327, 0.8416, ..., 0.7316, 0.0207, -0.0154],
        [0.2119, -0.1712, 0.7282, ..., 0.5682, -0.1327, 0.2052],
        [0.0563, -0.1682, 0.4429, ..., 0.2863, -0.0212, -0.0481],
        ...,
        [0.2163, -0.0836, 0.4823, ..., 0.3438, -0.1904, 0.0936],
        [0.0975, 0.2394, 0.3527, ..., 0.2178, -0.1093, -0.2017],
        [0.0185, 0.2494, 0.4354, ..., 0.2713, -0.0690, -0.4120]]
        logitsV.shape = ArraySeq(128, 65)
        targetsV.shape = ArraySeq(128)
        loss.shape = ArraySeq()
        loss = tensor dtype=float32, shape=[], device=CPU 
        4.2519
        */
        (logitsV, loss)


    def generate(idx: Tensor[Int64], max_new_tokens: Int) =
      var idx_ = idx.copy_(idx)
      // idx is (B, T) array of indices in the current context
      for i <- 0 until max_new_tokens
      do
        // crop idx to the last block_size tokens
        val idx_cond = idx_(`:`, (-blockSize`:`º))
        // println(s"idx_ ${idx_.shape} -> idx_cond ${idx_cond.shape}")
        // println(s"idx_ |${idx_.toSeq.takeRight(blockSize)}| -> idx_cond |${idx_cond.toSeq}|")
        // get the predictions
        val (logits_t, loss) = apply(idx_cond)
        // focus only on the last time step
        val logits = logits_t(`:`, -1, `:`) // becomes (B, C)
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


  end BigramLanguageModel3


  // Create a model
  val m4 = BigramLanguageModel3(vocab_size, block_size, n_embed)
  m4.train()
  // create a PyTorch optimizer
  val optimizer4 = torch.optim.AdamW(m4.parameters, lr=1e-3)

  for iter <- 0 until 5000 //max_iters
  do
    // every once in a while evaluate the loss on train and val sets
    if (iter % eval_interval == 0) || (iter == max_iters - 1)
    then
      //println(s"Estimate loss")
      val losses = estimate_loss(m4)
      println(s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}")
      //print(s"step ${iter}: train loss ${losses("train"):.4f}, val loss ${losses("val"):.4f}")

    // sample a batch of data
    val (xb, yb) = get_batch("train")

    // evaluate the loss
    val (logits, loss) = m4(xb, yb)
    optimizer4.zeroGrad(setToNone=true)
    loss.backward()
    optimizer4.step()

  println("Debug m4.generate() !") 
  // val next6 = m4.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64), max_new_tokens=500)(0)
  val next6 = m4.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64), max_new_tokens=500)(0)
  val decoded6 = decode(next6.toSeq)
  println(s"decode 6:'$decoded6'")

  
  def main(args: Array[String]): Unit =
    ()

end BiGramX
