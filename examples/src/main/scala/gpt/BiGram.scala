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
import org.bytedeco.javacpp.Pointer

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
// https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html
// https://stackoverflow.com/questions/63751883/using-gpu-inside-docker-container-cuda-version-n-a-and-torch-cuda-is-availabl
val device = if torch.cuda.isAvailable then CUDA else CPU
//println(s"Using device: $device")
val eval_iters = 200
val n_embed = 32 // 64
val head_size = 16
val n_head = 4
val n_layer = 4
val dropout = 0.2
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

type simpleIndex = Option[Int] | Int
type SI = simpleIndex

// TODO indexingSlice now has operator. See how to use
// TODO: use inline match or macros. Issue cannot match on tuples
// See https://stackoverflow.com/questions/75873631/tuples-in-scala-3-compiler-operations-for-typeclass-derivation 
// Easier to split the matches
extension (inline a:Slice | Option[Int] | Int = None)
  @targetName("index_colon")
  inline def `:`(inline b: Slice | Option[Int] | Int = None) = 
    (a, b) match
      case (i1: SI, i2: SI) => 
        Slice(start=i1, end=i2)
      case (i1:SI, Slice(s2,e2,st2)) =>
        (s2, e2, st2) match 
          case (None, _:Some[Int], None) => 
            Slice(i1, s2, e2)
          case (_:Some[Int], _:Some[Int], None) =>
            Slice(i1, s2, e2)
          case (_, _, _:Some[Int]) => throw RuntimeException("Step already exists.")
          case (_:Some[Int], _, _) => throw RuntimeException("Start index already exists.")
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
  val nn = 99

  def test1 =

    val s00 = `:`
    assert(s00 == Slice(None,None,None))

    val s01 = `::`
    assert(s01 == Slice(None,None,None))

    val s02 = None `:` None
    assert(s02 == Slice(None,None,None))

    val s03 = º`:`º 
    assert(s03 == Slice(None,None,None))

    val s04 = 0`:`nn
    assert(s04 == Slice(Some(0),Some(99),None))

    val s05 = º`:`nn
    assert(s05 == Slice(None,Some(99),None))

    // Operator is right associative
    val s06 = (º `:` nn) `:` 2
    assert(s06 == Slice(None,Some(99),Some(2)))

    val s07 = nn`:`º
    assert(s07 == Slice(Some(99),None,None))

    // Operator is right associative
    val s08_ = (º `:` 3)
    val s08 = nn `:` s08_
    assert(s08 == Slice(Some(99),None,Some(3)))

    // Operator is right associative
    val s09 = nn `:` º `:` 3
    assert(s09 == Slice(Some(99),None,Some(3)))

    // Operator is right associative
    val s10 = º `:` nn `:` 2
    assert(s10 == Slice(None,Some(99),Some(2)))

    try
      val s8 = (0 `:` º) `:` (10 `:` 15)
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



def moduleName(m: Module): String =
    //m.getClass().getSimpleName()
    m.toString()

def moduleClass(m: Module): String =
    m.getClass().getSimpleName()

/**
  * Collects information of a module and returns this as a string. Complex modules
  * are shown hierarchically. Information includes the modules `toString` output
  * that usually holds the variable name and class parameter values. We also add
  * the number of tensor parameter amd their value in the leaf modules. For the 
  * other modules the sum of the number of tensor parameters are shown. 
  * 
  * Use this output to "debug" your networks
  *
  * @param m
  * @return string
  */
def doModuleInfoString(m:Module, indent: Int): String =
  val parametersCount = m.parameters.size
  if m.modules.isEmpty 
  then 
    val parametersSize = m.parameters.map(_.numel).mkString("<", ",", ">")
    val thisModule = s"${moduleName(m)}: #$parametersCount $parametersSize "
    thisModule
  else
    val parametersSize = m.parameters.map(_.numel).sum
    val thisModule = s"${moduleName(m)}: #$parametersCount $parametersSize "
    thisModule + m.namedChildren
      .map((name, module) => s"${" " * (indent + 2)}$name: " + doModuleInfoString(module, indent + 2))
      .mkString("(\n", "\n", s"\n${" " * indent})")

/**
  * Collects information of a module and returns this as a string. Complex modules
  * are shown hierarchically. Information includes the modules `toString` output
  * that usually holds the variable name and class parameter values. We also add
  * the number of tensor parameter amd their value in the leaf modules. For the 
  * other modules the sum of the number of tensor parameters are shown. 
  * 
  * Use this output to "debug" your networks
  *
  * @param m
  * @return string
  */
def moduleInfoString(m:Module): String =
  doModuleInfoString(m, 0)        

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
object BiGram:

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

  val text = os.read(dataFile)
  // 1115394
  // println(text.length())
  // val text = "a"*1115394
  // val extras = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .!,;.:\n"
  
  // here are all the unique characters that occur in this text
  val chars = SortedSet(text:_*)
  // val chars = SortedSet((text + extras):_*)
  println(s"chars = ${chars.mkString(", ")}")
  val vocab_size = chars.size
  println(s"vocab_size = $vocab_size")

  // create a mapping from characters to integers
  val stoi = chars.zipWithIndex.map((ch, i) => ch -> i.toLong).toMap
  val itos = stoi.map((ch,i) => i -> ch)
  def encode(s: String) = s.map( c => stoi(c) )
  def decode(s: Seq[Long]) = s.map( i => itos(i) ).mkString

  println(s""""BiGram!" = "${decode(encode("BiGram!"))}"""")

  // Train and test splits
  val data = torch.Tensor(encode(text)).long
  val n = (0.9 * len(data)).toInt // first 90% will be train, rest val
  val train_data = data(º`:`n)
  val val_data = data(n`:`º)

  torch.manualSeed(1337)

  // data loading
  def get_batch(split: String) = 
    // generate a small batch of data of inputs x and targets y
    val data = if split == "train" then train_data else val_data
    // could have used .long
    val ix = torch.randint(0, len(data) - block_size, Seq(batch_size)).to(dtype = int32) //.to(dtype = int64)
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

  println("xb (default CUDA if it exists):")
  println(xb) // our input to the transformer
  println("yb (default CUDA if it exists):")
  println(yb) // our input to the transformer
  println(s"xb (set $device):")
  xb.to(device)
  println(xb) // our input to the transformer
  

  torch.manualSeed(1337)
  
  trait BigramLanguageModel extends nn.Module:
    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None): (Tensor[Float32], Tensor[Float32])
    def generate(idx: Tensor[Int64], max_new_tokens: Int): Tensor[Int64]
    def apply(x: Tensor[Int64], y: Tensor[Int64]): (Tensor[Float32], Tensor[Float32])
    def apply(x: Tensor[Int64]): (Tensor[Float32], Tensor[Float32])
  // class BigramLanguageModel0(vocabSize: Int) extends BigramLanguageModel: 

  //class BigramLanguageModel0(vocabSize: Int) extends nn.Module: 
  class BigramLanguageModel0(vocabSize: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, vocabSize) )
                                                      .to( device )

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =

      // idx and targets are both (B,T) tensor of integers
      val logits = token_embedding_table( idx ) // (B,T,C)

      if targets.isEmpty
      then
        val zero = torch.Tensor(0.0f, device=device) 
        (logits, zero)
      else
        val Seq(b,t,c) = logits.shape
        val logitsV = logits.view(b*t, c)  // batch size x number of classes
        val targetsV = targets.get.view(b*t) 
        val loss = F.crossEntropy(logitsV, targetsV)
        (logitsV, loss)


    def generate(idx: Tensor[Int64], max_new_tokens: Int) =
      var idx_ = idx.copy_(idx)
      // idx is (B, T) array of indices in the current context
      for i <- 0 until max_new_tokens
      do
        // get the predictions
        val (logits_t, loss) = apply(idx_)
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


  end BigramLanguageModel0

  val input0 = torch.randn(Seq(3, 5), requiresGrad=true)
  val target0 = torch.randint(0, 5, Seq(3), dtype=torch.int64)
  val loss0 = F.crossEntropy(input0, target0)
  loss0.backward()
  println(s"loss0 = $loss0")

  // Example of target with class probabilities
  val input1 = torch.randn(Seq(3, 5), requiresGrad=true)
  val target1 = F.softmax( input=torch.randn(Seq(3, 5)), dim=1L)
  val loss1 = F.crossEntropy(input1, target1)
  loss1.backward()
  println(s"loss1 = $loss1")

  val target2 = torch.randn(Seq(3, 5)).softmax(dim=1L)
  val loss2 = F.crossEntropy(input1, target2)
  loss2.backward()
  println(s"loss2 = $loss2")
  
  val m0 = BigramLanguageModel0(vocab_size)
  val (logits3, loss3) = m0(xb, yb)
  println(s"batch_size * block_size = ${batch_size * block_size}")
  println(s"logits.shape = ${logits3.shape}")
  println(s"loss m0 = ${loss3.item}")    
  
  val next1 = m0.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device), max_new_tokens=100)(0)
  val decoded1 = decode(next1.toSeq)
  println(s"decode:'$decoded1'")


  // create a PyTorch optimizer
  val optimizer0 = torch.optim.AdamW(m0.parameters, lr=1e-3)

  //val batch_size = 32
  var loss4: Tensor[Float32] = _
  // loss=4.5633 -> 4.5606503 for 100 steps
  // loss=4.5633 -> 4.4979854 for 100000 steps
  for steps <- 0 until 10 // increase number of steps for good results... 
  do      
    // sample a batch of data
    val (xb, yb) = get_batch("train")

    // evaluate the loss
    val (logits, loss) = m0(xb, yb)
    loss4 = loss
    optimizer0.zeroGrad(setToNone=true)
    loss.backward()
    optimizer0.step()

  println(loss4.item)

  val next2 = m0.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  val decoded2 = decode(next2.toSeq)
  println(s"decode 2:'$decoded2'")


  // https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
  // TODO: @torch.no_grad()
  def estimate_loss(model: BigramLanguageModel) = 
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
        val (x, y) = get_batch(split)
        val (logits, loss) = model(x, y)
        // TODO: no assignment operator available
        losses(Seq(k)) = loss.item
      out(split) = losses.mean.item
    model.train()
    out

  // Create a model
  val m1 = BigramLanguageModel0(vocab_size)
  m1.to(device)
  m1.train()
  // create a PyTorch optimizer
  val optimizer1 = torch.optim.AdamW(m1.parameters, lr=1e-3)

  for iter <- 0 until 10 //max_iters
  do
    // every once in a while evaluate the loss on train and val sets
    if (iter % eval_interval == 0) || (iter == max_iters - 1)
    then
      val losses = estimate_loss(m1)
      println(s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}")
      //print(s"step ${iter}: train loss ${losses("train"):.4f}, val loss ${losses("val"):.4f}")

    // sample a batch of data
    val (xb, yb) = get_batch("train")

    // evaluate the loss
    val (logits, loss) = m1(xb, yb)
    optimizer1.zeroGrad(setToNone=true)
    loss.backward()
    optimizer1.step()

  val next3 = m1.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  val decoded3 = decode(next3.toSeq)
  println(s"decode 3:'$decoded3'")




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



  // Changes to the BiGram
  // No need to pass `vocabSize` explicitly, but we keep this as is for flexibility
  // Using an intermediate phase 
  //  - don't go to the directly to the embeddings for the logits
  //  - instead go through an intermediate phase
  //  - n_embed short for number of imbedding dimensions (make it global with value 32)
  // Embedding now give use "token embeddings" (`tokenEmbed`)
  //  - To go from `tokenEmbed` to `logits` we will use a linear layer called `lm_head`
  //  - `lm_head` is short for language model head
  //  - It goes from nEmbed size to vocabSize
  // We now one spurious intermediate layer, so inference should execute as we have done above
  class BigramLanguageModel1(vocabSize: Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) ).to(device = device)
    val lm_head = register( nn.Linear(nEmbed, vocabSize) ).to(device = device)

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =

      // idx and targets are both (B,T) tensor of integers
      val token_embed = token_embedding_table( idx ) // (B,T,C) where C is nEmbed
      val logits = lm_head( token_embed ) // (B,T,vocabSize)

      if targets.isEmpty
      then
        val zero = torch.Tensor(0.0f) 
        (logits, zero)
      else
        val Seq(b,t,c) = logits.shape
        val logitsV = logits.view(b*t, c)  // batch size x number of classes
        val targetsV = targets.get.view(b*t) 
        val loss = F.crossEntropy(logitsV, targetsV)
        (logitsV, loss)


    def generate(idx: Tensor[Int64], max_new_tokens: Int) =
      var idx_ = idx.copy_(idx)
      // idx is (B, T) array of indices in the current context
      for i <- 0 until max_new_tokens
      do
        // get the predictions
        val (logits_t, loss) = apply(idx_)
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


  end BigramLanguageModel1

  // Create a model
  println("Token embedding: BigramLanguageModel1")
  val m2 = BigramLanguageModel1(vocab_size, n_embed)
  m2.to(device)
  m2.train()
  // create a PyTorch optimizer
  val optimizer2 = torch.optim.AdamW(m2.parameters, lr=1e-3)
  println(totalNuParameters(m2))
  println(moduleInfoString(m2))

  for iter <- 0 until 10 //max_iters
  do
    // every once in a while evaluate the loss on train and val sets
    if (iter % eval_interval == 0) || (iter == max_iters - 1)
    then
      val losses = estimate_loss(m2)
      println(s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}")
      //print(s"step ${iter}: train loss ${losses("train"):.4f}, val loss ${losses("val"):.4f}")

    // sample a batch of data
    val (xb, yb) = get_batch("train")

    // evaluate the loss
    val (logits, loss) = m2(xb, yb)
    optimizer2.zeroGrad(setToNone=true)
    loss.backward()
    optimizer2.step()

  val next4 = m2.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  val decoded4 = decode(next4.toSeq)
  println(s"decode 4:'$decoded4'")

  // So far we have taken the indices and encoded them based on the identities of the tokens
  // We now also encode the position of the tokens and not just their identity
  // So now we add a second position embedding table which is also an `Embedding` from `block_size` to `nEmbed`
  // - position_embedding_table is the new position embedding parameters
  // - pos is the positions from 0 to T-1
  // - The position embedding returns a vector (embedding) for each position
  // - Broadcasting now allows us to add the same position embedding to each batch B
  // - we add an `x` that hold not only the identity embeddings for each token, but also 
  // the position embeddings of each of these tokens
  // NOTE: Karpathy states that the `x` is "translation invariant", so this will not help (1:01:30)
  // Note that even though `x`has position encodings, we do not have the positions of the 
  // tokens from `idx`. We use this when we add the self-attention head
  class BigramLanguageModel2(vocabSize: Int, blockSize:Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    val lm_head = register( nn.Linear(nEmbed, vocabSize) )

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =
      val Seq(b,t) = idx.shape

      // idx and targets are both (B,T) tensor of integers
      // idx is (B,T)
      val token_embed = token_embedding_table( idx ) // (B,T,C) where C is nEmbed
      // positions of tokens
      val pos = torch.arange(0L,t, device=device) // (T) were T is the block size?
      //val pos = torch.arange(0L,blockSize, device=device) // (T) were T is the block size?
      val pos_embed = position_embedding_table( pos ) // (T,C)
      // Add the position embeddings
      val x = token_embed + pos_embed // (B,T,C)
      val logits = lm_head( token_embed ) // (B,T,vocabSize)

      if targets.isEmpty
      then
        val zero = torch.Tensor(0.0f) 
        (logits, zero)
      else
        val Seq(b,t,c) = logits.shape
        val logitsV = logits.view(b*t, c)  // batch size x number of classes
        val targetsV = targets.get.view(b*t) 
        val loss = F.crossEntropy(logitsV, targetsV)
        (logitsV, loss)


    def generate(idx: Tensor[Int64], max_new_tokens: Int) =
      var idx_ = idx.copy_(idx)
      // idx is (B, T) array of indices in the current context
      for i <- 0 until max_new_tokens
      do
        // get the predictions
        // original val (logits_t, loss) = apply(idx_)
        val idx_cond = idx_(`:`, (-blockSize`:`º))
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


  end BigramLanguageModel2


  // Create a model
  println("Token + positional embedding: BigramLanguageModel2")
  val m3 = BigramLanguageModel2(vocab_size, block_size, n_embed)
  m3.to(device)
  m3.train()
  // create a PyTorch optimizer
  val optimizer3 = torch.optim.AdamW(m3.parameters, lr=1e-3)
  println(totalNuParameters(m3))
  println(moduleInfoString(m3))

  // TODO: max_iters
  for iter <- 0 until 0 //max_iters
  do
    // every once in a while evaluate the loss on train and val sets
    if (iter % eval_interval == 0) || (iter == max_iters - 1)
    then
      val losses = estimate_loss(m3)
      println(s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}")
      //print(s"step ${iter}: train loss ${losses("train"):.4f}, val loss ${losses("val"):.4f}")

    // sample a batch of data
    val (xb, yb) = get_batch("train")

    // evaluate the loss
    val (logits, loss) = m3(xb, yb)
    optimizer3.zeroGrad(setToNone=true)
    loss.backward()
    optimizer3.step()

  val next5 = m3.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  val decoded5 = decode(next5.toSeq)
  println(s"decode 5:'$decoded5'")
  

  // version 4: self-attention!
  torch.manualSeed(1337)
  val (b1, t1, c1) = (4,8,32) // batch, time, channels
  val x = torch.randn(Seq(b1,t1,c1))
    
  // let's see a single Head perform self-attention
  val head_size_1 = 16
  val key = nn.Linear(c1, head_size_1, hasBias=false)
  val query = nn.Linear(c1, head_size_1, hasBias=false)
  val value = nn.Linear(c1, head_size_1, hasBias=false)
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
  class Head_1[D <: FloatNN: Default](
          n_embed: Int, 
          head_size: Int, 
          block_size: Int
          ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    val key = register( nn.Linear[D](n_embed, head_size, hasBias=false) )
    val query = register( nn.Linear[D](n_embed, head_size, hasBias=false) )
    val value = register( nn.Linear[D](n_embed, head_size, hasBias=false) )
    val ones = torch.ones[D](Seq(block_size, block_size), dtype=key.paramType)
    val tril = registerBuffer(torch.tril(ones), "tril")

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

    override def toString(): String = s"${getClass.getSimpleName()}(n_embed=$n_embed, head_size=$head_size, block_size=$block_size)"


  /**
   * 
   * @param vocabSize - number o tokens
   * @param blockSize - number of tokens taken from the text input to apply to the NN
   * @param nEmbed - number of values use for latent represent 
   */
  class BigramLanguageModel3(vocabSize: Int, blockSize:Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    // head_size = n_embed, block_size = T
    val sa_head = register( Head_1(n_embed = nEmbed, head_size = head_size, block_size = blockSize) ) //, drop = 0.5)
    // val lm_head = nn.Linear(nEmbed, vocabSize)
    // Just to test Head_1 use head_size instead on embed_size
    val lm_head = register( nn.Linear(head_size, vocabSize) )

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
      val x1 = sa_head(x0) // apply one head of self-attention (B,T,C) - for test use (B,T,H)
      // val x1 = blocks(x0) // (B,T,C)
      // val x2 = ln_f(x2) // (B,T,C)
      // mat1 and mat2 shapes cannot be multiplied (512x16 and 64x65)
      // Linear(inFeatures=64, outFeatures=16, bias=false)
      val logits = lm_head( x1 ) //  (B,T,H) @ (H,vocabSize) -> (B,T,vocabSize)

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
  println("Single head attention: BigramLanguageModel3")
  torch.manualSeed(1337)
  val m4 = BigramLanguageModel3(vocab_size, block_size, n_embed)
  m4.to(device)
  m4.train()
  println(totalNuParameters(m4))
  println(moduleInfoString(m4))
  // create a PyTorch optimizer
  // lr=1e-3 suggested by Andrej Karpathy doe no work
  /*
    val batch_size = 32      // how many independent sequences will we process in parallel?
    val block_size = 8       // what is the maximum context length for predictions?
    val max_iters = 5000 
    val eval_interval = 500  
    val learning_rate = 1e-3 // suggested by Andrej Karpathy does no work, 1e-4 requires more iterations
    val eval_iters = 200
    val n_embed = 32 
    val head_size = 16
    val n_head = 4
    val n_layer = 4
    val dropout = 0.0

    Andrej Karpathy video shows:
      - train loss: 2.3940
      - eval loss: 2.4084
      - step: 4500

  Our results with lr=1e-3 results in an ever increasing loss. 
  
  Our results with lr=1e-4
    4977 parameters
    step 0: train loss 4.174592, val loss 4.1752186
    step 500: train loss 3.8263874, val loss 3.8312778
    step 1000: train loss 3.495566, val loss 3.5154147
    step 1500: train loss 3.2679763, val loss 3.2866778
    step 2000: train loss 3.2155032, val loss 3.2392676
    step 2500: train loss 3.2529557, val loss 3.1758127
    step 3000: train loss 3.1138036, val loss 3.1758943
    step 3500: train loss 3.1285887, val loss 3.2269742
    step 4000: train loss 3.0974145, val loss 3.1522665
    step 4500: train loss 3.0333853, val loss 3.062629
    step 4999: train loss 3.0327094, val loss 3.0292172
    Debug m4.generate() !
    decode 6:'








    I
    C,ud-Ttllll'wk ing t wcS
    K
    jovinhervthue matlanrgcyime so end wan S
    whcLith
    i.aaty

    skS m
    tpyo-kOosaulchecrnstrne.

    auaLyytw
    Vys
    sMseve ugd, tles nhTss aralsth'

    Ihv ote, id prsthleeh adEreve fs


    YS
    yI
    r
    Ginutels
    eirthekveEpld,jte olth are fat p kuthpthkrliree thin
    '
    KR
    ChIseasoe:
    LO
    G
    s b;
    x?
    Tcm
    KDjainr o!sherenat


    P
    Cdo, cdrerrethhEyar u,re
    XFor b
    ROyere n tr n uavgrm
    hEan k owYseltohk
    ce

    WyAhn akthsme wumenl'sRns!
    KHy ysI cince Ao.

    Oe rEoto pAu pI atr kind anv in m
    gBnn,o httneetu wte'


  Could not match the training error of 2.4. After 25000 iterations we reached:
    step 24999: 
    train loss 2.5203102
    val loss 2.5169795
  The loss at this point was fluctuating.
  */
  val optimizer4 = torch.optim.AdamW(m4.parameters, lr=1e-4)

  // TODO: reactivate
  val max_iters4 = 1 // 5000 // 25000
  for iter <- 0 until max_iters4
  do
    // every once in a while evaluate the loss on train and val sets
    if (iter % eval_interval == 0) || (iter == max_iters4 - 1)
    then
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
  val losses = estimate_loss(m4)
  println(s"step ${max_iters4-1}: train loss ${losses("train")}, val loss ${losses("val")}")

  // GPU: step 4500: train loss 3.0333855, val loss 3.0626287
  // GPU: step 24999: train loss 2.626873, val loss 2.5710692
  // TODO: reactivate
  // println("Debug m4.generate() !") 
  // // val next6 = m4.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  val next6 = m4.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  val decoded6 = decode(next6.toSeq)
  println(s"decode 6:'$decoded6'")


  def train(m : BigramLanguageModel, learningRate: Double, maxIterations: Int): Unit =
    m.to(device)
    // print the number of parameters in the model
    val nuParams = m.parameters.map(_.numel).sum
    //println(s"${nuParams/1e6}M parameters")
    println(s"Device = ${device}")
    println(s"${nuParams} parameters")
    println(s"learningRate = ${learningRate}")
    println(s"maxIterations = ${maxIterations}")
    m.train()
    // create a PyTorch optimizer
    val optimizer = torch.optim.AdamW(m.parameters, lr=learningRate)

    for iter <- 0 until maxIterations
    do
      // every once in a while evaluate the loss on train and val sets
      if (iter % eval_interval == 0) || (iter == maxIterations - 1)
      then
        val losses = estimate_loss(m)
        println(s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}")
        //print(s"step ${iter}: train loss ${losses("train"):.4f}, val loss ${losses("val"):.4f}")

      // sample a batch of data
      val (xb, yb) = get_batch("train")

      // evaluate the loss
      val (logits, loss) = m(xb, yb)
      optimizer.zeroGrad(setToNone=true)
      loss.backward()
      optimizer.step()
    val losses = estimate_loss(m)
    println(s"step ${maxIterations}: train loss ${losses("train")}, val loss ${losses("val")}")


  val SI = (1000, Vector("B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"))
  val BINARY = (1024, Vector("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"))

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
  def humanReadableSize(bytes: Long, si: Boolean = false): String = 
    // See https://en.wikipedia.org/wiki/Byte
    val (baseValue, unitStrings) =
      if (si)
        SI
      else
        BINARY

    def getExponent(curBytes: Long, baseValue: Int, curExponent: Int = 0): Int =
      if (curBytes < baseValue) 
      then
        curExponent
      else
        val newExponent = 1 + curExponent
        // getExponent(curBytes / (baseValue * newExponent), baseValue, newExponent)
        getExponent(curBytes / baseValue, baseValue, newExponent)

    val exponent = getExponent(bytes, baseValue)
    val divisor = Math.pow(baseValue, exponent)
    val unitString = unitStrings(exponent)

    // Divide the bytes and show one digit after the decimal point
    f"${bytes / divisor}%.1f $unitString"

  // @see https://stackoverflow.com/a/24805871/2051561
  def formatSize(v: Long): String =
    if (v < 1024) 
    then
      v + " B"
    else
      val z = (63 - java.lang.Long.numberOfLeadingZeros(v)) / 10
      val units = " KMGTPE".charAt(z)
      String.format("%.1f %sB", v.toDouble / (1L << (z*10)), units)

  def timeOf() =
    ???

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
    
  val mxBean = java.lang.management.ManagementFactory.getPlatformMXBean(classOf[java.lang.management.ThreadMXBean])

  inline def elapsedCPU[R](inline block: => R): (Long, R) = {
    val t0 = mxBean.getCurrentThreadCpuTime()
    val r = block
    val t1 = mxBean.getCurrentThreadCpuTime()
    val elapsed = t1 - t0
    // elapsed time in nanoseconds
    (elapsed, r)
  }

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

  // https://github.com/sbrunk/storch/issues/5
  // http://bytedeco.org/news/2018/07/17/bytedeco-as-distribution/
  // http://bytedeco.org/javacpp/apidocs/org/bytedeco/javacpp/PointerScope.html
  // https://github.com/microsoft/scala_torch#memory-management
  def train1(m : BigramLanguageModel, learningRate: Double, maxIterations: Int): Unit =
    m.to(device)
    // print the number of parameters in the model
    val nuParams = m.parameters.map(_.numel).sum
    //println(s"${nuParams/1e6}M parameters")
    println(s"Device = ${device}")
    println(s"${nuParams} parameters")
    println(s"learningRate = ${learningRate}")
    println(s"maxIterations = ${maxIterations}")
    println(s"dropout = ${dropout}")
    m.train()
    // create a PyTorch optimizer
    val optimizer = torch.optim.AdamW(m.parameters, lr=learningRate)


    var delta = 0L
    var total = 0L
    for iter <- 0 until maxIterations
    do
      // make sure we deallocate intermediate tensors in time
      Using.resource(new PointerScope()) { p => 
        if (iter % eval_interval == 0) || (iter == maxIterations - 1)
        then
          val losses = estimate_loss(m)
          // println(s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}")
          val memoryBytes = humanReadableSize( Pointer.physicalBytes() )
          delta = delta / eval_interval
          val accumulted = humanReadableDuration(total)
          val perIteration = humanReadableDuration(delta)
          println(s"step ${iter}: train loss ${losses("train")}, val loss ${losses("val")}, mem $memoryBytes @ ${accumulted}, mean $perIteration")
          //print(s"step ${iter}: train loss ${losses("train"):.4f}, val loss ${losses("val"):.4f}")
          delta = 0L

        val elapsed = elapsedOnly {
          // sample a batch of datas
          val (xb, yb) = get_batch("train")

          // evaluate the loss
          val (logits, loss) = m(xb, yb)
          optimizer.zeroGrad(setToNone=true)
          loss.backward()
          optimizer.step()
        }
        delta = delta + elapsed
        total = total + elapsed
      }
      // every once in a while evaluate the loss on train and val sets
    val losses = estimate_loss(m)
    val accumulted = humanReadableDuration(total)
    val perIteration = humanReadableDuration(total / maxIterations)
    println(s"step ${maxIterations}: train loss ${losses("train")}, val loss ${losses("val")}, @ ${accumulted}, mean $perIteration")

  // TODO: reactivate
  // GPU: step 4500: train loss 3.0333855, val loss 3.0626287
  // GPU: step 24999: train loss 2.626873, val loss 2.5710692
  println("Single head attention (b): BigramLanguageModel3")
  torch.manualSeed(1337)
  val m5 = BigramLanguageModel3(vocab_size, block_size, n_embed)
  println(totalNuParameters(m5))
  println(moduleInfoString(m5))

  // TODO: reactivate
  // train(m5, 1e-4, 25000) // gpu
  // // println("Debug m5.generate() !") 
  // // // val next7 = m5.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  // val next7 = m5.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  // val decoded7 = decode(next7.toSeq)
  // println(s"decode 7:'$decoded7'")

  // Multi-head attention v1


  def register_i[M1 <: Module, M2 <: Module](parent: M1, child: M2, i: Int, n: String = "")(using name: sourcecode.Name): M2 =
    val name_ = if n.trim().isEmpty() then name.value else n.trim()
    val name_i = s"${name_}_$i"
    println(s"${moduleClass(parent)} registering ${name_i}:${moduleClass(child)}")
    parent.register(child, name_i)


  /**
   * multiple heads of self-attention in parallel
   */
  class MultiHeadAttention_1[D <: FloatNN: Default](
                            numHeads: Int, 
                            nEmbed: Int, 
                            headSize: Int, 
                            blockSize: Int
                          ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    // Cannot register with the same name
    // val hs = 0 until numHeads map{ _ => register(Head_1(nEmbed, headSize, blockSize)) }
    val hs = 0 until numHeads map{ i => register_i(this, Head_1(nEmbed, headSize, blockSize), i) }
    // TODO: ModuleList registers submodules. Do we still register this one?
    val heads = register( nn.ModuleList( hs:_* ) )

    def forward(x: Tensor[D]): Tensor[D] =
        //torch.cat(heads.modules.map( h => h(x) ), dim=1)
        torch.cat(heads.map( h => h(x) ).toSeq, dim= -1)

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"MultiHeadAttention_1(numHeads=$numHeads, nEmbed=$nEmbed, headSize=$headSize, blockSize=$blockSize)"


  end MultiHeadAttention_1

  /**
   * 
   * @param vocabSize - number o tokens
   * @param blockSize - number of tokens taken from the text input to apply to the NN
   * @param nEmbed - number of values use for latent represent 
   */
  class BigramLanguageModel4(vocabSize: Int, blockSize:Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    // head_size = n_embed, block_size = T
    val sa_heads = register( MultiHeadAttention_1(
                                              numHeads = 4, 
                                              nEmbed = nEmbed, // i.e: with nEmbed = 32, get 4 heads of 8 dimensional self-attention 
                                              headSize = nEmbed/4,
                                              blockSize = blockSize) 
                                            ) //, drop = 0.5)
    // val lm_head = nn.Linear(nEmbed, vocabSize)
    val lm_head = register( nn.Linear(nEmbed, vocabSize) )

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
      val x1 = sa_heads(x0) // apply multiple heads of self-attention (B,T,C) - C = nEmbed
      // val x1 = blocks(x0) // (B,T,C)
      // val x2 = ln_f(x2) // (B,T,C)
      val logits = lm_head( x1 ) //  (B,T,C) @ (C,vocabSize) -> (B,T,vocabSize)

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


  end BigramLanguageModel4

  // Andrej Karpathy gets 2.2858 @ 4500 iterations
  // Here we get  3.1948392 @ 4500 iterations
  // Loss seems to vary while still converging until 3.115641 @ 7500 
  // At 8000 the loss explodes
  // torch.manualSeed(1337)
  /*
    lr = 1e-3

    lr = 1e-4

    lr = 5e-5
    lr = 2.5e-5

    lr = 1.5e-5
    step 0: train loss 4.3145924, val loss 4.3047194
    step 500: train loss 4.1578984, val loss 4.1541634
    step 1000: train loss 4.021874, val loss 4.0226283
    step 1500: train loss 3.8988528, val loss 3.9039721
    step 2000: train loss 3.778538, val loss 3.7922485
    step 2500: train loss 3.6588502, val loss 3.6669707
    step 3000: train loss 3.5310557, val loss 3.5450792
    step 3500: train loss 3.43383, val loss 3.4569433
    step 4000: train loss 3.3714166, val loss 3.39132
    step 4500: train loss 3.3257911, val loss 3.350137
    step 5000: train loss 3.3006864, val loss 3.3212936
    step 5500: train loss 3.2539845, val loss 3.3066149
    step 6000: train loss 3.2607672, val loss 3.2855794
    step 6500: train loss 3.240766, val loss 3.264896
    step 7000: train loss 3.2341163, val loss 3.2789524
    step 7500: train loss 3.220194, val loss 3.23849
    step 8000: train loss 3.2175868, val loss 3.2448592
    step 8500: train loss 3.2074897, val loss 3.2803164
    step 9000: train loss 3.329978, val loss 3.3710482
    step 9500: train loss 3.3176076, val loss 3.3755133
    step 10000: train loss 3.263973, val loss 3.3106585
    step 10500: train loss 3.1931515, val loss 3.2370577
    step 11000: train loss 3.1550646, val loss 3.239709
    step 11500: train loss 3.158176, val loss 3.186146
    step 12000: train loss 3.1671298, val loss 3.2042825
    step 12500: train loss 3.0862591, val loss 3.165631
    step 13000: train loss 3.134143, val loss 3.1711876
    step 13500: train loss 3.1777081, val loss 3.1412072
    step 14000: train loss 3.1187584, val loss 3.150917
    step 14500: train loss 3.7870364, val loss 3.1683056
    step 15000: train loss 3.1235626, val loss 3.0829206
    step 15500: train loss 3.1629584, val loss 3.1274188
    step 16000: train loss 3.0788772, val loss 3.2236586
    step 16500: train loss 3.2564025, val loss 3.1015463
    step 17000: train loss 3.8839633, val loss 3.2737646
    step 17500: train loss 3.059018, val loss 3.080443
    step 18000: train loss 3.0584357, val loss 3.0818381
    step 18500: train loss 3.088948, val loss 3.0591516
    step 19000: train loss 3.0200274, val loss 3.0359771
    step 19500: train loss 3.0271075, val loss 3.0594947
    step 20000: train loss 2.985976, val loss 3.019711
    step 20500: train loss 3.1742153, val loss 3.0725644
    step 21000: train loss 3.1546068, val loss 3.1267648
    step 21500: train loss 3.0989442, val loss 3.130278
    step 22000: train loss 3.2724228, val loss 3.1326475
    step 22500: train loss 3.0416803, val loss 3.0480652
    step 23000: train loss 3.118403, val loss 3.0931244
    step 23500: train loss 3.147756, val loss 3.095541
    step 24000: train loss 3.0405772, val loss 3.051168
    step 24500: train loss 3.0144293, val loss 3.055157
    step 24999: train loss 3.0445592, val loss 3.0505664
    step 24999: train loss 3.097772, val loss 3.0937912

    lr = 1e-5
    step 0: train loss 4.315746, val loss 4.3061743
    step 500: train loss 4.2083063, val loss 4.2047343
    step 1000: train loss 4.109281, val loss 4.1095076
    step 1500: train loss 4.024676, val loss 4.021858
    step 2000: train loss 3.9401476, val loss 3.9419503
    step 2500: train loss 3.861138, val loss 3.868681
    step 3000: train loss 3.7746782, val loss 3.7817297
    step 3500: train loss 3.6901476, val loss 3.7049506
    step 4000: train loss 3.599073, val loss 3.617259
    step 4500: train loss 3.5131109, val loss 3.5384142
    step 5000: train loss 3.452971, val loss 3.4619794
    step 5500: train loss 3.399948, val loss 3.4254942
    step 6000: train loss 3.3541067, val loss 3.3918
    step 6500: train loss 3.3242495, val loss 3.3732038
    step 7000: train loss 3.3144944, val loss 3.3490424
    step 7500: train loss 3.2901514, val loss 3.2941566
    step 8000: train loss 3.2899778, val loss 3.308439
    step 8500: train loss 3.2639534, val loss 3.2906058
    step 9000: train loss 3.2651227, val loss 3.2723944
    step 9500: train loss 3.2395923, val loss 3.2861238
    step 10000: train loss 3.2434728, val loss 3.257814
    step 10500: train loss 3.2285821, val loss 3.23281
    step 11000: train loss 3.2198544, val loss 3.2416165
    step 11500: train loss 3.2021954, val loss 3.2313745
    step 12000: train loss 3.195072, val loss 3.2142315
    step 12500: train loss 3.1960852, val loss 3.2163675
    step 13000: train loss 3.1769931, val loss 3.2013638
    step 13500: train loss 3.17453, val loss 3.2119668
    step 14000: train loss 3.1472147, val loss 3.1825323
    step 14500: train loss 3.1611233, val loss 3.192211
    step 15000: train loss 3.1517265, val loss 3.1621974
    step 15500: train loss 3.1394618, val loss 3.1598687
    step 16000: train loss 3.1233463, val loss 3.145328
    step 16500: train loss 3.1227674, val loss 3.1421418
    step 17000: train loss 3.1164768, val loss 3.1276824
    step 17500: train loss 3.1011841, val loss 3.0985348
    step 18000: train loss 3.0856524, val loss 3.11533
    step 18500: train loss 3.0842745, val loss 3.0987678
    step 19000: train loss 3.049956, val loss 3.1043591
    step 19500: train loss 3.0564034, val loss 3.0689766
    step 20000: train loss 3.0590668, val loss 3.0758286
    step 20500: train loss 3.0560205, val loss 3.0690722
    step 21000: train loss 3.0467145, val loss 3.0635276
    step 21500: train loss 3.0318224, val loss 3.0459983
    step 22000: train loss 3.025454, val loss 3.0337
    step 22500: train loss 3.0058165, val loss 3.0480902
    step 23000: train loss 3.0240664, val loss 3.0332391
    step 23500: train loss 2.9987218, val loss 3.023562
    step 24000: train loss 2.985587, val loss 3.0277314
    step 24500: train loss 2.9775257, val loss 3.002483
    step 24999: train loss 2.9854958, val loss 3.0055265
    step 24999: train loss 2.9771202, val loss 3.0027666

    lr = 5e-6

  */
  // nohup ./mill examples.runMain gpt.BiGram > expm4_1_5b.txt 2>&1 &
  // torch.manual_seed(torch.initial_seed())
  // torch.cuda.manual_seed_all
  println("Multi-head attention BigramLanguageModel4")
  val m6 = BigramLanguageModel4(vocab_size, block_size, n_embed)
  println(totalNuParameters(m6))
  println(moduleInfoString(m6))
  //train(m6, 1e-3, 25000)
  //train(m6, 1e-4, 25000)
  // train(m6, 1.5e-5, 45000)
  // torch.manualSeed(1337)
  torch.manualSeed(6106)
  // train(m6, 5e-6, 25000)
  // TODO: reactivate
  // train(m6, 1.3e-5, 75000) // 2.582383 at 10500 
  // train(m6, 1.4e-5, 75000) // start diverging at 10500 
  // TODO: reactivate
  // train(m6, 1.35e-5, 75000) // 2.5889513 at 75000
  // // Seed(6106) GPU: step 75000: train loss 2.5031831, val loss 2.5004642
  // // Seed(1337) GPU: step 75000: train loss 2.4987864, val loss 2.5220866
  // val next8 = m6.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  // val decoded8 = decode(next8.toSeq)
  // println(s"decode 8:'$decoded8'")


  // Adding a feed forward layer to combine 


  /*
  class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    */

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


  /** a simple linear layer followed by a non-linearity
    *
    * @param numHeads
    * @param nEmbed
    * @param headSize
    * @param blockSize
    */
  class FeedFoward[D <: FloatNN: Default](
                          nEmbed: Int 
                        ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:
    val net = register( nn.Sequential(
                    nn.Linear(nEmbed, nEmbed),
                    nn.ReLU()
                    // nn.Linear(nEmbed, 4 * nEmbed),
                    // nn.ReLU(),
                    // nn.Linear(4 * nEmbed, nEmbed),
                    // nn.Dropout(dropout),
              )
            )

    def forward(x: Tensor[D]): Tensor[D] = net(x)

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"FeedFoward(nEmbed = $nEmbed)"

  end FeedFoward



  /**
   * 
   * @param vocabSize - number o tokens
   * @param blockSize - number of tokens taken from the text input to apply to the NN
   * @param nEmbed - number of values use for latent represent 
   */
  class BigramLanguageModel5(vocabSize: Int, blockSize:Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    // head_size = n_embed, block_size = T
    val sa_heads = register( MultiHeadAttention_1(
                                              numHeads = 4, 
                                              nEmbed = nEmbed, // i.e: with nEmbed = 32, get 4 heads of 8 dimensional self-attention 
                                              headSize = nEmbed/4,
                                              blockSize = blockSize) 
                                            ) //, drop = 0.5)
    val ffwd = register( FeedFoward(nEmbed) )
    val lm_head = register( nn.Linear(nEmbed, vocabSize) )

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
      val x1 = sa_heads(x0) // apply multiple heads of self-attention (B,T,C) - C = nEmbed
      val x2 = ffwd(x1) // (B, T, C)
      // val x1 = blocks(x0) // (B,T,C)
      // val x2 = ln_f(x2) // (B,T,C)
      val logits = lm_head( x2 ) //  (B,T,C) @ (C,vocabSize) -> (B,T,vocabSize)

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


  end BigramLanguageModel5

  // nohup ./mill examples.runMain gpt.BiGram > expm5_1_5a.txt 2>&1 &
  // Andrej karpathy gets 2.2412 @ 4500
  // We get 4.1882143 @ 4500 At this point we are already diverging 
  // Solution is more stable but convergence irratic. 
  torch.manualSeed(1337)
  println("Multi-head attention + FFWD BigramLanguageModel5")
  val m7 = BigramLanguageModel5(vocab_size, block_size, n_embed)
  println(totalNuParameters(m7))
  println(moduleInfoString(m7))
  // train(m7, 1e-4, 25000)
  // train(m7, 1e-5, 25000)
  //train(m7, 1.5e-5, 25000)
  // train(m7, 1e-5, 75000) // GPU: diverges
  // train(m7, 1.3e-5, 75000)
  // train(m7, 1.5e-5, 75000)
  // TODO: reactivate
  // train(m7, 2e-5, 75000) // CPU: ok, GPU: deviates
  // train(m7, 1e-6, 75000) // GPU: step 75000: train loss 3.1923885, val loss 3.2303805
  // train(m7, 5e-6, 75000) // GPU: step 75000: train loss 2.7268057, val loss 2.7338605
  // train(m7, 6e-6, 75000) // GPU: step 75000: train loss 2.6771119, val loss 2.6831217
  // train(m7, 7e-6, 75000) // GPU: step 75000: train loss 2.6430547, val loss 2.649345
  // train(m7, 7.2e-6, 75000) // GPU: step 75000: train loss 2.637256, val loss 2.6441076
  // // train(m7, 7.5e-6, 75000) // GPU: diverges
  // //train(m7, 8e-6, 75000) // GPU: diverges
  // val next9 = m7.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  // val decoded9 = decode(next9.toSeq)
  // println(s"decode 9:'$decoded9'")

  
// Adding blocks

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
                          vocabSize: Int
                        ) extends torch.nn.modules.TensorModule[D]: 

    // n_embd: embedding dimension, n_head: the number of heads we'd like
    val headSize = nEmbed / nHead
    val sa = register( MultiHeadAttention_1(
                                              numHeads = 4, 
                                              nEmbed = nEmbed, // i.e: with nEmbed = 32, get 4 heads of 8 dimensional self-attention 
                                              headSize = headSize,
                                              blockSize = blockSize) 
                                            ) //, drop = 0.5)
    val ffwd = register( FeedFoward(nEmbed) )
    // val lm_head = register( nn.Linear(nEmbed, vocabSize) )
    // TODO: does not exist
    // val ln1 = nn.LayerNorm(nEmbed)
    // val ln2 = nn.LayerNorm(nEmbed)

    def forward(x: Tensor[D]): Tensor[D] = ffwd( sa(x) )

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(nEmbed = $nEmbed)"

  end Block



  /**
   * use blocks of multi-head attention
   * 
   * @param vocabSize - number o tokens
   * @param blockSize - number of tokens taken from the text input to apply to the NN
   * @param nEmbed - number of values use for latent represent 
   */
  class BigramLanguageModel6(vocabSize: Int, blockSize:Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    val nHead = 4
    val blocks = register( 
      nn.Sequential(
          Block(nEmbed, nHead, blockSize, vocabSize),
          Block(nEmbed, nHead, blockSize, vocabSize),
          Block(nEmbed, nHead, blockSize, vocabSize)
        )
      )
    val lm_head = register( nn.Linear(nEmbed, vocabSize) )
    // Possible memory leak
    // positions of tokens
    // val pos = torch.arange(0L,blockSize, dtype=int64, device=device) // (T) were T is the block size?
    val pos = torch.arange(0L,blockSize, dtype=int64, device=device) // (T) were T is the block size?

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =
      val Seq(b,t) = idx.shape

      // idx and targets are both (B,T) tensor of integers
      // idx is (B,T)
      val token_embed = token_embedding_table( idx ) // (B,T,C) where C is nEmbed
      // Possible memory leak
      // positions of tokens
      // val pos = torch.arange(0L,t, device=device) // (T) were T is the block size?
      val pos_embed = position_embedding_table( pos ) // (T,C)
      // Add the position embeddings
      val x0 = token_embed + pos_embed // (B,T,C)
      val x1 = blocks(x0) // apply blocks of self-attention (B,T,C) - C = nEmbed
      val logits = lm_head( x1 ) //  (B,T,C) @ (C,vocabSize) -> (B,T,vocabSize)

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


  end BigramLanguageModel6

  // nohup ./mill examples.runMain gpt.BiGram > expm5_1_5a.txt 2>&1 &
  // Andrej karpathy gets 2.2412 @ 4500
  // We get 4.1882143 @ 4500 At this point we are already diverging 
  // Solution is more stable but convergence irratic. 
  torch.manualSeed(1337)
  println("Self attention Blocks BigramLanguageModel6")
  val m8 = BigramLanguageModel6(vocab_size, block_size, n_embed)
  println(totalNuParameters(m8))
  println(moduleInfoString(m8))
  // TODO: reactivate
  // // train(m8, 1.0e-5, 75000) // GPU: diverges
  // // train(m8, 1.5e-5, 75000) // breaks
  // // train(m8, 7.2e-6, 75000) // GPU: diverges, model 5 - step 75000: train loss 2.637256, val loss 2.6441076
  // //train(m8, 1.0e-6, 75000) // GPU: out of memory
  // train(m8, 2.0e-6, 75000) // GPU: out of memory
  // val next10 = m8.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  // val decoded10 = decode(next10.toSeq)
  // println(s"decode 10:'$decoded10'")
  // 1/0

/*
train(m8, 1.0e-6, 75000)
step 33000: train loss 3.4841979, val loss 3.5126445
Exception in thread "main" java.lang.ExceptionInInitializerError
        at gpt.BiGram.main(BiGram.scala)
Caused by: java.lang.RuntimeException: CUDA out of memory. Tried to allocate 2.00 MiB (GPU 0; 23.69 GiB total capacity; 23.37 GiB already allocated; 2.81 MiB free; 23.37 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
*/


  // Adding residual connections


  /**
   * Add residual connections 
   * 
   */
  class Block_2[D <: FloatNN: Default](
                          nEmbed: Int, 
                          nHead: Int,
                          blockSize: Int,
                          vocabSize: Int
                        ) extends torch.nn.modules.TensorModule[D]: 

    // n_embd: embedding dimension, n_head: the number of heads we'd like
    val headSize = nEmbed / nHead
    val sa = register( MultiHeadAttention_2(
                                              numHeads = 4, 
                                              nEmbed = nEmbed, // i.e: with nEmbed = 32, get 4 heads of 8 dimensional self-attention 
                                              headSize = headSize,
                                              blockSize = blockSize) 
                                            ) //, drop = 0.5)
    val ffwd = register( FeedFoward_2(nEmbed) )
    // val lm_head = register( nn.Linear(nEmbed, vocabSize) )
    // TODO: does not exist
    // val ln1 = nn.LayerNorm(nEmbed)
    // val ln2 = nn.LayerNorm(nEmbed)

    def forward(x: Tensor[D]): Tensor[D] = 
      val x1 = x + sa(x)
      x1 + ffwd(x)

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(nEmbed = $nEmbed)"

  end Block_2


  /**
   * multiple heads of self-attention in parallel with residual connection
   * 
   * "the element-wise addition F(x) + x makes sense only if F(x) and x have 
   * the same dimensions. If their dimensions are different, we can replace 
   * the identity mapping with a linear transformation (i.e. multiplication 
   * by a matrix W), and perform F(x) + Wx instead."
   * https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55
   */
  class MultiHeadAttention_2[D <: FloatNN: Default](
                            numHeads: Int, 
                            nEmbed: Int, 
                            headSize: Int, 
                            blockSize: Int
                          ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    // Cannot register with the same name
    // val hs = 0 until numHeads map{ _ => register(Head_1(nEmbed, headSize, blockSize)) }
    val hs = 0 until numHeads map{ i => register_i(this, Head_1(nEmbed, headSize, blockSize), i) }
    val heads = register( nn.ModuleList( hs:_* ) )
    val proj = register( nn.Linear(nEmbed, nEmbed) )

    def forward(x: Tensor[D]): Tensor[D] =
        //torch.cat(heads.modules.map( h => h(x) ), dim=1)
        val out = torch.cat(heads.map( h => h(x) ).toSeq, dim= -1)
        // linear combination of out back into path - should this not be on x?
        proj(out)

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass().getSimpleName()}(numHeads=$numHeads, nEmbed=$nEmbed, headSize=$headSize, blockSize=$blockSize)"


  end MultiHeadAttention_2

  /** a simple linear layer followed by a non-linearity
   * Added residual connectiopn directly into [[nn.Sequention]].
   * Could have done it after as another Linear layer. 
   *
   * @param numHeads
   * @param nEmbed
   * @param headSize
   * @param blockSize
   */
  class FeedFoward_2[D <: FloatNN: Default](
                          nEmbed: Int 
                        ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:
    val net = register( nn.Sequential(
                    // Increase output dimension by 4
                    nn.Linear(nEmbed, 4L * nEmbed),
                    nn.ReLU(),
                    // Decrease output dimension by 4
                    nn.Linear(4L*nEmbed, nEmbed), // residual network implented using projection - why not on x directly?
                    // nn.Dropout(dropout),
              )
            )

    def forward(x: Tensor[D]): Tensor[D] = net(x)

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass().getSimpleName()}(nEmbed = $nEmbed)"

  end FeedFoward_2


  /**
   * use blocks of multi-head attention
   * Add residual connections
   * 
   * @param vocabSize - number o tokens
   * @param blockSize - number of tokens taken from the text input to apply to the NN
   * @param nEmbed - number of values use for latent represent 
   */
  class BigramLanguageModel7(vocabSize: Int, blockSize:Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    val nHead = 4
    val blocks = register( 
      nn.Sequential(
          Block_2(nEmbed, nHead, blockSize, vocabSize),
          Block_2(nEmbed, nHead, blockSize, vocabSize),
          Block_2(nEmbed, nHead, blockSize, vocabSize)
        )
      )
    val lm_head = register( nn.Linear(nEmbed, vocabSize) )
    // Possible memory leak
    // positions of tokens
    // val pos = torch.arange(0L,blockSize, dtype=int64, device=device) // (T) were T is the block size?
    val pos = torch.arange(0L,blockSize, dtype=int64, device=device) // (T) were T is the block size?

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =
      val Seq(b,t) = idx.shape

      // idx and targets are both (B,T) tensor of integers
      // idx is (B,T)
      val token_embed = token_embedding_table( idx ) // (B,T,C) where C is nEmbed
      // Possible memory leak
      // positions of tokens
      // val pos = torch.arange(0L,t, device=device) // (T) were T is the block size?
      val pos_embed = position_embedding_table( pos ) // (T,C)
      // Add the position embeddings
      val x0 = token_embed + pos_embed // (B,T,C)
      val x1 = blocks(x0) // apply blocks of self-attention (B,T,C) - C = nEmbed
      val logits = lm_head( x1 ) //  (B,T,C) @ (C,vocabSize) -> (B,T,vocabSize)

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


  end BigramLanguageModel7

  // nohup ./mill examples.runMain gpt.BiGram > expm5_1_5a.txt 2>&1 &
  // Andrej karpathy gets 2.08 @ 4500 (train loss is at 1.9993 which shows overfitting) 
  // We get 3.1151032 @ 4500 
  // Solution is more stable but convergence irratic. 
  torch.manualSeed(1337)
  println("Self attention Blocks + Residual connections - BigramLanguageModel7")
  val m9 = BigramLanguageModel7(vocab_size, block_size, n_embed)
  println(totalNuParameters(m9))
  println(moduleInfoString(m9))
  // train(m9, 1.0e-6, 75000) // step 75000: train loss 2.8870237, val loss 2.9272516
  // train(m9, 2.0e-6, 75000)  // step 75000: train loss 2.675596, val loss 2.6954281
  // train(m9, 3.0e-6, 75000)  // step 75000: train loss 2.5539615, val loss 2.56638
  // train(m9, 5.0e-6, 75000)  // step 75000: train loss 2.426662, val loss 2.437024
  // train(m9, 8.0e-6, 75000)  // diverges from step 22500: train loss 3.0226552, val loss 3.1092987
  // train(m9, 7.0e-6, 75000)  // diverges from step 29000: train loss 2.6225424, val loss 2.619104
  // train(m9, 6.0e-6, 75000)  // step 75000: train loss 2.3925982, val loss 2.39741
  // export TF_ENABLE_ONEDNN_OPTS=0
  // TODO: reactivate
  // // train(m9, 6.0e-6, 75000)  // GPU: diverges
  // // train(m9, 6.0e-6, 75000)  // GPU: out of memory
  // // train1(m9, 6.0e-6, 75000)  // GPU: step 75000: train loss 2.3715358, val loss 2.3802633, @ 00 00:21:35.509, mean 00 00:00:00.017
  // train1(m9, 6.0e-6, 200_000)  // GPU: 
  // // train1(m9, 8.0e-6, 75000)  // GPU: diverges after step 7000: train loss 2.9647808, val loss 2.9952092, mem 1.8 GiB @ 00 00:01:54.857, mean 00 00:00:00.016
  // // train1(m9, 1.0e-5, 75000)  // GPU: diverges step 9500: train loss 3.0968044, val loss 3.1299486, mem 1.8 GiB @ 00 00:02:40.739, mean 00 00:00:00.017
  // // train1(m9, 1.5e-5, 75000)  // GPU: diverges after step 5000: train loss 4891399.5, val loss 1130637.2, mem 1.8 GiB @ 00 00:01:25.711, mean 00 00:00:00.017
  // // train(m9, 6.0e-6, 100_000)  
  // // // train(m9, 1.0e-5, 75000) // breaks
  // // // train(m9, 1.5e-5, 75000) 
  // val next11 = m9.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  // val decoded11 = decode(next11.toSeq)
  // println(s"decode 11:'$decoded11'")
  // 1/0


    /**
   * multiple heads of self-attention in parallel with residual connection
   * Add dropout
   * 
   * "the element-wise addition F(x) + x makes sense only if F(x) and x have 
   * the same dimensions. If their dimensions are different, we can replace 
   * the identity mapping with a linear transformation (i.e. multiplication 
   * by a matrix W), and perform F(x) + Wx instead."
   * https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55
   */
  class MultiHeadAttention_3[D <: FloatNN: Default](
                            numHeads: Int, 
                            nEmbed: Int, 
                            headSize: Int, 
                            blockSize: Int
                          ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:

    // Cannot register with the same name
    // val hs = 0 until numHeads map{ _ => register(Head_2(nEmbed, headSize, blockSize)) }
    val hs = 0 until numHeads map{ i => register_i(this, Head_2(nEmbed, headSize, blockSize), i) }
    val heads = register( nn.ModuleList( hs:_* ) )
    val proj = register( nn.Linear(nEmbed, nEmbed) )
    val drop = register( nn.Dropout(dropout) )

    def forward(x: Tensor[D]): Tensor[D] =
        //torch.cat(heads.modules.map( h => h(x) ), dim=1)
        val out = torch.cat(heads.map( h => h(x) ).toSeq, dim= -1)
        // linear combination of out back into path - should this not be on x?
        drop( proj(out) )

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass().getSimpleName()}(numHeads=$numHeads, nEmbed=$nEmbed, headSize=$headSize, blockSize=$blockSize)"


  end MultiHeadAttention_3

  /**
   * Add residual connections 
   * 
   */
  class Block_3[D <: FloatNN: Default](
                          nEmbed: Int, 
                          nHead: Int,
                          blockSize: Int,
                          vocabSize: Int
                        ) extends torch.nn.modules.TensorModule[D]: 

    // n_embd: embedding dimension, n_head: the number of heads we'd like
    val headSize = nEmbed / nHead
    val sa = register( MultiHeadAttention_3(
                                              numHeads = 4, 
                                              nEmbed = nEmbed, // i.e: with nEmbed = 32, get 4 heads of 8 dimensional self-attention 
                                              headSize = headSize,
                                              blockSize = blockSize) 
                                            ) //, drop = 0.5)
    val ffwd = register( FeedFoward_2(nEmbed) )
    // val lm_head = register( nn.Linear(nEmbed, vocabSize) )
    val ln1 = register( nn.LayerNorm(Seq(nEmbed)) )
    val ln2 = register( nn.LayerNorm(Seq(nEmbed)) )

    def forward(x: Tensor[D]): Tensor[D] = 
      val x1 = x + sa( ln1(x) )
      x1 + ffwd( ln2(x) )

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(nEmbed = $nEmbed)"

  end Block_3




  /**
   * use blocks of multi-head attention
   * Add residual connections
   * 
   * @param vocabSize - number o tokens
   * @param blockSize - number of tokens taken from the text input to apply to the NN
   * @param nEmbed - number of values use for latent represent 
   */
  class BigramLanguageModel8(vocabSize: Int, blockSize:Int, nEmbed: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    val nHead = 4
    val blocks = register( 
      nn.Sequential(
          Block_3(nEmbed, nHead, blockSize, vocabSize),
          Block_3(nEmbed, nHead, blockSize, vocabSize),
          Block_3(nEmbed, nHead, blockSize, vocabSize), 
          nn.LayerNorm(Seq(nEmbed))
        )
      )
    val lm_head = register( nn.Linear(nEmbed, vocabSize) )

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
      val logits = lm_head( x1 ) //  (B,T,C) @ (C,vocabSize) -> (B,T,vocabSize)

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


  end BigramLanguageModel8

  // nohup ./mill examples.runMain gpt.BiGram > expm5_1_5a.txt 2>&1 &
  // Andrej karpathy gets 2.0607 @ 5000 (train loss is at 1.9829 which shows overfitting) 
  // We get 3.1151032 @ 4500 
  // Solution is more stable but convergence irratic. 
  torch.manualSeed(1337)
  println("Self attention Blocks + Residual connections + layer norm - BigramLanguageModel8")
  val m10 = BigramLanguageModel8(vocab_size, block_size, n_embed)
  println(totalNuParameters(m10))
  println(moduleInfoString(m10))
  // train(m10, 1.0e-6, 75000) // step 75000: train loss 2.8870237, val loss 2.9272516
  // train(m10, 2.0e-6, 75000)  // step 75000: train loss 2.675596, val loss 2.6954281
  // train(m10, 3.0e-6, 75000)  // step 75000: train loss 2.5539615, val loss 2.56638
  // train(m10, 5.0e-6, 75000)  // step 75000: train loss 2.426662, val loss 2.437024
  // train(m10, 8.0e-6, 75000)  // diverges from step 22500: train loss 3.0226552, val loss 3.1092987
  // train(m10, 7.0e-6, 75000)  // diverges from step 29000: train loss 2.6225424, val loss 2.619104
  // train(m10, 6.0e-6, 75000)  // step 75000: train loss 2.3925982, val loss 2.39741
  // export TF_ENABLE_ONEDNN_OPTS=0
  // // TODO: reactivate
  // // train(m10, 6.0e-6, 75000)  // step 75000: train loss 2.4230623, val loss 2.4327655
  // // train(m10, 7.0e-6, 75000)  // step 75000: train loss 2.4224014, val loss 2.4300454
  // // train(m10, 8.0e-6, 75000)  // GPU out of memeory
  // // train(m10, 9.0e-6, 75000)  // GPU out of memory
  // // train1(m10, 8.0e-6, 75000)  // GPU
  // train1(m10, 9.0e-6, 75000)  // GPU
  // // // train(m10, 1.0e-5, 75000) // breaks
  // // // train(m10, 1.5e-5, 75000) 
  // // // TODO: reactivate
  // val next12 = m10.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  // val decoded12 = decode(next12.toSeq)
  // println(s"decode 12:'$decoded12'")
  // 1/0



  // Add Dropout

  /**
   * one head of self-attention
   * add dropout
   */
  class Head_2[D <: FloatNN: Default](
          n_embed: Int, 
          head_size: Int, 
          block_size: Int
          //drop: Double
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
      val soft = F.softmax(mask, dim= -1) // (B, T, T)
      val wei = drop( soft )
      // perform the weighted aggregation of the values
      val v = value(x) // (B,T,H)
      val out = wei `@` v // (B, T, T) @ (B, T, H) -> (B, T, H)
      out

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(n_embed=$n_embed, head_size=$head_size, block_size=$block_size)"
  end Head_2

  /** a simple linear layer followed by a non-linearity
   * Added residual connectiopn directly into [[nn.Sequention]].
   * Could have done it after as another Linear layer. 
   * Add Dropout
   *
   * @param numHeads
   * @param nEmbed
   * @param headSize
   * @param blockSize
   */
  class FeedFoward_3[D <: FloatNN: Default](
                          nEmbed: Int 
                        ) extends torch.nn.modules.TensorModule[D]: // extends nn.Module:
    val net = register( nn.Sequential(
                    // Increase output dimension by 4
                    nn.Linear(nEmbed, 4L * nEmbed),
                    nn.ReLU(),
                    // Decrease output dimension by 4
                    nn.Linear(4L*nEmbed, nEmbed), // residual network implented using projection - why not on x directly?
                    nn.Dropout(dropout),
              )
            )

    def forward(x: Tensor[D]): Tensor[D] = net(x)

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass().getSimpleName()}(nEmbed = $nEmbed)"

  end FeedFoward_3


  /**
   * Add residual connections 
   * Add dropout
   * 
   */
  class Block_4[D <: FloatNN: Default](
                          nEmbed: Int, 
                          nHead: Int,
                          blockSize: Int,
                          vocabSize: Int
                        ) extends torch.nn.modules.TensorModule[D]: 

    // n_embd: embedding dimension, n_head: the number of heads we'd like
    val headSize = nEmbed / nHead
    val sa = register( MultiHeadAttention_3(
                                              numHeads = 4, 
                                              nEmbed = nEmbed, // i.e: with nEmbed = 32, get 4 heads of 8 dimensional self-attention 
                                              headSize = headSize,
                                              blockSize = blockSize) 
                                            ) //, drop = 0.5)
    val ffwd = register( FeedFoward_3(nEmbed) )
    // val lm_head = register( nn.Linear(nEmbed, vocabSize) )
    val ln1 = register( nn.LayerNorm(Seq(nEmbed)) )
    val ln2 = register( nn.LayerNorm(Seq(nEmbed)) )

    def forward(x: Tensor[D]): Tensor[D] = 
      val x1 = x + sa( ln1(x) )
      x1 + ffwd( ln2(x) )

    def apply(x:Tensor[D]): Tensor[D] = forward(x)

    override def toString(): String = s"${getClass.getSimpleName()}(nEmbed = $nEmbed)"

  end Block_4



  /**
   * use blocks of multi-head attention
   * Add residual connections
   * Add Dropout
   * 
   * @param vocabSize - number o tokens
   * @param blockSize - number of tokens taken from the text input to apply to the NN
   * @param nEmbed - number of values use for latent represent 
   * @param nBlocks - number of blocks
   * @param nHead- number of heads per block
   * 
   */
  class BigramLanguageModel9(
    vocabSize: Int, 
    blockSize:Int, 
    nEmbed: Int,
    nBlocks: Int,
    nHead: Int
    ) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = register( nn.Embedding(vocabSize, nEmbed) )
    val position_embedding_table = register( nn.Embedding(blockSize, nEmbed) )
    val blocks_i = 0 until nBlocks map { i => Block_4(nEmbed, nHead, blockSize, vocabSize) }
    val blocks = register(  nn.Sequential( blocks_i:_* ) )
    val ln_f = register( nn.LayerNorm(Seq(nEmbed)) )
    val lm_head = register( nn.Linear(nEmbed, vocabSize) )

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


  end BigramLanguageModel9

  // nohup ./mill examples.runMain gpt.BiGram > expm5_1_5a.txt 2>&1 &
  // Andrej karpathy gets 2.0607 @ 5000 (train loss is at 1.9829 which shows overfitting) 
  // We get 3.1151032 @ 4500 
  // Solution is more stable but convergence irratic. 
  torch.manualSeed(1337)
  println("Self attention Blocks + Residual connections + layer norm + Dropout - BigramLanguageModel9")
  val m11 = BigramLanguageModel9( vocabSize = vocab_size, blockSize = block_size, nEmbed = n_embed, nBlocks = 3, nHead = 4)
  println(totalNuParameters(m11))
  println(moduleInfoString(m11))
  // train1(m11, 9.0e-6, 75000)  // GPU drop=0.5
  // train1(m11, 9.0e-6, 75000)  // GPU drop=0.2
  // train1(m11, 1.0e-5, 75000)  // GPU drop=0.2
  train1(m11, 1.0e-4, 75000)  // GPU drop=0.2
  val next13 = m11.generate(idx = torch.zeros(Seq(1, block_size), dtype=torch.int64, device=device), max_new_tokens=500)(0)
  val decoded13 = decode(next13.toSeq)
  println(s"decode 13:'$decoded13'")
  1/0


  // https://stackoverflow.com/questions/16809134/how-to-get-a-list-of-programs-running-with-nohup
  
  def main(args: Array[String]): Unit =
    ()

end BiGram


/*
'/workspaces/storch/.metals/millw --mill-version 0.11.4 --import ivy:com.lihaoyi::mill-contrib-bloop: mill.contrib.bloop.Bloop/install'
'/workspaces/storch/.metals/millw --mill-version 0.11.5 --import ivy:com.lihaoyi::mill-contrib-bloop: mill.contrib.bloop.Bloop/install'

From Mill docs

mill --import "ivy:com.lihaoyi::mill-contrib-bloop::" mill.contrib.bloop.Bloop/install
mill mill.bsp.BSP/install

*/

/*

<
Embedding(numEmbeddings=65, embeddingDim=32, paddingIdx=None, maxNorm=None, normType=Some(2.0), scaleGradByFreq=false, sparse=false ) : ArraySeq(2080),
Embedding(numEmbeddings=8, embeddingDim=32, paddingIdx=None, maxNorm=None, normType=Some(2.0), scaleGradByFreq=false, sparse=false ) : ArraySeq(256),
MultiHeadAttention_1(numHeads=4, nEmbed=32, headSize=8, blockSize=8) : ArraySeq(),
FeedFoward(nEmbed = 32) : ArraySeq(),
Linear(inFeatures=32, outFeatures=65, bias=true) : ArraySeq(2080, 65)>

ArraySeq(2080, 256, 2080, 65)


<
Embedding(numEmbeddings=65, embeddingDim=32, paddingIdx=None, maxNorm=None, normType=Some(2.0), scaleGradByFreq=false, sparse=false ) : ArraySeq(2080),
Embedding(numEmbeddings=8, embeddingDim=32, paddingIdx=None, maxNorm=None, normType=Some(2.0), scaleGradByFreq=false, sparse=false ) : ArraySeq(256),
MultiHeadAttention_1(numHeads=4, nEmbed=32, headSize=8, blockSize=8) : ArraySeq(),
FeedFoward(nEmbed = 32)(
  (net): Sequential(
    (0): Linear(inFeatures=32, outFeatures=32, bias=true)
    (1): ReLU
  )
) : ArraySeq(1024, 32),
Sequential(
  (0): Linear(inFeatures=32, outFeatures=32, bias=true)
  (1): ReLU
) : ArraySeq(1024, 32),
Linear(inFeatures=32, outFeatures=32, bias=true) : ArraySeq(1024, 32),
ReLU : ArraySeq(),
Linear(inFeatures=32, outFeatures=65, bias=true) : ArraySeq(2080, 65)>

ArraySeq(2080, 256, 1024, 32, 2080, 65)







*/

