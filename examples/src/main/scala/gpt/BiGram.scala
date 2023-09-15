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
val batch_size = 32 // how many independent sequences will we process in parallel?
val block_size = 8 // what is the maximum context length for predictions?
val max_iters = 3000
val eval_interval = 300
val learning_rate = 1e-2
//val device = 'cuda' if torch.cuda.is_available() else 'cpu'
val device = if torch.cuda.isAvailable then CUDA else CPU
//println(s"Using device: $device")
val eval_iters = 200
val n_embed = 32
val head_size = 16
// ------------


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

  //class BigramLanguageModel0(vocabSize: Int) extends nn.Module: 
  class BigramLanguageModel0(vocabSize: Int) extends BigramLanguageModel: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = nn.Embedding(vocabSize, vocabSize)

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =

      // idx and targets are both (B,T) tensor of integers
      val logits = token_embedding_table( idx ) // (B,T,C)

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
        val (logits_t, loss) = apply(idx)
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
  
  val m0 = BigramLanguageModel0(vocab_size)
  val (logits3, loss3) = m0(xb, yb)
  println(s"batch_size * block_size = ${batch_size * block_size}")
  println(s"logits.shape = ${logits3.shape}")
  println(s"loss=${loss3.item}")    
  
  val next1 = m0.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64), max_new_tokens=100)(0)
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

  val next2 = m0.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64), max_new_tokens=500)(0)
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

  // Create a model
  val m1 = BigramLanguageModel0(vocab_size)
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

  val next3 = m1.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64), max_new_tokens=500)(0)
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
    val token_embedding_table = nn.Embedding(vocabSize, nEmbed)
    val lm_head = nn.Linear(nEmbed, vocabSize)

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
        val (logits_t, loss) = apply(idx)
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
  val m2 = BigramLanguageModel1(vocab_size, n_embed)
  // create a PyTorch optimizer
  val optimizer2 = torch.optim.AdamW(m2.parameters, lr=1e-3)

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

  val next4 = m2.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64), max_new_tokens=500)(0)
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
    val token_embedding_table = nn.Embedding(vocabSize, nEmbed)
    val position_embedding_table = nn.Embedding(blockSize, nEmbed)
    val lm_head = nn.Linear(nEmbed, vocabSize)

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =
      val Seq(b,t) = idx.shape

      // idx and targets are both (B,T) tensor of integers
      // idx is (B,T)
      val token_embed = token_embedding_table( idx ) // (B,T,C) where C is nEmbed
      // positions of tokens
      val pos = torch.arange(0L,t, device=device) // (T) were T is the block size?
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
        val (logits_t, loss) = apply(idx)
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
  val m3 = BigramLanguageModel2(vocab_size, block_size, n_embed)
  // create a PyTorch optimizer
  val optimizer3 = torch.optim.AdamW(m3.parameters, lr=1e-3)

  for iter <- 0 until 10 //max_iters
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

  val next5 = m3.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64), max_new_tokens=500)(0)
  val decoded5 = decode(next5.toSeq)
  println(s"decode 5:'$decoded5'")


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
      // batch, number of time steps, channels
      val Seq(b,t,c) = x.shape
      assert(block_size == t, "Block size must be equal to time step")

      val k = key(x)   // (B,T,C)
      val q = query(x) // (B,T,C)
      // compute attention scores ("affinities")
      val qk = q `@` k.transpose(-2, -1) * Tensor(c).pow(-0.5).to(dtype=q.dtype)  // (B, T, C) @ (B, C, T) -> (B, T, T)
      // val mask = qk.maskedFill(tril == 0, Float.NegativeInfinity) // (B, T, T)
      val mask = qk.maskedFill(tril((º`:`n), (º`:`n)) == 0, Float.NegativeInfinity).to(dtype=q.dtype) // (B, T, T)
      // val softmax = F.softmax(mask, dim= -1) // (B, T, T)
      // val wei = dropout(softmax)
      val wei = F.softmax(mask, dim= -1) // (B, T, T)
      // perform the weighted aggregation of the values
      val v = value(x) // (B,T,C)
      val out = wei `@` v // (B, T, T) @ (B, T, C) -> (B, T, C)
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
    val sa_head = Head1(n_embed = nEmbed, head_size = vocabSize, block_size = blockSize) //, drop = 0.5)
    val lm_head = nn.Linear(nEmbed, vocabSize)

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =
      val Seq(b,t) = idx.shape

      // idx and targets are both (B,T) tensor of integers
      // idx is (B,T)
      val token_embed = token_embedding_table( idx ) // (B,T,C) where C is nEmbed
      // positions of tokens
      val pos = torch.arange(0L,t, device=device) // (T) were T is the block size?
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
        // crop idx to the last block_size tokens
        val idx_cond = idx(`:`, -blockSize, `:`)
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
  // create a PyTorch optimizer
  val optimizer4 = torch.optim.AdamW(m4.parameters, lr=1e-3)

  for iter <- 0 until 5000 //max_iters
  do
    // every once in a while evaluate the loss on train and val sets
    if (iter % eval_interval == 0) || (iter == max_iters - 1)
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

  val next6 = m4.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64), max_new_tokens=500)(0)
  val decoded6 = decode(next6.toSeq)
  println(s"decode 6:'$decoded6'")

  
  def main(args: Array[String]): Unit =
    ()

end BiGram

// 
//   class NeuralNetwork extends nn.Module:
//     val flatten = nn.Flatten()
//     val linearReluStack = register(nn.Sequential(
//       nn.Linear(28*28, 512),
//       nn.ReLU(),
//       nn.Linear(512, 512),
//       nn.ReLU(),
//       nn.Linear(512, 10),
//     ))
//     
//     def apply(x: Tensor[Float32]) =
//       val flattened = flatten(x)
//       val logits = linearReluStack(flattened)
//       logits
// 
  // val o = get_batch("train")
  // resnet.sala [200]
  // val model = BigramLanguageModel(vocabSize = vocabSize)
  // val loss = estimate_loss(model)

//   class BiGram():
// 
//     def main(args: Array[String]): Unit =
//       ()
// 
//   end BiGram



/*
torch.randint returns Tensor[None]

I am trying to replicate a Python script from a Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY). In this script he has the following code: 

```Python
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

When I code the `randint` in Scala I get:

```Scala
def get_batch(split: "train" | "val") = 
    // generate a small batch of data of inputs x and targets y
    val data = if split == "train" then train_data else val_data
    val ix = torch.randint(0, len(data) - block_size, Seq(batch_size))
```

Unfortunately the compiler reports:

```Scala
    val ix: Tensor[Nothing] = torch.randint(0, len(data) - block_size, Seq(batch_size))
```

Note that `len` returns an `Int` and the last parameter is a `Seq[Int]`. 

On inspection, the code I find is:

```Scala
  def randint(low: Long, high: Int, size: Seq[Int]) =
    // TODO Handle Optional Generators properly
    val generator = new org.bytedeco.pytorch.GeneratorOptional()
    Tensor(
      torchNative.torch_randint(low, high, size.toArray.map(_.toLong), generator)
    )
```

The IDE shows a `Tensor[Nothing]`. And that is as far as I got. I cannot seem get into the ByeDeco code (native?). 

Minor quibble, I noticed that the parameters `low` and high are both long, however the Storch parameters have the `high` as an `Int`. Better to use a `Long`?

I then found the unit tests with:

```Scala
    val randintTensor = randint(low, high + 1, Seq(100000)).to(dtype = float32)
```

The use of the sequence is surprising because in the [torch.randint documentation](https://pytorch.org/docs/stable/generated/torch.randint.html#torch.randint) the `size` parameter is a tuple that holds the size (dimensions) of the Tensor. Here are the examples.


```Python
>>> torch.randint(3, 5, (3,))
tensor([4, 3, 4])


>>> torch.randint(10, (2, 2))
tensor([[0, 2],
        [5, 5]])


>>> torch.randint(3, 10, (2, 2))
tensor([[4, 5],
        [6, 7]])
```

The other point is that the C++ call has the [following signature](https://github.com/pytorch/pytorch/blob/0dc7f6df9d00428cd175018e2bf9b45a8ec39b9c/aten/src/ATen/native/TensorFactories.cpp#L848):


```C++
Tensor randint(int64_t high, IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  return native::randint(high, size, c10::nullopt /* generator*/, dtype, layout, device, pin_memory);
}
```
  
I am assuming that `IntArrayRef` is the 
So my questons are: 

1. Can we avoid the need to create a sequence that is nor used anywere else?
1. Any way to create the Tensor with the desired shape?





*/