package gpt

// cSpell: ignore gpt, hyperparameters, logits, softmax
// cSpell: ignore CUDA, torchvision
// cSpell: ignore dtype
// cSpell: ignore stoi, itos
// cSpell: ignore nn, probs

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
  val stoi = chars.zipWithIndex.map((ch, i) => ch -> i).toMap
  val itos = stoi.map((ch,i) => i -> ch)
  def encode(s: String) = s.map( c => stoi(c) )
  def decode(s: Seq[Int]) = s.map( i => itos(i) ).mkString

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
  
  class BigramLanguageModel(vocabSize: Int) extends nn.Module: 

    // each token directly reads off the logits for the next token from a lookup table
    val token_embedding_table = nn.Embedding(vocabSize, vocabSize)

    def forward(idx: Tensor[Int64], targets: Option[Tensor[Int64]] = None) =

      // idx and targets are both (B,T) tensor of integers
      val logits = token_embedding_table( idx ) // (B,T,C)

      if targets.isEmpty
      then
        (logits, torch.Tensor(0.0f))
      else
        val shape = logits.shape
        val (b,t,c) = (shape(0), shape(1), shape(2))
        val logitsV = logits.view(b*t, c)  // batch size x number of classes b*t = 32*8 = 256
        val targetsV = targets.get.view(b*t) // 256
        val loss = F.crossEntropy(logitsV, targetsV)
        (logitsV, loss)


/*
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
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
    def generate(idx: Tensor[Int64], max_new_tokens: Int) =
      var idx_ = idx.copy_(idx)
      // idx is (B, T) array of indices in the current context
      for i <- 0 until max_new_tokens
      do
        println(i)
        // get the predictions
        val (logits_t, loss) = apply(idx)
        println(s"\t${logits_t.shape}")
        // focus only on the last time step
        val logits = logits_t(Slice(), -1, Slice()) // becomes (B, C)
        println(s"\t${logits.shape}")
        // apply softmax to get probabilities
        val probs = F.softmax(logits, dim = -1L) // (B, C)
        println(s"\t${probs.shape}")
        // sample from the distribution
        val idx_next = torch.multinomial(probs, numSamples=1) // (B, 1)
        println(s"\t${idx_next.shape}")
        println(s"next = \t${idx_next.item}")
        // append sampled index to the running sequence
        idx_ = torch.cat(Seq(idx_, idx_next), dim=1) // (B, T+1)
        println(s"\t$idx_")
      ???

    def apply(x: Tensor[Int64], y: Tensor[Int64]) =
      //forward(x, Some(y.to(dtype = DType.float32)) )
      forward(x, Some(y) )

    def apply(x: Tensor[Int64]) =
      forward(x, None )


  end BigramLanguageModel

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

  val m = BigramLanguageModel(vocab_size)
  val (logits, loss) = m(xb, yb)
  println(s"batch_size * block_size = ${batch_size * block_size}")
  println(s"logits.shape = ${logits.shape}")
  println(s"loss=${loss.item}")    
  
  val next = m.generate(idx = torch.zeros(Seq(1, 1), dtype=torch.int64), max_new_tokens=100)



  // https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
  // TODO: @torch.no_grad()
  def estimate_loss(model: BigramLanguageModel) = 
    val out = scala.collection.mutable.Map[String, Float]()
    model.eval()
    for 
      split <- List("train", "val")
    do
      println(s"Estimate '$split' loss")
      val losses: Tensor[Float32] = torch.zeros(eval_iters)
      for 
        k <- 0 until eval_iters
      do
        val (x, y) = get_batch(split)
        val (logits, loss) = model(x, y)
        // TODO: no assignment operator available
        losses(k) += loss.item
      out(split) = losses.mean.item
    model.train()
    out


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