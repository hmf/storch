/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 // cSpell:ignore CUDA, MNIST
 // cSpell:ignore sonatype, progressbar, redist
 
 package commands

//> using scala "3.3"
//> using repository "sonatype-s01:snapshots"
//> using lib "dev.storch::vision:0.0-3e0f9b1-SNAPSHOT"
//> using lib "me.tongfei:progressbar:0.9.5"
//> using lib "com.github.alexarchambault::case-app:2.1.0-M24"
//> using lib "org.scala-lang.modules::scala-parallel-collections:1.0.4"
// replace with pytorch-platform-gpu if you have a CUDA capable GPU
//> using lib "org.bytedeco:pytorch-platform:2.0.1-1.5.9"
// enable for CUDA support
////> using lib "org.bytedeco:cuda-platform:12.1-8.9-1.5.9"
////> using lib "org.bytedeco:cuda-platform-redist:12.1-8.9-1.5.9"

import ImageClassifier.{Prediction, predict, train}
import caseapp.*
import caseapp.core.argparser.{ArgParser, SimpleArgParser}
import caseapp.core.app.CommandsEntryPoint
import com.sksamuel.scrimage.{ImmutableImage, ScaleMethod}
import me.tongfei.progressbar.{ProgressBar, ProgressBarBuilder}
import org.bytedeco.javacpp.PointerScope
import org.bytedeco.pytorch.{InputArchive, OutputArchive}
import os.Path
import torch.*
import torch.Device.{CPU, CUDA}
import torch.optim.Adam
import torchvision.models.resnet.{ResNet, ResNetVariant}

import java.nio.file.Paths
import scala.collection.parallel.CollectionConverters.ImmutableSeqIsParallelizable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.DurationInt
import scala.concurrent.{Await, Future}
import scala.util.{Random, Try, Using}

/** Example script for training an image-classification model on your own images */
object ImageClassifier extends CommandsEntryPoint:

  val fileTypes = Seq("jpg", "png")

  case class Metrics(loss: Float, accuracy: Float)

  torch.manualSeed(0)
  val random = new Random(seed = 0)

  extension (number: Double) def format: String = "%1.5f".format(number)

  def train(options: TrainOptions): Unit =
    val device = if torch.cuda.isAvailable then CUDA else CPU
    println(s"Using device: $device")

    val datasetDir = os.Path(options.datasetDir, base = os.pwd)

    /** verify that we can read all images of the dataset */
    if os
        .walk(datasetDir)
        .filter(path => fileTypes.contains(path.ext))
        .par
        .map { path =>
          val readTry = Try(ImmutableImage.loader().fromPath(path.toNIO)).map(_ => ())
          readTry.failed.foreach(e => println(s"Could not read $path"))
          readTry
        }
        .exists(_.isFailure)
    then
      println("Could not read all images in the dataset. Stopping.")
      System.exit(1)
    val classes = os.list(datasetDir).filter(os.isDir).map(_.last).sorted
    val classIndices = classes.zipWithIndex.toMap
    println(s"Found ${classIndices.size} classes: ${classIndices.mkString("[", ", ", "]")}")
    val pathsPerLabel = classes.map { label =>
      label -> os
        .list(datasetDir / label)
        .filter(path => fileTypes.contains(path.ext))
    }.toMap
    val pathsWithLabel =
      pathsPerLabel.toSeq.flatMap((label, paths) => paths.map(path => path -> label))
    println(s"Found ${pathsWithLabel.size} examples")
    println(
      pathsPerLabel
        .map((label, paths) => s"Found ${paths.size} examples for class $label")
        .mkString("\n")
    )

    val sample = random.shuffle(pathsWithLabel).take(options.take.getOrElse(pathsWithLabel.length))
    val (trainData, testData) = sample.splitAt((sample.size * 0.9).toInt)
    println(s"Train size: ${trainData.size}")
    println(s"Eval size:  ${testData.size}")

    val model: ResNet[Float32] = options.baseModel.factory(numClasses = classes.length)
    println(s"Model architecture: ${options.baseModel}")
    val transforms = options.baseModel.factory.DEFAULT.transforms

    if options.pretrained then
      println(s"Loading pre-trained model from: ${options.baseModel.factory.DEFAULT.url}")
      val weights = torch.hub.loadStateDictFromUrl(options.baseModel.factory.DEFAULT.url)
      // Don't load the classification head weights, as we they are specific to the imagenet classes
      // and their output size (1000) usually won't match the number of classes of our dataset.
      model.loadStateDict(
        weights.filterNot((k, v) => Set("fc.weight", "fc.bias").contains(k))
      )
    model.to(device)

    val optimizer = Adam(model.parameters, lr = options.learningRate)
    val lossFn = torch.nn.loss.CrossEntropyLoss()
    val numEpochs = options.epochs
    val batchSize = options.batchSize
    val trainSteps = (trainData.size / batchSize.toFloat).ceil.toInt
    val evalSteps = (testData.size / options.batchSize.toFloat).ceil.toInt

    // Lazily loads inputs and transforms them into batches of tensors in the shape the model expects.
    def dataLoader(
        dataset: Seq[(Path, String)],
        shuffle: Boolean,
        batchSize: Int
    ): Iterator[(Tensor[Float32], Tensor[Int64])] =
      val loader = ImmutableImage.loader()
      (if shuffle then random.shuffle(dataset) else dataset)
        .grouped(batchSize)
        .map { batch =>
          val (inputs, labels) = batch.unzip
          // parallelize loading to improve GPU utilization
          val transformedInputs =
            inputs.par.map(path => transforms.transforms(loader.fromPath(path.toNIO))).seq
          assert(transformedInputs.forall(t => !t.isnan.any.item))
          (
            transforms.batchTransforms(torch.stack(transformedInputs)),
            torch.stack(labels.map(label => Tensor(classIndices(label)).to(dtype = int64)))
          )
        }

    def trainDL = dataLoader(trainData, shuffle = true, batchSize)

    def evaluate(): Metrics =
      val testDL = dataLoader(testData, shuffle = false, batchSize = batchSize)
      val evalPB =
        ProgressBarBuilder().setTaskName(s"Evaluating        ").setInitialMax(evalSteps).build()
      evalPB.setExtraMessage(s" " * 36)
      val isTraining = model.isTraining
      if isTraining then model.eval()
      val (loss, correct) = testDL
        .map { (inputBatch, labelBatch) =>
          Using.resource(new PointerScope()) { p =>
            val pred = model(inputBatch.to(device))
            val label = labelBatch.to(device)
            val loss = lossFn(pred, label).item
            val correct = pred.argmax(dim = 1).eq(label).sum.item
            evalPB.step()
            (loss, correct)
          }
        }
        .toSeq
        .unzip
      val metrics = Metrics(
        Tensor(loss).mean.item,
        Tensor(correct).sum.item / testData.size.toFloat
      )
      evalPB.setExtraMessage(
        s"    Loss: ${metrics.loss.format}, Accuracy: ${metrics.accuracy.format}"
      )
      evalPB.close()
      if isTraining then model.train()
      metrics

    for epoch <- 1 to numEpochs do
      val trainPB = ProgressBarBuilder()
        .setTaskName(s"Training epoch $epoch/$numEpochs")
        .setInitialMax(trainSteps)
        .build()
      var runningLoss = 0.0
      var step = 0
      var evalMetrics: Metrics = Metrics(Float.NaN, accuracy = 0)
      for (input, label) <- trainDL do {
        optimizer.zeroGrad()
        // Using PointerScope ensures that all intermediate tensors are deallocated in time
        Using.resource(new PointerScope()) { p =>
          val pred = model(input.to(device))
          val loss = lossFn(pred, label.to(device))
          loss.backward()
          // add a few sanity checks
          assert(
            model.parameters.forall(t => !t.isnan.any.item),
            "Parameters containing nan values"
          )
          assert(
            model.parameters.forall(t => !t.grad.isnan.any.item),
            "Gradients containing nan values"
          )
          optimizer.step()
          runningLoss += loss.item
        }
        trainPB.setExtraMessage(" " * 21 + s"Loss: ${(runningLoss / step).format}")
        trainPB.step()
        if ((step + 1) % (trainSteps / 4.0)).toInt == 0 then
          evalMetrics = evaluate()
          runningLoss = 0.0
        step += 1
      }
      trainPB.close()
      println(
        s"Epoch $epoch/$numEpochs, Training loss: ${(runningLoss / step).format}, Evaluation loss: ${evalMetrics.loss.format}, Accuracy: ${evalMetrics.accuracy.format}"
      )
      val checkpointDir = os.Path(options.checkpointDir, os.pwd) / "%02d".format(epoch)
      os.makeDir.all(checkpointDir)
      val oa = OutputArchive()
      model.to(CPU).save(oa)
      oa.save_to((checkpointDir / "model.pt").toString)
      os.write(checkpointDir / "model.txt", options.baseModel.toString())
      os.write(checkpointDir / "classes.txt", classes.mkString("\n"))

  case class Prediction(label: String, confidence: Double)

  def predict(options: PredictOptions): Prediction =
    val modelDir = options.modelDir match
      case None =>
        val checkpoints = os.list(os.pwd / "checkpoints", sort = true)
        if checkpoints.isEmpty then
          println("Not checkpoint found. Did you train a model?")
          System.exit(1)
        checkpoints.last
      case Some(value) => os.Path(value, os.pwd)
    println(s"Trying to load model from $modelDir")
    val classes = os.read.lines(modelDir / "classes.txt")
    val modelVariant =
      ResNetVariant.valueOf(os.read(modelDir / "model.txt"))
    val model: ResNet[Float32] = modelVariant.factory(numClasses = classes.length)
    val transforms = modelVariant.factory.DEFAULT.transforms
    val ia = InputArchive()
    ia.load_from((modelDir / "model.pt").toString)
    model.load(ia)
    model.eval()
    val image = ImmutableImage.loader().fromPath(Paths.get(options.imagePath))
    val transformedImage =
      transforms.batchTransforms(transforms.transforms(image)).unsqueeze(dim = 0)
    val prediction = model(transformedImage)
    val TensorTuple(confidence, index) =
      torch.nn.functional.softmax(prediction, dim = 1)().max(dim = 1)
    val predictedLabel = classes(index.item.toInt)
    Prediction(predictedLabel, confidence.item)

  override def commands: Seq[Command[?]] = Seq(Train, Predict)
  override def progName: String = "image-classifier"

implicit val customArgParser: ArgParser[ResNetVariant] =
  SimpleArgParser.string.xmap(_.toString(), ResNetVariant.valueOf)

@HelpMessage("Train an image classification model")
case class TrainOptions(
    @HelpMessage(
      "Path to images. Images are expected to be stored in one directory per class i.e. cats/cat1.jpg cats/cat2.jpg dogs/dog1.jpg ..."
    )
    datasetDir: String,
    @HelpMessage(
      s"ResNet variant to use. Possible values are: ${ResNetVariant.values.mkString(", ")}. Defaults to ResNet50."
    )
    baseModel: ResNetVariant = ResNetVariant.ResNet50,
    @HelpMessage("Load pre-trained weights for base-model")
    pretrained: Boolean = true,
    @HelpMessage("Where to save model checkpoints")
    checkpointDir: String = "checkpoints",
    @HelpMessage("The maximum number of images to take for training")
    take: Option[Int] = None,
    batchSize: Int = 8,
    @HelpMessage("How many epochs (iterations over the input data) to train")
    epochs: Int = 1,
    learningRate: Double = 1e-5
)
@HelpMessage("Predict which class an image belongs to")
case class PredictOptions(
    @HelpMessage("Path to an image whose class we want to predict")
    imagePath: String,
    @HelpMessage(
      "Path to to the serialized model created by running 'train'. Tries the latest model in 'checkpoints' if not set."
    )
    modelDir: Option[String]
)


/**
 * ./mill examples.runMain commands.Train -h
 * ./mill examples.runMain commands.Train --dataset-dir ./data/mnist --checkpoint-dir ../../tmp --pretrained=false
 * ./mill examples.runMain commands.Train --dataset-dir ./data/mnist --pretrained=false
 * ./mill examples.runMain commands.Train --dataset-dir ./data/mnist --pretrained=false --epochs 10
 * ./mill examples.runMain commands.Train --dataset-dir /mnt/ssd2/hmf/datasets/computer_vision/kaggle_cats_and_dogs/pet_images/ --pretrained=false --checkpoint-dir ../../tmp
 * ./mill examples.runMain commands.Train --dataset-dir /mnt/ssd2/hmf/datasets/computer_vision/kaggle_cats_and_dogs/pet_images --checkpoint-dir ~/.cache/storch/hub/checkpoints
 * 
 * 
 * @see https://github.com/sbrunk/storch/discussions/41
 * @see https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
 * @see https://www.microsoft.com/en-us/download/details.aspx?id=54765
 * 
 * Notre following files from Microsoft are corrupted (use Kaggle version):
 * - /kagglecatsanddogs_5340/pet_images/Cat/666.jpg
 * - /kagglecatsanddogs_5340/pet_images/Cat/10404.jpg
 * - /kagglecatsanddogs_5340/pet_images/Dog/11702.jpg
 * 
 * [114/114] examples.runMain 
Using device: Device(CUDA,-1)
SLF4J: No SLF4J providers were found.
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See https://www.slf4j.org/codes.html#noProviders for further details.
SLF4J: Class path contains SLF4J bindings targeting slf4j-api versions 1.7.x or earlier.
SLF4J: Ignoring binding found at [jar:file:/home/hmf/.cache/coursier/v1/https/repo1.maven.org/maven2/ch/qos/logback/logback-classic/1.1.2/logback-classic-1.1.2.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See https://www.slf4j.org/codes.html#ignoredBindings for an explanation.
Found 2 classes: [Cat -> 0, Dog -> 1]
Found 24959 examples
Found 12490 examples for class Cat
Found 12469 examples for class Dog
Train size: 22463
Eval size:  2496
Model architecture: ResNet50
Exception in thread "main" java.lang.RuntimeException: PytorchStreamReader failed reading zip archive: failed finding central directory
Exception raised from valid at /__w/javacpp-presets/javacpp-presets/pytorch/cppbuild/linux-x86_64-gpu/pytorch/caffe2/serialize/inline_container.cc:178 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x57 (0x7f87f51ae4d7 in /home/hmf/.javacpp/cache/pytorch-2.0.1-1.5.9-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x64 (0x7f87f517836b in /home/hmf/.javacpp/cache/pytorch-2.0.1-1.5.9-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libc10.so)
frame #2: caffe2::serialize::PyTorchStreamReader::valid(char const*, char const*) + 0x8e (0x7f85eecba99e in /home/hmf/.javacpp/cache/pytorch-2.0.1-1.5.9-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #3: caffe2::serialize::PyTorchStreamReader::init() + 0x9e (0x7f85eecbadfe in /home/hmf/.javacpp/cache/pytorch-2.0.1-1.5.9-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #4: caffe2::serialize::PyTorchStreamReader::PyTorchStreamReader(std::shared_ptr<caffe2::serialize::ReadAdapterInterface>) + 0x7f (0x7f85eecbc68f in /home/hmf/.javacpp/cache/pytorch-2.0.1-1.5.9-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #5: torch::jit::pickle_load(std::vector<char, std::allocator<char> > const&) + 0x15e (0x7f85efe1bbbe in /home/hmf/.javacpp/cache/pytorch-2.0.1-1.5.9-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libtorch_cpu.so)
frame #6: Java_org_bytedeco_pytorch_global_torch_pickle_1load___3B + 0xc9 (0x7f85ea5601a9 in /home/hmf/.javacpp/cache/pytorch-2.0.1-1.5.9-linux-x86_64-gpu.jar/org/bytedeco/pytorch/linux-x86_64-gpu/libjnitorch.so)
frame #7: [0x7f882094453a]

	at org.bytedeco.pytorch.global.torch.pickle_load(Native Method)
	at torch.ops.CreationOps.pickleLoad(CreationOps.scala:326)
	at torch.ops.CreationOps.pickleLoad$(CreationOps.scala:37)
	at torch.package$.pickleLoad(package.scala:32)
	at torch.ops.CreationOps.pickleLoad(CreationOps.scala:340)
	at torch.ops.CreationOps.pickleLoad$(CreationOps.scala:37)
	at torch.package$.pickleLoad(package.scala:32)
	at torch.hub$.loadStateDictFromUrl(hub.scala:40)
	at commands.ImageClassifier$.train(ImageClassifier2.scala:114)
	at commands.Train$.run(ImageClassifier2.scala:323)
	at commands.Train$.run(ImageClassifier2.scala:323)
	at caseapp.core.app.CaseApp.main(CaseApp.scala:162)
	at caseapp.core.app.CaseApp.main(CaseApp.scala:133)
	at commands.Train$.main(ImageClassifier2.scala:329)
	at commands.Train.main(ImageClassifier2.scala)
1 targets failed

 */
object Train extends Command[TrainOptions]:
  override def run(options: TrainOptions, remainingArgs: RemainingArgs): Unit = train(options)
  override def main(args: Array[String]): Unit = 
    if (args.isEmpty)
    then
      fullHelpAsked(finalHelp.progName)
    else 
      super.main(args)


/**
 * $ ./mill examples.runMain commands.Predict --dataset-dir ./data/mnist
 */
object Predict extends Command[PredictOptions]:
  override def run(options: PredictOptions, remainingArgs: RemainingArgs): Unit =
    val Prediction(label, confidence) = predict(options)
    println(s"Class: $label, confidence: $confidence")


/*
  object ResNet50 extends ResNetFactory:
    def apply[D <: BFloat16 | Float32 | Float64: Default](numClasses: Int = 1000) =
      ResNet(Bottleneck, Seq(3, 4, 6, 3), numClasses = numClasses)()
    val IMAGENET1K_V1 = Weights(
      url = weightsBaseUrl + "resnet50-0676ba61.pth",
      transforms = Presets.ImageClassification(cropSize = 224)
    )
    val IMAGENET1K_V2 = Weights(
      url = weightsBaseUrl + "resnet50-11ad3fa6.pth",
      transforms = Presets.ImageClassification(cropSize = 224, resizeSize = 232)
    )
    val DEFAULT = IMAGENET1K_V2

  private val weightsBaseUrl =
    "https://github.com/sbrunk/storch/releases/download/pretrained-weights/"

https://github.com/sbrunk/storch/releases/

https://github.com/sbrunk/storch/tree/pretrained-weights
https://github.com/sbrunk/storch/releases/download/pretrained-weights/resnet50-0676ba61.pth

https://github.com/sbrunk/storch/releases/download/pretrained-weights/resnet50-11ad3fa6.pth

https://stackoverflow.com/questions/71617570/pytorchstreamreader-failed-reading-zip-archive-failed-finding-central-directory

hmf@gandalf:~/.cache/storch/hub/checkpoints$ pwd
/home/hmf/.cache/storch/hub/checkpoints
hmf@gandalf:~/.cache/storch/hub/checkpoints$ ls
resnet50-11ad3fa6.pth
hmf@gandalf:~/.cache/storch/hub/checkpoints$ ls -lh
total 70M
-rw-rw-r-- 1 hmf hmf 70M jul 19 10:19 resnet50-11ad3fa6.pth
hmf@gandalf:~/.cache/storch/hub/checkpoints$ ls -lH
total 71016
-rw-rw-r-- 1 hmf hmf 72712658 jul 19 10:19 resnet50-11ad3fa6.pth


*/