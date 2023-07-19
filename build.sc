// cSpell:ignore sbt
// cSpell:ignore javac
// cSpell:ignore javacpp, sourcecode
// cSpell:ignore Agg
// cSpell:ignore redist, mkl, cuda, cudann, openblas
// cSpell:ignore progressbar, progressbar, munit, munit-scalacheck, scrimage


import $ivy.`com.lihaoyi::mill-contrib-bloop:`
// Import JavaCPP to get host OS name
import $ivy.`org.bytedeco:javacpp:1.5.9`


import mill._
import mill.scalalib._
import mill.contrib.bloop.Bloop

val scrImageVersion = "4.0.34"
val pytorchVersion = "2.0.1"
val cudaVersion = "12.1-8.9"
val openblasVersion = "0.3.23"
val mklVersion = "2023.1"
val ScalaVersion = "3.3.0"
val javaCppVersion = "1.5.9"

val mUnitVersion        = "1.0.0-M6" // "1.0.0-M3" https://mvnrepository.com/artifact/org.scalameta/munit
val ivyMunit          = ivy"org.scalameta::munit::$mUnitVersion"
val ivyMunitInterface = "munit.Framework"


val enableGPU = false

// https://gitlab.com/hmf/stensorflow/-/blob/main/build.sc
// object core extends SbtModule with Bloop.Module {
// https://github.com/scalameta/metals-vscode/issues/1403
trait CommonSettings extends SbtModule with Bloop.Module {
  def scalaVersion = ScalaVersion

  // List((pytorch,2.0.1), (mkl,2023.1), (openblas,0.3.23))
  val javaCppPresetLibs = Seq(
    (if (enableGPU) "pytorch-gpu" else "pytorch") -> pytorchVersion,
    "mkl" -> mklVersion,
    "openblas" -> openblasVersion
    // TODO remove cuda (not cuda-redist) once https://github.com/bytedeco/javacpp-presets/issues/1376 is fixed
    ) ++ (if (enableGPU) Seq("cuda-redist" -> cudaVersion, "cuda" -> cudaVersion) else Seq())

  println(javaCppPresetLibs)

  // val javaCppPlatform = org.bytedeco.sbt.javacpp.Platform.current
  def javaCPPPlatform = T{ org.bytedeco.javacpp.Loader.Detector.getPlatform }

  def ivyDeps = Agg(
      // https://github.com/bytedeco/javacpp-presets/tree/master/pytorch
      //ivy"org.bytedeco::pytorch:$pytorchVersion-${javaCppVersion};classifier=${javaCPPPlatform()}",
      ivy"org.bytedeco:pytorch:$pytorchVersion-${javaCppVersion};classifier=${javaCPPPlatform()}",
      ivy"org.bytedeco:pytorch-platform:$pytorchVersion-${javaCppVersion}",
      // Additional dependencies required to use CUDA, cuDNN, and NCCL
      ivy"org.bytedeco:pytorch-platform-gpu:$pytorchVersion-${javaCppVersion}",
      // Additional dependencies to use bundled CUDA, cuDNN, and NCCL
      ivy"org.bytedeco:cuda-platform-redist:$cudaVersion-${javaCppVersion}",
      // Additional dependencies to use bundled full version of MKL
      ivy"org.bytedeco:mkl-platform-redist:$mklVersion-${javaCppVersion}",
      ivy"org.typelevel::spire:0.18.0",
      ivy"org.typelevel::shapeless3-typeable:3.3.0",
      ivy"com.lihaoyi::os-lib:0.9.1",
      ivy"com.lihaoyi::sourcecode:0.3.0",
      ivy"dev.dirs:directories:26"
  )

}

trait TestCommonSettings extends TestModule {
    def ivyDeps = Agg(
        ivy"org.scalameta::munit:0.7.29",
        ivy"org.scalameta::munit-scalacheck:0.7.29"
      )
    def testFramework = "munit.Framework"
}

object core extends CommonSettings {

  object test extends SbtModuleTests with TestCommonSettings
}

object vision extends SbtModule with CommonSettings {
   def moduleDeps = Seq(core)

  // def ivyDeps = super.ivyDeps() ++ Agg(
  def ivyDeps = super[CommonSettings].ivyDeps() ++ Agg(
      ivy"com.sksamuel.scrimage:scrimage-core:$scrImageVersion",
      ivy"com.sksamuel.scrimage:scrimage-webp:$scrImageVersion",
    )

  object test extends SbtModuleTests with TestCommonSettings
}


object examples extends CommonSettings {
   def moduleDeps = Seq(vision)
  def ivyDeps = super.ivyDeps() ++ Agg(
    ivy"me.tongfei:progressbar:0.9.5",
    ivy"com.github.alexarchambault::case-app:2.1.0-M24",
    ivy"org.scala-lang.modules::scala-parallel-collections:1.0.4"
    )

  object test extends SbtModuleTests with TestCommonSettings
}



/*
hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/storch$ ./millw --mill-version 0.11.1 --import ivy:com.lihaoyi::mill-contrib-bloop: mill.contrib.Bloop/install
[build.sc] [44/49] cliImports 
Cannot resolve external module mill.contrib.Bloop

hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/storch$ ./mill --mill-version 0.11.1 --import ivy:com.lihaoyi::mill-contrib-bloop: mill.contrib.Bloop/install
[build.sc] [41/49] compile 
[info] compiling 1 Scala source to /mnt/ssd2/hmf/VSCodeProjects/storch/out/mill-build/compile.dest/classes ...
[info] done compiling
[build.sc] [49/49] scriptImportGraph 
Cannot resolve --mill-version. Try `mill resolve _` to see what's available.

hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/storch$ ./mill --import ivy:com.lihaoyi::mill-contrib-bloop: mill.contrib.Bloop/install
[build.sc] [41/49] compile 
[info] compiling 1 Scala source to /mnt/ssd2/hmf/VSCodeProjects/storch/out/mill-build/compile.dest/classes ...
[info] done compiling
[build.sc] [49/49] scriptImportGraph 
Cannot resolve external module mill.contrib.Bloop

./millw --mill-version 0.11.1 --import ivy:com.lihaoyi::mill-contrib-bloop: mill.contrib.Bloop/install
./mill --import ivy:com.lihaoyi::mill-contrib-bloop: mill.contrib.Bloop/install
./mill --import ivy:com.lihaoyi::mill-contrib-bloop:  mill.contrib.bloop.Bloop/install


hmf@gandalf:/mnt/ssd2/hmf/VSCodeProjects/storch$ ./mill --debug mill.bsp.BSP/install
[build.sc] [5/49] mill.scalalib.ZincWorkerModule.zincLogDebug 
Closing previous worker: mill.scalalib.ZincWorkerModule.worker
[build.sc] [41/49] compile 
[debug] [zinc] IncrementalCompile -----------
[debug] IncrementalCompile.incrementalCompile
[debug] previous = Stamps for: 6 products, 1 sources, 4 libraries
[debug] current source = Set(/mnt/ssd2/hmf/VSCodeProjects/storch/out/mill-build/generateScriptSources.dest/millbuild/build.sc)
[debug] > initialChanges = InitialChanges(Changes(added = Set(), removed = Set(), changed = Set(), unmodified = ...),Set(),Set(),API Changes: Set())
[debug] No changes
[2/2] mill.bsp.BSP.install 
Overwriting BSP connection file: /mnt/ssd2/hmf/VSCodeProjects/storch/.bsp/mill-bsp.json
Enabled debug logging for the BSP server. If you want to disable it, you need to re-run this install command without the --debug option.

*/
