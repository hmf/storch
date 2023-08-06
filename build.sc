// cSpell:ignore storch, sbt, sonatype, laika, epub
// cSpell:ignore javac
// cSpell:ignore javacpp, sourcecode
// cSpell:ignore Agg
// cSpell:ignore redist, mkl, cuda, cudann, openblas
// cSpell:ignore progressbar, progressbar, munit, munit-scalacheck, scrimage

import $ivy.`org.typelevel::cats-effect:3.5.1`
// Laika core, EPUB and PDF
import $ivy.`org.planet42::laika-core:0.19.3`
import $ivy.`org.planet42::laika-io:0.19.3`
import $ivy.`org.planet42::laika-pdf:0.19.3`
//import $ivy.`org.scalameta::mdoc:2.3.7`

import $ivy.`com.lihaoyi::mill-contrib-bloop:`
// Import JavaCPP to get host OS name
import $ivy.`org.bytedeco:javacpp:1.5.9`


import mill._
import mill.scalalib._
import mill.define.ModuleRef
import coursier.maven.MavenRepository
import mill.contrib.bloop.Bloop


import java.util.Locale
import java.util.concurrent.Executors
import scala.concurrent.ExecutionContext
import scala.concurrent._
import ExecutionContext.Implicits.global

import cats.effect._
import cats.syntax.all._
import cats.implicits._
import cats.effect.IO._
//import cats.effect.IOInstances
import cats.effect.{IO, Clock}
import scala.concurrent.ExecutionContext
import scala.concurrent.duration._

// import cats.effect.IO
// import cats.effect.{ IO, IOApp, ExitCode }
// import cats.effect.{ Async, Resource }

import laika.api._
import laika.format._
import laika.io.implicits._
import laika.markdown.github.GitHubFlavor
import laika.parse.code.SyntaxHighlighting
import laika.io.api.TreeTransformer


import cats.effect.{ Async, Resource }
import laika.io.api.TreeTransformer
import laika.markdown.github.GitHubFlavor
import laika.io.model.RenderedTreeRoot


object StorchSitePlugin {
  val transformer = Transformer.from(Markdown)
                                  .to(HTML)
                                  .using(
                                    GitHubFlavor,
                                    SyntaxHighlighting
                                  )
                                  .build

// https://github.com/typelevel/Laika/discussions/235   
// https://github.com/search?q=repo%3Acom-lihaoyi%2Fmill%20cats&type=code

// def createTransformer[F[_]: Async: ContextShift]
//                       (blocker: Blocker): Resource[F, TreeTransformer[F]] =
//   Transformer
//     .from(Markdown)
//     .to(HTML)
//     .using(GitHubFlavor, SyntaxHighlighting)
//     .io(blocker)
//     .parallel[F]
//     .build                                  
  // https://github.com/typelevel/cats-effect/issues/280
  // https://typelevel.org/Laika/latest/02-running-laika/02-library-api.html
  def createTransformer[F[_]: Async]: Resource[F, TreeTransformer[F]] =
    Transformer
      .from(Markdown)
      .to(HTML)
      .using(GitHubFlavor)
      .parallel[F]
      .build


}

object docs extends CommonSettings {

  def laika : T[PathRef] = T {

      val cp = runClasspath().map(_.path)

      val dir = T.dest.toIO.getAbsolutePath
      println(dir)
      // val dirParams = mdocSources().map(pr => Seq(s"--in", pr.path.toIO.getAbsolutePath, "--out",  dir)).iterator.flatten.toSeq

      // Jvm.runLocal("mdoc.Main", cp, dirParams)


      val result1 = StorchSitePlugin.transformer.transform("hello *there*")
      println(result1)

      val result: IO[RenderedTreeRoot[IO]] = StorchSitePlugin.createTransformer[IO].use {
        t =>
          t.fromDirectory("docs")
          .toDirectory("target")
          .transform
      }

      PathRef(T.dest)
    }

}


val scrImageVersion = "4.0.34"
val pytorchVersion = "2.0.1"   //  "1.12.1" (1.12.1-1.5.8), "2.0.1" (2.0.1-1.5.9)
val cudaVersion =  "12.1-8.9" //  (11.8-8.6-1.5.8), "12.1-8.9" (12.1-8.9-1.5.9)
val openblasVersion = "0.3.23"
val mklVersion = "2023.1"
val ScalaVersion = "3.3.0"
val javaCppVersion = "1.5.10-SNAPSHOT" // "1.5.8", "1.5.9"
// ThisBuild / resolvers ++= Resolver.sonatypeOssRepos("snapshots")

val mUnitVersion        = "1.0.0-M6" // "1.0.0-M3" https://mvnrepository.com/artifact/org.scalameta/munit
val ivyMunit          = ivy"org.scalameta::munit::$mUnitVersion"
val ivyMunitInterface = "munit.Framework"

val enableGPU = false

val sonatypeReleases = Seq(
  MavenRepository("https://oss.sonatype.org/content/repositories/releases")
)
val sonatypeSnapshots = Seq(
  MavenRepository("https://oss.sonatype.org/content/repositories/snapshots")
)


// https://gitlab.com/hmf/stensorflow/-/blob/main/build.sc
// object core extends SbtModule with Bloop.Module {
// https://github.com/scalameta/metals-vscode/issues/1403
trait CommonSettings extends SbtModule with Bloop.Module {
  def scalaVersion = ScalaVersion

  // TODO: add to scaladoc
  // def scalacOptions = Seq("-groups", "-snippet-compiler:compile")

  // TODO: Not used
  // List((pytorch,2.0.1), (mkl,2023.1), (openblas,0.3.23))
  val javaCppPresetLibs = Seq(
    (if (enableGPU) "pytorch-gpu" else "pytorch") -> pytorchVersion,
    "mkl" -> mklVersion,
    "openblas" -> openblasVersion
    )

  def repositoriesTask = T.task {
    super.repositoriesTask() ++ sonatypeSnapshots
  }    

  // val javaCppPlatform = org.bytedeco.sbt.javacpp.Platform.current
  def javaCPPPlatform = T{ org.bytedeco.javacpp.Loader.Detector.getPlatform }

  def ivyDeps = Agg(
      // https://github.com/bytedeco/javacpp-presets/tree/master/pytorch
      ivy"org.bytedeco:pytorch:$pytorchVersion-${javaCppVersion};classifier=${javaCPPPlatform()}",
      ivy"org.bytedeco:pytorch-platform:$pytorchVersion-${javaCppVersion}",
      // // Additional dependencies required to use CUDA, cuDNN, and NCCL
      // ivy"org.bytedeco:pytorch-platform-gpu:$pytorchVersion-${javaCppVersion}",
      // // Additional dependencies to use bundled CUDA, cuDNN, and NCCL
      // ivy"org.bytedeco:cuda-platform-redist:$cudaVersion-${javaCppVersion}",
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
   def forkArgs = Seq("-Djava.awt.headless=true")
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
