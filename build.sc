// cSpell:ignore storch, sbt, sonatype, laika, epub
// cSpell:ignore javac
// cSpell:ignore javacpp, sourcecode
// cSpell:ignore Agg
// cSpell:ignore redist, mkl, cuda, cudann, openblas
// cSpell:ignore progressbar, progressbar, munit, munit-scalacheck, scrimage

import $ivy.`org.typelevel::cats-effect:3.5.1`
import mill.api.Loose
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
import mill.scalalib.publish.License
import mill.scalalib.publish.Developer
import mill.scalalib.publish.VersionControl
import mill.define.ModuleRef
import coursier.maven.MavenRepository
import mill.contrib.bloop.Bloop


import java.net.URL
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
import laika.io.model.RenderedTreeRoot


import cats.effect.{ Async, Resource }

import laika.helium.config._
import laika.ast.Path._
import laika.ast._

import laika.ast.LengthUnit._
import laika.ast._
import laika.helium.Helium
import laika.helium.config.Favicon
import laika.helium.config.HeliumIcon
import laika.helium.config.IconLink
import laika.helium.config.ImageLink
import laika.rewrite.nav.{ChoiceConfig, Selections, SelectionConfig}
import laika.rewrite.link.{ApiLinks, LinkConfig}


// https://github.com/typelevel/sbt-typelevel
// https://github.com/sbt/sbt
// https://github.com/sbt/librarymanagement


// Mill has a default top-level MainModule that already defines `version
// If you redefine it as a value (line below), an error will occur
// val version = "0.0"
// You can override it like this:
// override def version() = super.version()
// But it represents the Mill version
// val version_ = ""

val tlBaseVersion = "0.0" // your current series x.y
val version_ = s"$tlBaseVersion-SNAPSHOT"

val organization = "dev.storch"
val organizationName = "storch.dev"
val startYear = Some(2022)
val licenses = Seq(License.`Apache-2.0`)
val developers = List(
  // your GitHub handle and name
  tlGitHubDev("sbrunk", "Sören Brunk")
)

/**
 * Helper to create a `Developer` entry from a GitHub username.
 * TODO: remove, use variable
 */
def tlGitHubDev(user: String, fullName: String): Developer = {
  // TODO: org=storch, orgURL=storch.dev
  //Developer(id=user, name=fullName, url=s"https://github.com/$user", organization=None, organizationUrl=None)
  Developer(id=user, name=fullName, url=s"https://github.com/$user", organization=Some(organization), organizationUrl=Some(organizationName))
}



// publish to s01.oss.sonatype.org (set to true to publish to oss.sonatype.org instead)
val tlSonatypeUseLegacyHost = false

// publish website from this branch
val tlSitePublishBranch = Some("main")

val apiURL = Some(new URL("https://storch.dev/api/"))

// Some(s"https://github.com/$owner/$repo")
val scmInfo = VersionControl.github("sbrunk", "storch")



object StorchSitePlugin {

  //scmInfo.value.fold("https://github.com/sbrunk/storch")(_.browseUrl.toString),
  val browsableLink = scmInfo.browsableRepository.getOrElse("SCMInfo Missing")

  val tlSiteHeliumConfig = Helium.defaults.site
    .metadata(
      title = Some("Storch"),
      authors = developers.map(_.name),
      language = Some("en"),
      version = Some(version_)
    )
    .site
    .layout(
      contentWidth = px(860),
      navigationWidth = px(275),
      topBarHeight = px(50),
      defaultBlockSpacing = px(10),
      defaultLineHeight = 1.5,
      anchorPlacement = laika.helium.config.AnchorPlacement.Right
    )
    //        .site
    //        .favIcons(
    //          Favicon.external("https://typelevel.org/img/favicon.png", "32x32", "image/png")
    //        )
    .site
    .topNavigationBar(
      navLinks = Seq(
        IconLink.internal(
          Root / "api" / "index.html",
          HeliumIcon.api,
          options = Styles("svg-link")
        ),
        IconLink.external(
          browsableLink,
          HeliumIcon.github,
          options = Styles("svg-link")
        )
        //            IconLink.external("https://discord.gg/XF3CXcMzqD", HeliumIcon.chat),
        //            IconLink.external("https://twitter.com/typelevel", HeliumIcon.twitter)
      )
    )
    .site
    .landingPage(
      logo = Some(
        Image.internal(Root / "img" / "storch.svg", height = Some(Length(300, LengthUnit.px)))
      ),
      title = Some("Storch"),
      subtitle = Some("GPU Accelerated Deep Learning for Scala 3"),
      license = Some("Apache 2"),
      //          titleLinks = Seq(
      //            VersionMenu.create(unversionedLabel = "Getting Started"),
      //            LinkGroup.create(
      //              IconLink.external("https://github.com/abcdefg/", HeliumIcon.github),
      //              IconLink.external("https://gitter.im/abcdefg/", HeliumIcon.chat),
      //              IconLink.external("https://twitter.com/abcdefg/", HeliumIcon.twitter)
      //            )
      //          ),
      documentationLinks = Seq(
        TextLink.internal(Root / "about.md", "About"),
        TextLink.internal(Root / "installation.md", "Getting Started"),
        TextLink.internal(Root / "api" / "index.html", "API (Scaladoc)")
      ),
      projectLinks = Seq(
        IconLink.external(
          browsableLink,
          HeliumIcon.github,
          options = Styles("svg-link")
        )
      ),
      teasers = Seq(
        Teaser(
          "Build Deep Learning Models in Scala",
          """
            |Storch provides GPU accelerated tensor operations, automatic differentiation,
            |and a neural network API for building and training machine learning models.
            |""".stripMargin
        ),
        Teaser(
          "Get the Best of PyTorch & Scala",
          """
            |Storch aims to be close to the original PyTorch API, while still leveraging Scala's powerful type
            |system for safer tensor operations.
            |""".stripMargin
        ),
        Teaser(
          "Powered by LibTorch & JavaCPP",
          """
            |Storch is based on <a href="https://pytorch.org/cppdocs/">LibTorch</a>, the C++ library underlying PyTorch.
            |JVM bindings are provided by <a href="https://github.com/bytedeco/javacpp">JavaCPP</a> for seamless
            |interop with native code & CUDA support.
            |""".stripMargin
        )
      )
    )

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
      .withTheme(tlSiteHeliumConfig.build)
      .build


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

  override def repositoriesTask = T.task {
    super.repositoriesTask() ++ sonatypeSnapshots
  }    

  // val javaCppPlatform = org.bytedeco.sbt.javacpp.Platform.current
  def javaCPPPlatform = T{ org.bytedeco.javacpp.Loader.Detector.getPlatform }

  override def ivyDeps = Agg(
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
    override def ivyDeps = Agg(
        ivy"org.scalameta::munit:0.7.29",
        ivy"org.scalameta::munit-scalacheck:0.7.29"
      )
    def testFramework = "munit.Framework"
}

object core extends CommonSettings {

  // Only for site generation
  // See https://docs.scala-lang.org/scala3/guides/scaladoc/static-site.html
//  override def scalaDocOptions = {
//    super.scalaDocOptions()
//    //Seq("-siteroot", "mydocs", "-no-link-warnings")
//  }

  object test extends SbtModuleTests with TestCommonSettings
}

object vision extends SbtModule with CommonSettings {
   override def moduleDeps = Seq(core)

  // def ivyDeps = super.ivyDeps() ++ Agg(
  override def ivyDeps = super[CommonSettings].ivyDeps() ++ Agg(
      ivy"com.sksamuel.scrimage:scrimage-core:$scrImageVersion",
      ivy"com.sksamuel.scrimage:scrimage-webp:$scrImageVersion",
    )

  object test extends SbtModuleTests with TestCommonSettings
}


object examples extends CommonSettings {
  override def moduleDeps = Seq(vision)
  override def forkArgs = Seq("-Djava.awt.headless=true")
  override def ivyDeps = super.ivyDeps() ++ Agg(
    ivy"me.tongfei:progressbar:0.9.5",
    ivy"com.github.alexarchambault::case-app:2.1.0-M24",
    ivy"org.scala-lang.modules::scala-parallel-collections:1.0.4"
    )

  object test extends SbtModuleTests with TestCommonSettings
}



// https://mill-build.com/mill/Scala_Module_Config.html#_scaladoc_config
/*

docs.docJar
docs.docJarUseArgsFile
docs.docResources
docs.docSources
docs.javadocOptions
docs.scalaDocClasspath
docs.scalaDocOptions
docs.scalaDocPluginClasspath
docs.scalaDocPluginIvyDeps

hmf@gandalf:/mnt/ssd2/hmf/IdeaProjects/storch$ ./mill -i show docs.docJar
[build.sc] [41/49] compile
[info] compiling 1 Scala source to /mnt/ssd2/hmf/IdeaProjects/storch/out/mill-build/compile.dest/classes ...
[info] done compiling
[1/1] show > [43/43] docs.docJar
"ref:v0:75c9e461:/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/docJar.dest/out.jar"
hmf@gandalf:/mnt/ssd2/hmf/IdeaProjects/storch$ ./mill -i show docs.docResources
[1/1] show > [1/1] docs.docResources
[
  "ref:v0:c984eca8:/mnt/ssd2/hmf/IdeaProjects/storch/docs/docs"
]

"unresolved internal reference: api/index.html"

1. Add other module sources to "docs.docResources"
1. Set "scalaDocOptions" to:
   1. just generate the API
   1. Copy the results to the docs temporary folder /api folder
 */


object docs extends CommonSettings {
  override def moduleDeps = Seq(core, vision, examples)

  //override def scalaDocOptions = T{ Seq("-siteroot", "", "-no-link-warnings") }

  override def docResources: T[Seq[PathRef]] = T {
    core.docResources() ++
      vision.docResources() ++
      examples.docResources() ++
      super.docResources()
  }

  override def scalaDocClasspath: T[Loose.Agg[PathRef]] = T {
    core.scalaDocClasspath() ++
      vision.scalaDocClasspath() ++
      examples.scalaDocClasspath() ++
      super.scalaDocClasspath()
  }

  override def scalaDocPluginClasspath: T[Loose.Agg[PathRef]] = T {
    core.scalaDocPluginClasspath() ++
      vision.scalaDocPluginClasspath() ++
      examples.scalaDocPluginClasspath() ++
      super.scalaDocPluginClasspath()
  }

  // ef:v0:bcafb9d8:/mnt/ssd2/hmf/IdeaProjects/storch/out/vision/compile.dest/classes
  // ref:v0:23ec7aa2:/mnt/ssd2/hmf/IdeaProjects/storch/out/core/compile.dest/classes
  override def docSources: T[Seq[PathRef]] = T {
    /*
    T.log.info(core.docSources().mkString(","))
    T.log.info(vision.docSources().mkString(","))
    */

    core.docSources() ++
      vision.docSources() ++
      examples.docSources() ++
      super.docSources()
  }

  // where do the mdoc sources live ?
  def laikaSources = T.sources {
    super.millSourcePath
  }

  def laika : T[PathRef] = T {
    // Only works if -d or --debug flag used in mill
    T.log.debug("Starting Laika task")

    // Destination of task
    val target = T.dest.toIO.getAbsolutePath
    T.log.debug(s"Destination: $target")

    T.log.debug(s"laikaSources: ${laikaSources()}")

    //    val cp = runClasspath().map(_.path)
    // val dirParams = mdocSources().map(pr => Seq(s"--in", pr.path.toIO.getAbsolutePath, "--out",  dir)).iterator.flatten.toSeq
    // Jvm.runLocal("mdoc.Main", cp, dirParams)

    // TODO : remove
    //    val result1 = StorchSitePlugin.transformer.transform("hello *there*")
    //    println(result1)
    //    T.log.info(result1.toString)
    T.log.debug(s"sources = ${sources()}")
    T.log.debug(s"millSourcePath = ${millSourcePath}")
    T.log.debug(s"allSources = ${allSources()}")


    //val sources = T.source
    //T.log.debug(s"sources = $sources")

    // docs/about.md
    val result: IO[RenderedTreeRoot[IO]] = StorchSitePlugin.createTransformer[IO].use {
      t =>
        // TODO
        //t.fromDirectories()
        t.fromDirectory(millSourcePath.toIO.getAbsolutePath)
          .toDirectory(target)
          .transform
    }

    import cats.effect.unsafe.implicits.global

    val syncResult: RenderedTreeRoot[IO] = result.unsafeRunSync()
    T.log.debug(s"syncResult: $syncResult")

    PathRef(T.dest)
  }

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
