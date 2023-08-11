// cSpell:ignore storch, sbt, sonatype, laika, epub
// cSpell:ignore javac
// cSpell:ignore javacpp, sourcecode
// cSpell:ignore Agg
// cSpell:ignore redist, mkl, cuda, cudann, openblas
// cSpell:ignore progressbar, progressbar, munit, munit-scalacheck, scrimage

//import $ivy.`org.typelevel::cats-effect:3.5.1`
import laika.theme.ThemeProvider
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
import laika.io.model.FilePath
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


  val linkConfig = LinkConfig(apiLinks =
    Seq(
      // ApiLinks(baseUri = "http://localhost:4242/api/")
      ApiLinks(baseUri = "https://storch.dev/api/")
    )
  )
 val buildToolSelection = Selections(
                            SelectionConfig(
                              "build-tool",
                              ChoiceConfig("sbt", "sbt"),
                              ChoiceConfig("scala-cli", "Scala CLI")
                              ).withSeparateEbooks
                            )



  //scmInfo.value.fold("https://github.com/sbrunk/storch")(_.browseUrl.toString),
  val browsableLink = scmInfo.browsableRepository.getOrElse("SCMInfo Missing")

  // https://github.com/typelevel/Laika/blob/fc65b74006d42b122810c911d5768c4359969ae4/docs/src/05-extending-laika/02-creating-themes.md?plain=1#L357
  import laika.theme.{ Theme, ThemeBuilder, ThemeProvider }
  import laika.helium.builder

  class HeliumThemeBuilder (helium: Helium) extends ThemeProvider {

    def build[F[_]: Async]: Resource[F, Theme[F]] = {

      import helium._
      import laika.helium.builder

      val treeProcessor = new HeliumTreeProcessor[F](helium)

  }

  val tlSiteHeliumConfig = Helium.defaults
    .site
    .topNavigationBar(
      //homeLink = IconLink.internal(Root / "README.md", HeliumIcon.home),
      // homeLink =  ButtonLink.external("http://somewhere.com/", "Button Link"),
      //ButtonLink.external("http://somewhere.com/", "Button Link"),
      navLinks = Seq(
        //IconLink.internal(Root / "doc-2.md", HeliumIcon.download),
        TextLink.external("http://somewhere.com/", "Text Link"),
        ButtonLink.external("http://somewhere.com/", "Button Link")
      ),
      versionMenu = VersionMenu.create(
        versionedLabelPrefix = "Version:",
        unversionedLabel = "Choose Version"
      )
      //highContrast = true
    )
    .site.landingPage(
        logo = Some(Image(ExternalTarget("http://my-site/my-image.jpg"))),
        title = Some("Project Name"),
        subtitle = Some("Fancy Hyperbole Goes Here"),
        latestReleases = Seq(
          ReleaseInfo("Latest Stable Release", "2.3.5"),
          ReleaseInfo("Latest Milestone Release", "2.4.0-M2")
        ),
        license = Some("MIT"),
        titleLinks = Seq(
          VersionMenu.create(unversionedLabel = "Getting Started"),
          LinkGroup.create(
            IconLink.external("https://github.com/abcdefg/", HeliumIcon.github),
            IconLink.external("https://gitter.im/abcdefg/", HeliumIcon.chat),
            IconLink.external("https://twitter.com/abcdefg/", HeliumIcon.twitter)
          )
        ),
        documentationLinks = Seq(
          TextLink.internal(Root / "doc-1.md", "Doc 1"),
          TextLink.internal(Root / "doc-2.md", "Doc 2")
        ),
        projectLinks = Seq(
          TextLink.internal(Root / "doc-1.md", "Text Link"),
          ButtonLink.external("http://somewhere.com/", "Button Label"),
          LinkGroup.create(
            IconLink.internal(Root / "doc-2.md", HeliumIcon.demo),
            IconLink.internal(Root / "doc-3.md", HeliumIcon.info)
          )
        ),
        teasers = Seq(
          Teaser("Teaser 1", "Description 1"),
          Teaser("Teaser 2", "Description 2"),
          Teaser("Teaser 3", "Description 3")
        )
      )
//    .site
//    .metadata(
//      title = Some("Storch"),
//      authors = developers.map(_.name),
//      language = Some("en"),
//      version = Some(version_)
//    )
//    .site
//    .layout(
//      contentWidth = px(860),
//      navigationWidth = px(275),
//      topBarHeight = px(50),
//      defaultBlockSpacing = px(10),
//      defaultLineHeight = 1.5,
//      anchorPlacement = laika.helium.config.AnchorPlacement.Right
//    )
//    .site
//    .topNavigationBar(
////      // TODO: new
////      homeLink = IconLink.internal(Root / "api" / "index.html", HeliumIcon.home),
//      navLinks = Seq(
//        IconLink.internal(
//          Root / "api" / "index.html",
//          HeliumIcon.api,
//          options = Styles("svg-link")
//        ),
//        IconLink.external(
//          browsableLink,
//          HeliumIcon.github,
//          options = Styles("svg-link")
//        )
//      )
//    )
//    .site
//    .landingPage(
//      logo = Some(
//        Image.internal(Root / "img" / "storch.svg", height = Some(Length(300, LengthUnit.px)))
//      ),
//      title = Some("Storch"),
//      subtitle = Some("GPU Accelerated Deep Learning for Scala 3"),
//      license = Some("Apache 2"),
//      documentationLinks = Seq(
//        TextLink.internal(Root / "about.md", "About"),
//        TextLink.internal(Root / "installation.md", "Getting Started"),
//        TextLink.internal(Root / "api" / "index.html", "API (Scaladoc)")
//      ),
//      projectLinks = Seq(
//        IconLink.external(
//          browsableLink,
//          HeliumIcon.github,
//          options = Styles("svg-link")
//        )
//      ),
//      teasers = Seq(
//        Teaser(
//          "Build Deep Learning Models in Scala",
//          """
//            |Storch provides GPU accelerated tensor operations, automatic differentiation,
//            |and a neural network API for building and training machine learning models.
//            |""".stripMargin
//        ),
//        Teaser(
//          "Get the Best of PyTorch & Scala",
//          """
//            |Storch aims to be close to the original PyTorch API, while still leveraging Scala's powerful type
//            |system for safer tensor operations.
//            |""".stripMargin
//        ),
//        Teaser(
//          "Powered by LibTorch & JavaCPP",
//          """
//            |Storch is based on <a href="https://pytorch.org/cppdocs/">LibTorch</a>, the C++ library underlying PyTorch.
//            |JVM bindings are provided by <a href="https://github.com/bytedeco/javacpp">JavaCPP</a> for seamless
//            |interop with native code & CUDA support.
//            |""".stripMargin
//        )
//      )
//    )

//  val transformer = Transformer.from(Markdown)
//                                  .to(HTML)
//                                  .using(
//                                    GitHubFlavor,
//                                    SyntaxHighlighting
//                                  )
//                                  .build

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

  // Not working: https://github.com/typelevel/Laika/discussions/485
  // https://typelevel.org/Laika/latest/02-running-laika/02-library-api.html
  // https://github.com/typelevel/cats-effect/issues/280
  def createTransformer[F[_]: Async]: Resource[F, TreeTransformer[F]] =
    Transformer
      .from(Markdown)
      .to(HTML)
      .using(GitHubFlavor)
      .withConfigValue(linkConfig)
      .withConfigValue(buildToolSelection)
      //.withRawContent
      .parallel[F]
      //.withTheme(tlSiteHeliumConfig.build)
      .build


  def createTransformer(sources: String, targetHTML:String) = {
    // https://typelevel.org/Laika/latest/02-running-laika/02-library-api.html#separate-parsing-and-rendering
    val parser = MarkupParser
      .of(Markdown)
      .using(GitHubFlavor)
      .withConfigValue(linkConfig)
      .withConfigValue(buildToolSelection)
      //.withRawContent
      .parallel[IO]
      .withTheme(tlSiteHeliumConfig.build)
      .build

    val htmlRenderer = Renderer.of(HTML).parallel[IO].build
    val epubRenderer = Renderer.of(EPUB).parallel[IO].build
    val pdfRenderer = Renderer.of(PDF).parallel[IO].build

//    val allResources = for {
//      parse <- parser
//      html <- htmlRenderer
//      epub <- epubRenderer
//      pdf <- pdfRenderer
//    } yield (parse, html, epub, pdf)
//
//    import cats.syntax.all._
//
//    val transformOp: IO[Unit] = allResources.use {
//      case (parser, htmlRenderer, epubRenderer, pdfRenderer) =>
//        parser.fromDirectory(sources).parse.flatMap {
//          tree =>
//            val htmlOp = htmlRenderer.from(tree.root).toDirectory(targetHTML).render
//            val epubOp = epubRenderer.from(tree.root).toFile("out.epub").render
//            val pdfOp = pdfRenderer.from(tree.root).toFile("out.pdf").render
//            (htmlOp, epubOp, pdfOp).parMapN { (_, _, _) => () }
//        }
//    }

    val allResources = for {
      parse <- parser
      html <- htmlRenderer
    } yield (parse, html)

    import cats.syntax.all._

    val transformOp: IO[Unit] = allResources.use {
      case (parser, htmlRenderer) =>
        parser.fromDirectory(sources).parse.flatMap {
          tree =>
            val htmlOp = htmlRenderer.from(tree.root).toDirectory(targetHTML).render
            htmlOp.map(_ => ())
        }
    }
    transformOp
  }

  // https://github.com/typelevel/Laika/blob/fc65b74006d42b122810c911d5768c4359969ae4/io/src/main/scala/laika/helium/builder/HeliumTreeProcessor.scala#L38
  // HeliumTreeProcessor[F[_]: Sync](helium: Helium)
  /*
  private def addLandingPage: TreeProcessor[F] =
      siteSettings.content.landingPage
        .fold(noOp)(LandingPageGenerator.generate)

  https://github.com/typelevel/Laika/blob/main/io/src/main/scala/laika/helium/generate/LandingPageGenerator.scala#L29
*/


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

  /* TODO: remove
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

  override def docJarUseArgsFile: T[Boolean] = super.docJarUseArgsFile
*/

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


//  override def docJar: T[PathRef] = T {
//    val ref = super.docJar()
//    // docJar ref = /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/docJar.super/mill/scalalib/ScalaModule/docJar.dest/out.jar
//    T.log.info(s"docJar ref = ${ref.path.toIO.getAbsolutePath}")
//    ref
//  }

  // where do the mdoc sources live ?
  def laikaSources = T.sources {
    super.millSourcePath
  }

  def laika : T[PathRef] = T {
    // Only works if -d or --debug flag used in mill
    T.log.debug("Starting Laika task")

    // Destination of task
    val target = T.dest
    T.log.debug(s"Destination: ${target.toIO.getAbsolutePath}")

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

    // Get Jar file with API docs
    val javadoc = docJar()
    T.log.info(s"docJar ref = ${javadoc.path.toIO.getAbsolutePath}")
    // Path still contains the content
    // Extract the path by removing the Jar file name
    val dest: os.Path = javadoc.path / os.up
    T.log.info(s"docJar path = ${dest.toIO.getAbsolutePath}")
    // Delete the Jar file
    // Add the path to the laika directories
    val apiSource = dest / "javadoc"

    val siteTargetSource = target / "site_src"

    val source = millSourcePath
    T.log.info(s"Copied from ${source} to ${siteTargetSource}")
    os.copy(from = source, to = siteTargetSource)

    val siteTmp = target / "site"
    os.makeDir.all(siteTmp)

    val siteSource = millSourcePath / os.up / "site" / "src"
    // TODO: use?
    val templates = os.list(siteSource)
    templates.foreach{ p =>
      if (p.toString().contains("img")) {
        T.log.info(s"Copied from ${p} into ${siteTargetSource}")
        os.copy.into(from = p, to = siteTargetSource)
        // Not copied by Laika
        T.log.info(s"Copied from ${p} into ${siteTmp}")
        os.copy.into(from = p, to = siteTmp)
      }
    }

    val apiTarget = siteTargetSource / "api"
    T.log.info(s"Copied from ${apiSource} to ${apiTarget}")
    os.copy(from = apiSource, to = apiTarget)
    // Not copied by Laika
    os.copy(from = apiSource, to = siteTmp / "api")

    val docsSource = FilePath.fromNioPath(millSourcePath.toNIO)
    T.log.info(s"docsSource = $docsSource")
//    val apiSite = FilePath.fromNioPath(apiTarget.toNIO)
//    T.log.info(s"apiSite = $apiSite")
    T.log.info(s"From: siteTargetSource = $siteTargetSource")
    T.log.info(s"To: target = $siteTmp")

    // os.makeDir.all(siteTmp)

    // docs/about.md
//    val result: IO[RenderedTreeRoot[IO]] = StorchSitePlugin.createTransformer[IO].use {
//      t =>
//        // TODO
//        val paths = Seq(
//                      docsSource,
//                      apiSite
//                    )
////        t.fromDirectories(paths)
//          //t.fromDirectory(FilePath.fromNioPath(siteTargetSource.toNIO))
//          t.fromDirectory("/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src/")
//        //t.fromDirectory(millSourcePath.toIO.getAbsolutePath)
//          .toDirectory(siteTmp.toIO.getAbsolutePath)
//          .transform
//    }
//
//    import cats.effect.unsafe.implicits.global
//
//    val syncResult: RenderedTreeRoot[IO] = result.unsafeRunSync()
//    T.log.debug(s"syncResult: $syncResult")

    // Example of debugging
//    import cats.effect.unsafe.implicits.global
//    val result = StorchSitePlugin.createTransformer[IO].use {
//      t =>
//        t.fromDirectory(FilePath.fromNioPath(siteTargetSource.toNIO))
//          .toDirectory(siteTmp.toIO.getAbsolutePath)
//          .describe
//          .map(_.formatted)
//    }
//
//    val syncResult = result.unsafeRunSync()
//    T.log.debug(s"syncResult: $syncResult")

      // Does not work: https://github.com/typelevel/Laika/discussions/485
//
//    val result1: IO[RenderedTreeRoot[IO]] = StorchSitePlugin.createTransformer[IO].use {
//      t =>
//          t.fromDirectory(FilePath.fromNioPath(siteTargetSource.toNIO))
//          .toDirectory(siteTmp.toIO.getAbsolutePath)
//          .transform
//    }
//
//    import cats.effect.unsafe.implicits.global
//
//    val syncResult1: RenderedTreeRoot[IO] = result1.unsafeRunSync()
//    T.log.debug(s"syncResult1: $syncResult1")
//
//    T.log.debug("allDocuments:")
//    T.log.debug(syncResult1.allDocuments.mkString(",\n"))

    val result: IO[Unit] = StorchSitePlugin.createTransformer(siteTargetSource.toIO.getAbsolutePath, siteTmp.toIO.getAbsolutePath)
    import cats.effect.unsafe.implicits.global

    val syncResult1: Unit = result.unsafeRunSync()
    T.log.debug(s"syncResult1: $syncResult1")


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

/*
Unresolved internal reference: api/index.html using `fromDirectories`

I am trying to adapt an SBt project to use Mill. I have the following set-up:

1. API HTM at: /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/docJar.dest/api
2. Site markdown at: /mnt/ssd2/hmf/IdeaProjects/storch/docs

I use the following transformer:

```scala
    val docsSource = FilePath.fromNioPath(millSourcePath.toNIO)
    val apiSite = FilePath.fromNioPath(apiTarget.toNIO)

    val result: IO[RenderedTreeRoot[IO]] = StorchSitePlugin.createTransformer[IO].use {
      t =>
        val paths = Seq(
                      docsSource,
                      apiSite
                    )
        t.fromDirectories(paths)
          .toDirectory(target)
          .transform
    }
```

I am assuming that all the `from` paths are merged into a single root so that the `/api`shuld be visible. Here is (part of) the error I get:

```shell
docs.laika laika.parse.markup.DocumentParser$InvalidDocuments: One or more invalid documents:
/README

  [1]: unresolved internal reference: api/index.html


  ^

/about.md

  [1]: unresolved internal reference: api/index.html


  ^

  [1]: unresolved internal reference: api/index.html


  ^

/installation.md

  [1]: unresolved internal reference: api/index.html


  ^

  [1]: unresolved internal reference: api/index.html


  ^
```
The first thing I realize is that I don't have a README. In the `storch/docs` path I have the `directory.conf` file with (no mention of README):

```hocon
laika.navigationOrder = [
  about.md
  installation.md
  modules.md
  examples.md
  pre-trained-weights.md
  faq.md
  contributing.md
]
```

The second thing I realize is that the `\about.md` has no link to `api/index.html`. In fact I don't see any references to an `index.html`. Looking further in the error messages I see a different error, which does not seem to be related:

```shell
  [14]: One or more errors processing directive 'select': Error reading config for selections: Not found: 'laika.selections'

  @:select(build-tool)
  ^

  [54]: One or more errors processing directive 'select': Error reading config for selections: Not found: 'laika.selections'

  @:select(build-tool)
  ^

  [88]: One or more errors processing directive 'select': Error reading config for selections: Not found: 'laika.selections'

  @:select(build-tool)
  ^

  [147]: One or more errors processing directive 'select': Error reading config for selections: Not found: 'laika.selections'

  @:select(build-tool)
  ^

  [182]: One or more errors processing directive 'select': Error reading config for selections: Not found: 'laika.selections'

  @:select(build-tool)
  ^

```
I also see the markdown source uses the following directives:

@:api
@:select(build-tool)
@:choice(sbt)
@:callout(info)
@:callout(warning)
@:@

but the `about.md`does not have these directives (`/installation.md` does). However the source does use code fences tagged with `mdoc`. So my question are:
1. Are the mdoc code fences the cause for these references?
1. Is Laika processing these code fences? Should they not be ignored and processed as a standard code fence?

TIA



 */

/*
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
    // TODO: new
    homeLink = IconLink.internal(Root / "api" / "index.html", HeliumIcon.home),
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

 */