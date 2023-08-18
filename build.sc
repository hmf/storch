// cSpell:ignore storch, sbt, sonatype, laika, epub
// cSpell:ignore javac
// cSpell:ignore javacpp, sourcecode
// cSpell:ignore Agg
// cSpell:ignore redist, mkl, cuda, cudann, openblas
// cSpell:ignore progressbar, progressbar, munit, munit-scalacheck, scrimage

//import $ivy.`org.typelevel::cats-effect:3.5.1`
import laika.api.builder.ParserBuilder
import laika.io.api.TreeParser
import laika.theme.ThemeProvider
import mill.Target
import mill.api.Loose
import mill.util.Jvm

import scala.Console
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

import cats._, cats.data._, cats.implicits._

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
import laika.helium.config.{ ThemeNavigationSection, TextLink }

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

  val tlSiteHeliumConfig = Helium.defaults
    .site
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
    .site
    .topNavigationBar(
      // homeLink undefined, so use landing page
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
    .site.mainNavigation(
        depth = 2,
        includePageSections = false,
        appendLinks = Seq(
          ThemeNavigationSection("Related Projects",
            TextLink.external("https://pytorch.org/", "PyTorch"),
            TextLink.external("https://github.com/bytedeco/javacpp", "JavaCPP")
          )
        )
    )


  // Not working: https://github.com/typelevel/Laika/discussions/485
  // https://typelevel.org/Laika/latest/02-running-laika/02-library-api.html
  // https://github.com/typelevel/cats-effect/issues/280
  def createTransformer[F[_]: Async]: Resource[F, TreeTransformer[F]] =
    Transformer
      .from(Markdown)
      .to(HTML)
      .using(GitHubFlavor, SyntaxHighlighting)
      .withConfigValue(linkConfig)
      .withConfigValue(buildToolSelection)
      .withRawContent
      .parallel[F]
      .withTheme(tlSiteHeliumConfig.build)
      .build


  // We want to add am HTML snippet to the bottom of the landing page
  // A landing page an be fully defined in a landing-page.<suffix> file
  // in the root. If the suffix is HTML and no landing page is configured,
  // then this is the full index.html. If the suffix is Markdown and a
  // landing  page is configured, its contents are processed and added
  // to the end of the landing page index.html.
  // If the suffix is HTML and a landing page is configured, then its
  // contents are ignored because the transformer ignores all raw
  // content. To append HTML content we need to use:
  //   ParserBuilder.withRawContent
  // https://github.com/typelevel/Laika/discussions/489

  def createTransformer(sources: String, targetHTML:String) = {
    val parserBuilder: ParserBuilder = MarkupParser
      .of(Markdown)
      .using(GitHubFlavor, SyntaxHighlighting)
      .withConfigValue(linkConfig)
      .withConfigValue(buildToolSelection)
      .withRawContent

    // https://typelevel.org/Laika/latest/02-running-laika/02-library-api.html#separate-parsing-and-rendering
    val parser: Resource[IO, TreeParser[IO]] = parserBuilder
      .parallel[IO]
      .withTheme(tlSiteHeliumConfig.build)
      .build

    // https://github.com/typelevel/Laika/discussions/489
    val htmlRenderer = Renderer
                          .of(HTML)
                          .withConfig(parserBuilder.config)
                          .parallel[IO]
                          .withTheme(tlSiteHeliumConfig.build)
                          .build
    /* Example for generating books
    val epubRenderer = Renderer.of(EPUB).parallel[IO].build
    val pdfRenderer = Renderer.of(PDF).parallel[IO].build

    val allResources = for {
      parse <- parser
      html <- htmlRenderer
      epub <- epubRenderer
      pdf <- pdfRenderer
    } yield (parse, html, epub, pdf)

    import cats.syntax.all._

    val transformOp: IO[Unit] = allResources.use {
      case (parser, htmlRenderer, epubRenderer, pdfRenderer) =>
        parser.fromDirectory(sources).parse.flatMap {
          tree =>
            val htmlOp = htmlRenderer.from(tree.root).toDirectory(targetHTML).render
            val epubOp = epubRenderer.from(tree.root).toFile("out.epub").render
            val pdfOp = pdfRenderer.from(tree.root).toFile("out.pdf").render
            (htmlOp, epubOp, pdfOp).parMapN { (_, _, _) => () }
        }
    }
    */

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
  override def scalaVersion = ScalaVersion

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
      // Additional dependencies required to use CUDA, cuDNN, and NCCL
      // ivy"org.bytedeco:pytorch-platform-gpu:$pytorchVersion-${javaCppVersion}",
      // Additional dependencies to use bundled CUDA, cuDNN, and NCCL
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

  object test extends SbtModuleTests with TestCommonSettings
}

object vision extends SbtModule with CommonSettings {
   override def moduleDeps = Seq(core)

  override def ivyDeps = super[CommonSettings].ivyDeps() ++ Agg(
      ivy"com.sksamuel.scrimage:scrimage-core:$scrImageVersion",
      ivy"com.sksamuel.scrimage:scrimage-webp:$scrImageVersion",
    )

  object test extends SbtModuleTests with TestCommonSettings
}


object examples extends CommonSettings {
  override def moduleDeps = Seq(vision)
  override def forkArgs = Seq("-Djava.awt.headless=true")
  override def scalacOptions = Seq("-explain")
  override def ivyDeps = super.ivyDeps() ++ Agg(
    ivy"me.tongfei:progressbar:0.9.5",
    ivy"com.github.alexarchambault::case-app:2.1.0-M24",
    ivy"org.scala-lang.modules::scala-parallel-collections:1.0.4"
    )

  object test extends SbtModuleTests with TestCommonSettings
}


/**
 * Generates all documentation:
 * - API Scaladoc
 * - Laika HTML from Markdown source
 */
object docs extends CommonSettings {

  override def scalaVersion = T{ ScalaVersion }
  override def scalacOptions = T{ super.scalacOptions() ++  Seq("-groups") }

  override def moduleDeps = Seq(core, vision, examples)

  def scalaMdocVersion : T[String] = T("2.3.7")

  // https://github.com/scalameta/mdoc/issues/702
  // MDoc has its own dependencies on the Scala compiler and uses those
  // To use a later version of Scala 3, we need to download that version of the compiler
  def scalaMdocDep = T.task {
    Agg(
      ivy"org.scalameta::mdoc:${scalaMdocVersion()}"
        .exclude("org.scala-lang" -> "scala3-compiler_3")
        .exclude("org.scala-lang" -> "scala3-library_3"),
      ivy"org.scala-lang::scala3-compiler:${scalaVersion()}"
    ).map(Lib.depToBoundDep(_, scalaVersion()))
  }

  // Only downloads source code
  // resolveDeps(mdocDep, sources = true)
  def mDocLibs = T{ resolveDeps(scalaMdocDep )}

  val separator = java.io.File.pathSeparatorChar
  def toArgument(p: Agg[os.Path]) = p.mkString(s"$separator")


  override def ivyDeps = T {
    super.ivyDeps()
  }

  override def docResources: T[Seq[PathRef]] = T {
    core.docResources() ++
      vision.docResources() ++
      examples.docResources() ++
      super.docResources()
  }

  override def docSources: T[Seq[PathRef]] = T {
    core.docSources() ++
      vision.docSources() ++
      examples.docSources() ++
      super.docSources()
  }


  /*
    NOTE on issue using anonymous tasks
    Issue: https://github.com/com-lihaoyi/mill/issues/2694
    Answer: https://github.com/com-lihaoyi/mill/issues/2694#issuecomment-1677127114
    Example code: https://mill-build.com/mill/example/tasks/3-anonymous-tasks.html

    def anonTask(fileName: String): Task[String] = T.task {
      fileName + "Anon"
    }
    def helloFileData = T { anonTask("hello.txt")() }
    def printFileData(fileName: String) = T.command {
      Console.println(anonTask(fileName)())
    }

    Workaround:
  */
  def anonTask: Task[String => String] = T.task {
    s:String => s + "Anon"
  }

  // https://github.com/com-lihaoyi/mill/blob/main/docs/modules/ROOT/pages/Mill_Design_Principles.adoc
  def anonTask0(fileName: String): Task[String] = T.task {
    fileName + "Anon"
  }
  def helloFileDataO = T { anonTask0("hello.txt") }
  def helloFileData = T { anonTask.map(f => f("String-")) }
  def printFileData(fileName: String): Command[String] = T.command {
    val dest = T.dest.toString() + s"/$fileName"

    // Won't work
    // Target#apply() call cannot use `value newDest` defined within the T{...} block
    // val newDest = anonTask0(dest)
    // val result1 = newDest()
    // val result1 = newDest.apply()
    // Console.println(result1)

    // Won't work
    //  Target.ctx() / T.ctx() / T.* APIs can only be used with a T{...} block
    // val newDest0 = anonTask0( T.dest.toString() )
    // Console.println( newDest0() )

    // Workaround
    // Fails
    // val newDest0 = anonTask.map( f => f(T.dest.toString()) )
    // Console.println( newDest0() )
    val result0 = anonTask.apply()
    Console.println( result0(T.dest.toString()) )
    Console.println( result0( dest ) )

    dest
  }

  // TODO: add watch support?
  def mdocParams: Task[(Seq[os.Path], os.Path, Map[String, String]) => (Loose.Agg[os.Path], Seq[String])] = T.task {
    (mdocSources: Seq[os.Path],
     destination: os.Path,
     siteVariables: scala.collection.immutable.Map[String, String])
    => {
      //val cp = runClasspath().map(_.path)
      val cp = compileClasspath().map(_.path)
      val rp = mDocLibs().map(_.path)
      val dir = destination.toIO.getAbsolutePath
      val dirParams = mdocSources.map(pr => Seq(s"--in", pr.toIO.getAbsolutePath, "--out", dir)).iterator.flatten
      val vars = siteVariables.map { case (k, v) => s"--site.$k=$v" }
      val docClasspath = toArgument(cp)
      val params = Seq("--classpath", s"$docClasspath") ++
        (dirParams ++ vars).toSeq
          //.appended("--verbose")
      (rp, params)
    }
  }

  /**
   * Calls MDoc to parse the Markdown sources with scala code.
   * Care must be taken to replace the MDoc Scala 3 compiler
   * and use the one that is configured for the project.
   *
   * Note: We have to use a function return type task due to Mill
   * issues with anonymous. We must call this task explicitly
   * using the `apply` method.
   *
   * @see https://github.com/com-lihaoyi/mill/issues/2694
   * @see https://github.com/com-lihaoyi/mill/issues/2694#issuecomment-1677127114
   * @see https://github.com/scalameta/mdoc/issues/702
   * @see https://github.com/hmf/mdocMill
   * @return
   */
  def mdocLocal: Task[(Seq[os.Path], os.Path, Map[String, String]) => PathRef] = T.task {

    (mdocSources: Seq[os.Path],
     destination: os.Path,
     siteVariables: scala.collection.immutable.Map[String, String])
    => {
      val paramsFrom = mdocParams.apply()
      val (rp, params) = paramsFrom(mdocSources, destination, siteVariables)

      Jvm.runLocal("mdoc.Main", rp, params)

      PathRef(T.dest)
    }
  }

  /**
   * Calls MDoc to parse the Markdown sources with scala code.
   * Care must be taken to replace the MDoc Scala 3 compiler
   * and use the one that is configured for the project.
   *
   * Note: We have to use a function return type task due to Mill
   * issues with anonymous. We must call this task explicitly
   * using the `apply` method.
   *
   * @see https://github.com/com-lihaoyi/mill/issues/2694
   * @see https://github.com/com-lihaoyi/mill/issues/2694#issuecomment-1677127114
   * @see https://github.com/scalameta/mdoc/issues/702
   * @see https://github.com/hmf/mdocMill
   * @return
   */
  def mdoc: Task[(Seq[os.Path], os.Path, Map[String, String]) => PathRef] = T.task {

    (mdocSources : Seq[os.Path],
     destination: os.Path,
     siteVariables: scala.collection.immutable.Map[String, String])
    => {

      val paramsFrom = mdocParams.apply()
      val (rp, params) = paramsFrom(mdocSources, destination, siteVariables)

      Jvm.runSubprocess(
        mainClass = "mdoc.Main",
        classPath = rp,
        jvmArgs = forkArgs(),
        envArgs = forkEnv(),
        mainArgs = params,
        // Defaults
        workingDir = forkWorkingDir(),
        useCpPassingJar = runUseArgsFile()
      )

      PathRef(T.dest)
    }
  }

  // TODO: remove?
  // where do the mdoc sources live ?
  def laikaSources = T.sources {
    super.millSourcePath
  }



  def laikaBase: Task[Boolean => PathRef] = T.task {
    useLocalMDoc: Boolean => {
      // Only works if -d or --debug flag used in mill
      T.log.debug("Starting Laika task")

      // Destination of task
      val target = T.dest
      T.log.debug(s"Destination: ${target.toIO.getAbsolutePath}")
      T.log.debug(s"laikaSources: ${laikaSources()}")
      T.log.debug(s"millSourcePath = ${millSourcePath}")
      T.log.debug(s"allSources = ${allSources()}")

      // Generate the API docs and get the Jar file with API docs
      // See docResources and docSources
      T.log.info(s"docJar generation")
      val javadoc = docJar()
      T.log.debug(s"docJar ref = ${javadoc.path.toIO.getAbsolutePath}")
      // Path still contains the uncompressed contents also
      // Extract the path by removing the Jar file name
      val dest: os.Path = javadoc.path / os.up
      T.log.debug(s"docJar path = ${dest.toIO.getAbsolutePath}")
      // Delete the Jar file
      // Add the path to the laika directories
      val apiSource = dest / "javadoc"
      val siteTargetSource = target / "site_src"

      // Use mdoc to processes and copy sources
      T.log.info(s"MDoc processing and copy from ${millSourcePath} to ${siteTargetSource}")
      // os.copy(from = source, to = siteTargetSource)
      // TODO: remove/reduce cats-effects Map, print etc
      val siteVariables = scala.collection.immutable.Map(
        "VERSION" -> "1.0.0",
        "PYTORCH_VERSION" -> pytorchVersion,
        "JAVACPP_VERSION" -> javaCppVersion,
        "OPENBLAS_VERSION" -> openblasVersion,
        "CUDA_VERSION" -> cudaVersion
      )
      val mdocSources = laikaSources().map(_.path)
      T.log.debug(s"Use MDoc local = $useLocalMDoc")
      val mdocProc = if (useLocalMDoc)  mdocLocal.apply() else mdoc.apply()
      val r = mdocProc(mdocSources, siteTargetSource, siteVariables)
      T.log.debug(s"MDoc results written to $r")

      // Final destination of the site
      val siteTmp = target / "site"
      T.log.info(s"Creating temporary site at $siteTmp")
      os.makeDir.all(siteTmp)

      T.log.info(s"Copying addition site resources to $siteTmp")
      // Path to the pre-processed site source
      val siteSource = millSourcePath / os.up / "site" / "src"
      // Copy additional site resources
      // We could copy the templates, but do not
      // see https://github.com/typelevel/Laika/discussions/485#discussioncomment-6693405
      val templates = os.list(siteSource)
      templates.foreach { p =>
        if (
            (!p.toString().contains("default.template.html")) &&
            (!p.toString().contains("landing.template.html"))
          ) {
          T.log.info(s"\tCopied from ${p} into ${siteTargetSource}")
          os.copy.into(from = p, to = siteTargetSource)
          // Not copied by Laika
          T.log.info(s"\tCopied from ${p} into ${siteTmp}")
          os.copy.into(from = p, to = siteTmp)
        }
      }

      // Copy API docs to site
      val apiTarget = siteTargetSource / "api"
      T.log.info(s"Copy API from ${apiSource} to ${apiTarget}")
      os.copy(from = apiSource, to = apiTarget)
      // Not copied by Laika
      T.log.info(s"Copy API from ${apiSource} to ${siteTmp / "api"}")
      os.copy(from = apiSource, to = siteTmp / "api")

      T.log.info(s"siteTargetSource = $siteTargetSource")
      T.log.info(s"target = $siteTmp")
      // os.makeDir.all(siteTmp)

      // Example of debugging
      /*
      import cats.effect.unsafe.implicits.global
      val result = StorchSitePlugin.createTransformer[IO].use {
        t =>
          t.fromDirectory(FilePath.fromNioPath(siteTargetSource.toNIO))
            .toDirectory(siteTmp.toIO.getAbsolutePath)
            .describe
            .map(_.formatted)
      }

      val syncResult = result.unsafeRunSync()
      T.log.debug(s"syncResult: $syncResult")
      */

      T.log.info(s"Creating Laika Transformer")
      /*
      // Not working, use separate parser and renderer
      // Does not work: https://github.com/typelevel/Laika/discussions/485
      val result1: IO[RenderedTreeRoot[IO]] = StorchSitePlugin.createTransformer[IO].use {
        t =>
            t.fromDirectory(FilePath.fromNioPath(siteTargetSource.toNIO))
            .toDirectory(siteTmp.toIO.getAbsolutePath)
            .transform
      }

      import cats.effect.unsafe.implicits.global

      val syncResult1: RenderedTreeRoot[IO] = result1.unsafeRunSync()
      T.log.debug(s"syncResult1: $syncResult1")

      T.log.debug("allDocuments:")
      T.log.debug(syncResult1.allDocuments.mkString(",\n"))
      */
      T.log.debug(s"siteTargetSource = $siteTargetSource")
      val result: IO[Unit] = StorchSitePlugin.createTransformer(siteTargetSource.toIO.getAbsolutePath, siteTmp.toIO.getAbsolutePath)
      import cats.effect.unsafe.implicits.global

      T.log.info(s"Executing Laika Transformer")
      val syncResult1: Unit = result.unsafeRunSync()
      T.log.debug(s"syncResult1: $syncResult1")


      PathRef(T.dest)
    }
  }

  def laika : T[PathRef] = T {
    val site = laikaBase.apply()
    site(false)
  }

  def laikaLocal: T[PathRef] = T {
    val site = laikaBase.apply()
    site(true)
  }

  // TODO: https://github.com/typelevel/Laika/discussions/492
  // Local browsing

  /*
  https://github.com/typelevel/Laika/blob/dd872237fb552c8f8b4119a3c02ab2e55bad96fb/sbt/src/main/scala/laika/sbt/LaikaPlugin.scala#L135
  laikaPreview
  laikaPreviewConfig
  startPreviewServer

  https://github.com/typelevel/Laika/blob/dd872237fb552c8f8b4119a3c02ab2e55bad96fb/sbt/src/main/scala/laika/sbt/Tasks.scala#L249
  startPreviewServer

  val (_, cancel) = buildPreviewServer.value.allocated.unsafeRunSync()

  streams.value.log.info(
    s"Preview server started on port ${laikaPreviewConfig.value.port}. Press return/enter to exit."
  )

  try {
    System.in.read
  }
  finally {
    streams.value.log.info(s"Shutting down preview server...")
    cancel.unsafeRunSync()
  }

  buildPreviewServer
  ServerConfig

  import laika.sbt (not available)
  import laika.preview (not available)
  */

  // TODO: with watch
  def laikaServe = {

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
