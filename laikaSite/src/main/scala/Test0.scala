import cats.effect.IO._
import cats.effect.{Async, Clock, IO, Resource}
import laika.api.Transformer
import laika.ast.LengthUnit.px
import laika.ast.Path.Root
import laika.ast.{Image, Length, LengthUnit, Styles}
import laika.format.{HTML, Markdown}
import laika.helium.Helium
import laika.helium.config.{HeliumIcon, IconLink, Teaser, TextLink}
import laika.io.api.TreeTransformer
import laika.markdown.github.GitHubFlavor
import laika.rewrite.link.{ApiLinks, LinkConfig}
import laika.rewrite.nav.{ChoiceConfig, SelectionConfig, Selections}
import laika.api._
import laika.format._
import laika.io.implicits._
//import laika.io.model.FilePath
import laika.markdown.github.GitHubFlavor
import laika.parse.code.SyntaxHighlighting
import laika.io.api.TreeTransformer
import laika.io.model.RenderedTreeRoot

import scala.collection.immutable.Seq

// Mill stubs start
case class VersionControl(
                           browsableRepository: Option[String] = None,
                           connection: Option[String] = None,
                           developerConnection: Option[String] = None,
                           tag: Option[String] = None
                         )

object VersionControl {
  def github(owner: String, repo: String, tag: Option[String] = None): VersionControl =
    VersionControl(
      browsableRepository = Some(s"https://github.com/$owner/$repo"),
      connection = Some(s"https://github.com/$owner/$repo.git"),
      developerConnection = Some(s"ssh://github.com/:$owner/$repo.git"),
      tag = tag
    )
}

case class Developer(
                      id: String,
                      name: String,
                      url: String,
                      organization: Option[String] = None,
                      organizationUrl: Option[String] = None
                    )


// Mill stubs end


object StorchSitePlugin {

  // Some(s"https://github.com/$owner/$repo")
  val scmInfo = VersionControl.github("sbrunk", "storch")
  val tlBaseVersion = "0.0" // your current series x.y
  val version_ = s"$tlBaseVersion-SNAPSHOT"

  val organization = "dev.storch"
  val organizationName = "storch.dev"
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
  // https://github.com/typelevel/cats-effect/issues/280
  // https://typelevel.org/Laika/latest/02-running-laika/02-library-api.html
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


}

object Test0 {


  /*
  Destination: /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest
  laikaSources: Vector(ref:v0:49315e08:/mnt/ssd2/hmf/IdeaProjects/storch/docs)
  sources = Vector(ref:v0:a8a72cb2:/mnt/ssd2/hmf/IdeaProjects/storch/docs/src/main/scala, ref:v0:c984eca8:/mnt/ssd2/hmf/IdeaProjects/storch/docs/src/main/java)
  millSourcePath = /mnt/ssd2/hmf/IdeaProjects/storch/docs
  allSources = List(ref:v0:a8a72cb2:/mnt/ssd2/hmf/IdeaProjects/storch/docs/src/main/scala, ref:v0:c984eca8:/mnt/ssd2/hmf/IdeaProjects/storch/docs/src/main/java)
  docJar ref = /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/docJar.dest/out.jar
  docJar path = /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/docJar.dest
  Copied from /mnt/ssd2/hmf/IdeaProjects/storch/docs to /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src
  From: siteTargetSource = /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src
  To: target = /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site

   */
  def main(args: Array[String]): Unit = {
    Console.println("Laika Test 0")

    val siteTargetSource = "/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src"
    val siteTmp = "/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site"

    val result = StorchSitePlugin.createTransformer[IO].use {
      t =>
        // TODO; reactivate on 0.19.x
        // t.fromDirectory(FilePath.fromNioPath(siteTargetSource.toNIO))
        //t.fromDirectory(siteTargetSource.toIO.getAbsolutePath)
        t.fromDirectory(siteTargetSource)
          .toDirectory(siteTmp)
          .describe
          .map(_.formatted)
    }

    import cats.effect.unsafe.implicits.global

    val syncResult = result.unsafeRunSync()
    Console.println(s"syncResult: $syncResult")

  }
}

object Test1 {
  def main(args: Array[String]): Unit = {
    Console.println("Laika Test 1")

    val siteTargetSource = "/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src"
    val siteTmp = "/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site"

    // https://typelevel.org/Laika/latest/02-running-laika/02-library-api.html#separate-parsing-and-rendering
    val parser = MarkupParser
      .of(Markdown)
      .using(GitHubFlavor)
      .parallel[IO]
      .build

    val htmlRenderer = Renderer.of(HTML).parallel[IO].build
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
        parser.fromDirectory(siteTargetSource).parse.flatMap {
          tree =>
            Console.println(tree.root)
            val htmlOp = htmlRenderer.from(tree.root).toDirectory(siteTmp).render
            val epubOp = epubRenderer.from(tree.root).toFile("out.epub").render
            val pdfOp = pdfRenderer.from(tree.root).toFile("out.pdf").render
            (htmlOp, epubOp, pdfOp).parMapN { (_, _, _) => () }
        }
    }

    import cats.effect.unsafe.implicits.global

    val syncResult = transformOp.unsafeRunSync()
    Console.println(s"syncResult: $syncResult")

  }
}
