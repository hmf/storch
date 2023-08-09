// scala 2.13.3

/*
I am investigating how to tweak VSCode to be able to develop my Mill scripts with all the Metals goodies. According to the documentation and some question posted here, the use of ammonite scripts with Scala 2 should not be a problem. However, directly using a Mill script does not work because we don't have access to the Mill libraries (VSCode sees this as a simple ammonite script so no mill imports work). 

I find that once the `build.sc` file is loaded and used, none of the IDE intellisense works. Nether for the ivy imports or the default Mill jars seem to visible to the IDE. The base Scala library is however visible. Shouldn't Intellisense work on all the ivy imports? 

When I use the name `build.worksheet.sc` the ivy import report errors correctly interactivelly, but all standard import thereafter fail. Shouldn't the following work?

```scala
import $ivy.`org.typelevel::cats-effect:3.5.1`

import cats.effect._
```
TIA
*/





// cSpell:ignore storch, sbt, sonatype, laika, epub
// cSpell:ignore javac
// cSpell:ignore javacpp, sourcecode
// cSpell:ignore Agg
// cSpell:ignore redist, mkl, cuda, cudann, openblas
// cSpell:ignore progressbar, progressbar, munit, munit-scalacheck, scrimage

import $ivy.`org.typelevel::cats-effect:3.5.1`
import $ivy.`org.planet42::laika-core:0.19.3`
import $ivy.`org.planet42::laika-io:0.19.3`
import $ivy.`org.planet42::laika-pdf:0.19.3`

import cats.effect._
//import cats.syntax.all._

import java.util.Locale
import java.util.concurrent.Executors
import scala.concurrent.ExecutionContext
import scala.concurrent._
import ExecutionContext.Implicits.global

val x = 1
val y = 2
val z = x + y

import laika.api._
import laika.format._
import laika.io.implicits._
import laika.markdown.github.GitHubFlavor
import laika.parse.code.SyntaxHighlighting
import laika.io.api.TreeTransformer

val transformer = Transformer.from(Markdown)
                                .to(HTML)
                                .using(
                                  GitHubFlavor,
                                  SyntaxHighlighting
                                )
                                .build


val result1 = transformer.transform("hello *there*")
println(result1)

// This exists
// https://repo1.maven.org/maven2/com/lihaoyi/mill-main_2.13/
// https://repo1.maven.org/maven2/com/lihaoyi/mill-main_2.13/0.11.1/

// import $ivy.`com.lihaoyi::mill-main:0.11.1`
// not found: https://repo1.maven.org/maven2/com/lihaoyi/mill-main_3/0.11.1/mill-main_3-0.11.1.pom
// import $ivy.`com.lihaoyi::mill-main_2.13:0.11.1`
// not found: https://repo1.maven.org/maven2/com/lihaoyi/mill-main_2.13_3/0.11.1/mill-main_2.13_3-0.11.1.pom
// import $ivy.`com.lihaoyi:::mill-main:0.11.1`
import $ivy.`com.lihaoyi:mill-main_2.13:0.11.1`
import $ivy.`com.lihaoyi:mill-scalalib_2.13:0.11.1`
import $ivy.`com.lihaoyi:mill-contrib-bloop_2.13:0.11.1`

// https://repo1.maven.org/maven2/com/lihaoyi/

import mill._
import mill.scalalib._
import mill.define.ModuleRef
import coursier.maven.MavenRepository
import mill.contrib.bloop.Bloop

val ScalaVersion = "3.3.0"


trait CommonSettings extends SbtModule with Bloop.Module {
  def scalaVersion = T{ ScalaVersion }

  def aTask = T {
    T.log.debug("This works")
  }
}

/*
// Mill macros break
// could not find implicit value for parameter outerCtx0: mill.define.Ctxscalac

object docs extends CommonSettings {
}
*/
