This a is README Title
======================

Some text for the README title.
We reference the [FAQ](faq.md).
We can also reference the [FAQ](../faq.md).

## Experiments

### Single README 

```
Parser(s):
    Markdown
Renderer:
    HTML
Extension Bundles:
    Laika's Default Extensions (supplied by library)
    Laika's directive support (supplied by library)
    Laika's built-in directives (supplied by library)
    Document Type Matcher for Markdown (supplied by parser)
    Github-flavored Markdown (supplied by parser)
    Directives for theme 'Helium' (supplied by theme)
    Extensions for theme 'Helium' (supplied by theme)
Theme:
    Helium
Settings:
    Strict Mode: false
    Accept Raw Content: false
    Render Formatted: true
Sources:
    Markup File(s)
        /README.md: file '/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src/README.md'
    Template(s)
        -
    Configuration Files(s)
        -
    CSS for PDF
        -
    Copied File(s)
        -
    Root Directories
        /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src
Target:
    Directory '/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site'
```

1. Just providing a README with a title and no configured (Helium) theme
   1. Only a `helium` folder is created in the target with content
   1. No other files are copied
      1.  Result printed 
          ```scala
           RenderedTreeRoot(
             RenderedTree(/,None,List(),None),
             Right(TemplateRoot - TemplateSpans: 1 . 
               TemplateContextReference(
                   cursor.currentDocument.content,true,laika.parse.GeneratedSource$@5af5d76f)
             ),
            laika.config.ObjectConfig@3d5596e2,OutputContext(html,html),
            ConfigurablePathTranslator(
               TranslatorConfig(None,README,index,None),
               OutputContext(html,html),/refPath,<function1>),
               StyleDeclarationSet(Set(/),
               Set(),
               laika.bundle.Precedence$High$@1eb9a3ef),
               None,
               Vector(
                   BinaryInput(Stream(..),/helium/laika-helium.js,Selected(TreeSet(html)),None), 
                   BinaryInput(Stream(..),/helium/landing.page.css,Selected(TreeSet(html)),None), 
                   BinaryInput(Stream(..),/helium/icofont.min.css,Selected(TreeSet(html)),None), 
                   BinaryInput(Stream(..),/helium/fonts/icofont.woff,All,None), 
                   BinaryInput(Stream(..),/helium/fonts/icofont.woff2,All,None), 
                   BinaryInput(Stream(..),/helium/laika-helium.css,Selected(TreeSet(html)),None)
                  )
          )
           ``` 
   1. `syncResult1.allDocuments.mkString(",\n")` is empty    

### README and 1 simple Markdown file

This is the same as above but we now add a single additional Markdown source file.

```
syncResult: Parser(s):
  Markdown
Renderer:
  HTML
Extension Bundles:
  Laika's Default Extensions (supplied by library)
  Laika's directive support (supplied by library)
  Laika's built-in directives (supplied by library)
  Document Type Matcher for Markdown (supplied by parser)
  Github-flavored Markdown (supplied by parser)
  Directives for theme 'Helium' (supplied by theme)
  Extensions for theme 'Helium' (supplied by theme)
Theme:
  Helium
Settings:
  Strict Mode: false
  Accept Raw Content: false
  Render Formatted: true
Sources:
  Markup File(s)
    /README.md: file '/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src/README.md'
    /faq.md: file '/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src/faq.md'
  Template(s)
    -
  Configuration Files(s)
    -
  CSS for PDF
    -
  Copied File(s)
    -
  Root Directories
    /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src
Target:
  Directory '/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site'
```

Same result as above.

### Using version 0.18.2 with 1 README and 1 Markdown file

Se get the same result as above

```
Parser(s):
  Markdown
Renderer:
  HTML
Extension Bundles:
  Laika's Default Extensions (supplied by library)
  Laika's directive support (supplied by library)
  Laika's built-in directives (supplied by library)
  Document Type Matcher for Markdown (supplied by parser)
  Github-flavored Markdown (supplied by parser)
  Directives for theme 'Helium' (supplied by theme)
  Extensions for theme 'Helium' (supplied by theme)
Settings:
  Strict Mode: false
  Accept Raw Content: false
  Render Formatted: true
Sources:
  Markup File(s)
    File '/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src/faq.md'
    File '/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src/README.md'
  Template(s)
    -
  Configuration Files(s)
    -
  CSS for PDF
    -
  Copied File(s)
    -
  Root Directories
    /mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site_src
Target:
  Directory '/mnt/ssd2/hmf/IdeaProjects/storch/out/docs/laika.dest/site'
syncResult1: RenderedTreeRoot(RenderedTree(/,None,List(),None),
TemplateRoot - TemplateSpans: 1
. TemplateContextReference(cursor.currentDocument.content,true,laika.parse.GeneratedSource$@5922c88b)
,laika.config.ObjectConfig@70b2bc14,StyleDeclarationSet(Set(/),Set(),laika.bundle.Precedence$High$@60b5e7e9),None,Vector(BinaryInput(/helium/laika-helium.js,Allocate(cats.effect.kernel.Resource$$$Lambda$6140/0x00007f03452bb548@7ff806d3),Selected(TreeSet(html)),None), BinaryInput(/helium/landing.page.css,Allocate(cats.effect.kernel.Resource$$$Lambda$6140/0x00007f03452bb548@3051ff37),Selected(TreeSet(html)),None), BinaryInput(/helium/icofont.min.css,Allocate(cats.effect.kernel.Resource$$$Lambda$6140/0x00007f03452bb548@40e44bc7),Selected(TreeSet(html)),None), BinaryInput(/helium/fonts/icofont.woff,Allocate(cats.effect.kernel.Resource$$$Lambda$6140/0x00007f03452bb548@f21669d),All,None), BinaryInput(/helium/fonts/icofont.woff2,Allocate(cats.effect.kernel.Resource$$$Lambda$6140/0x00007f03452bb548@30503734),All,None), BinaryInput(/helium/laika-helium.css,Eval(IO(...)),Selected(TreeSet(html)),None)))
allDocuments:
```


https://github.com/sbt/sbt-site
https://github.com/sbt/sbt-site/blob/main/laika/src/sbt-test/laika/blog-post/project/CustomDirectives.scala
https://github.com/sbt/sbt-site/blob/main/laika/src/main/scala/com/typesafe/sbt/site/laika/LaikaSitePlugin.scala

import laika.sbt.LaikaPlugin.autoImport.{ Laika, laikaHTML, laikaSite }
import laika.sbt.LaikaPlugin

https://github.com/typelevel/Laika/blob/main/sbt/src/main/scala/laika/sbt/LaikaPlugin.scala
https://github.com/typelevel/Laika/blob/main/sbt/src/main/scala/laika/sbt/Tasks.scala
