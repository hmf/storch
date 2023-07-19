/**
  * 
  * ./mill examples.runMain main "Hello"
  *  [114/114] examples.runMain 
  *  Hello from @main Foo
  *  
  * ./mill examples.runMain main1 101
  * [114/114] examples.runMain 
  * Hello from number @main Foo
  * 
  */
object Foo:

  @main
  def main(text: String) =
    println("Hello from text @main Foo")

  @main
  def main1(number: Int) =
    println("Hello from number @main Foo")

  // def main(args: Array[String]): Unit = 
  //   println("Hello from static Foo main")
