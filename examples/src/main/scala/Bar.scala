/**
  * 
  * ./mill examples.runMain barMain "Hello"
  * ./mill examples.runMain barMain1 101
  * ./mill examples.runMain Bar
  */
object Bar:

  // Becomes a main class barMain
  @main
  def barMain(text: String) =
    println("Hello from text @main Bar")

  // Becomes a main class barMain1
  @main
  def barMain1(number: Int) =
    println("Hello from number @main Bar")

  // Standard Bar class
  def main(args: Array[String]): Unit = 
    println("Hello from static Bar main")
