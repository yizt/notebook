

## 命令行参数解析



```scala
import java.io.File
case class Config(foo: Int = -1, out: File = new File("."), xyz: Boolean = false,
  libName: String = "", maxCount: Int = -1, verbose: Boolean = false, debug: Boolean = false,
  mode: String = "", files: Seq[File] = Seq(), keepalive: Boolean = false,
  jars: Seq[File] = Seq(), kwargs: Map[String,String] = Map())
```





```scala
val parser = new scopt.OptionParser[Config]("scopt") {
  head("scopt", "3.x")

  opt[Int]('f', "foo").action( (x, c) =>
    c.copy(foo = x) ).text("foo is an integer property")

  opt[File]('o', "out").required().valueName("").
    action( (x, c) => c.copy(out = x) ).
    text("out is a required file property")

  opt[(String, Int)]("max").action({
      case ((k, v), c) => c.copy(libName = k, maxCount = v) }).
    validate( x =>
      if (x._2 > 0) success
      else failure("Value  must be >0") ).
    keyValueName("", "").
    text("maximum count for ")

  opt[Seq[File]]('j', "jars").valueName(",...").action( (x,c) =>
    c.copy(jars = x) ).text("jars to include")

  opt[Map[String,String]]("kwargs").valueName("k1=v1,k2=v2...").action( (x, c) =>
    c.copy(kwargs = x) ).text("other arguments")

  opt[Unit]("verbose").action( (_, c) =>
    c.copy(verbose = true) ).text("verbose is a flag")

  opt[Unit]("debug").hidden().action( (_, c) =>
    c.copy(debug = true) ).text("this option is hidden in the usage text")

  help("help").text("prints this usage text")

  arg[File]("...").unbounded().optional().action( (x, c) =>
    c.copy(files = c.files :+ x) ).text("optional unbounded args")

  note("some notes.".newline)

  cmd("update").action( (_, c) => c.copy(mode = "update") ).
    text("update is a command.").
    children(
      opt[Unit]("not-keepalive").abbr("nk").action( (_, c) =>
        c.copy(keepalive = false) ).text("disable keepalive"),
      opt[Boolean]("xyz").action( (x, c) =>
        c.copy(xyz = x) ).text("xyz is a boolean property"),
      opt[Unit]("debug-update").hidden().action( (_, c) =>
        c.copy(debug = true) ).text("this option is hidden in the usage text"),
      checkConfig( c =>
        if (c.keepalive && c.xyz) failure("xyz cannot keep alive")
        else success )
    )
}

// parser.parse returns Option[C]
parser.parse(args, Config()) match {
  case Some(config) =>
    // do stuff

  case None =>
    // arguments are bad, error message will have been displayed
}
```





参考：http://www.learn4master.com/big-data/spark/spark-program-use-scopt-parse-arguments

​         http://scopt.github.io/scopt/3.5.0/api/index.html#scopt.OptionParser

​         https://github.com/scopt/scopt