

a) cython使用

setup.py文件

```python
import setuptools
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension(
        'video.fcn.compute_overlap',
        ['video/fcn/compute_overlap.pyx'],
        include_dirs=[np.get_include()]
    ),
]

setuptools.setup(
    ext_modules    = extensions,
    setup_requires = ["cython>=0.28"]
)
```



在引用文件中

```python
import pyximport
pyximport.install()
```



执行如下命令

```powershell
python setup.py build_ext -i
```



2. python setup.py build_ext -i

   ```
   (base) D:\workspace\numpy_neuron_network>python setup.py build_ext -i
   running build_ext
   skipping 'nn\clayers.c' Cython extension (up-to-date)
   building 'nn.clayers' extension
   C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\BIN\x86_amd64\cl.exe /c /nologo /Ox /W3 /GL /DNDEBUG /MD -ID:\Anaconda3\lib\site-packages\numpy\core\include -ID:\Anaconda3\include -ID:\Anaconda3\include "-IC:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\INCLUDE" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.10240.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\NETFXSDK\4.6.1\include\um" /Tcnn\clayers.c /Fobuild\temp.win-amd64-3.6\Release\nn\clayers.obj
   clayers.c
   d:\anaconda3\include\pyconfig.h(222): fatal error C1083: Cannot open include file: 'basetsd.h': No such file or directory
   error: command 'C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\BIN\\x86_amd64\\cl.exe' failed with exit status 2

   (base) D:\workspace\numpy_neuron_network>
   ```

3.  conv_forward(X,weights["K1"],weights["b1"])

   ```python
   <ipython-input-1-0e22459fd448> in forward(X)
        20 # 定义前向传播
        21 def forward(X):
   ---> 22     nuerons["conv1"]=conv_forward(X,weights["K1"],weights["b1"])
        23     nuerons["conv1_relu"]=relu_forward(nuerons["conv1"])
        24 

   ~/numpy_neuron_network/nn/clayers.pyx in nn.clayers.conv_forward()
        13 
        14 
   ---> 15 def conv_forward(np.ndarray[double, ndim=4] z,
        16                  np.ndarray[double, ndim=4] K,
        17                  np.ndarray[double, ndim=1] b,

   ValueError: Buffer dtype mismatch, expected 'double' but got 'float'

   ```

   解决方法：增加.astype(np.float64)

   ```python
   weights["K1"]=weights_scale*np.random.randn(1,f1,3,3).astype(np.float64)
   weights["b1"]=np.zeros(f1).astype(np.float64)
   weights["W2"]=weights_scale*np.random.randn(f1*13*13,10).astype(np.float64)
   weights["b2"]=np.zeros(10).astype(np.float64)
   ```

   ​

   3. cython并未加速

      ```

      ```

      参考：http://docs.cython.org/en/latest/src/quickstart/cythonize.html

      http://docs.cython.org/en/latest/src/userguide/early_binding_for_speed.html

      http://docs.cython.org/en/latest/src/tutorial/numpy.html


2. 创建scala项目，使用骨架原型：scala-archetype-simple

```shell
mvn archetype:generate -DgroupId=com.es -DartifactId=risk -Dversion=0.0.1 -DarchetypeArtifactId=scala-quickstart-archetype -DarchetypeGroupId=pl.org.miki -DinteractiveMode=false 
```

3.  File /lib/xgboost4j.dll was not found inside JAR.

   sdfs

4. ​

   ```shell
   Caused by: java.lang.UnsatisfiedLinkError: ml.dmlc.xgboost4j.java.XGBoostJNI.RabitInit([Ljava/lang/String;)I
   at ml.dmlc.xgboost4j.java.XGBoostJNI.RabitInit(Native Method)
   at ml.dmlc.xgboost4j.java.Rabit.init(Rabit.java:65)
   at ml.dmlc.xgboost4j.scala.spark.XGBoost$$anonfun$buildDistributedBoosters$1.apply(XGBoost.scala:130)
   at ml.dmlc.xgboost4j.scala.spark.XGBoost$$anonfun$buildDistributedBoosters$1.apply(XGBoost.scala:116)
   at org.apache.spark.rdd.ZippedPartitionsRDD2.compute(ZippedPartitionsRDD.scala:89)
   at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:324)
   at org.apache.spark.rdd.RDD$$anonfun$7.apply(RDD.scala:337)

   ```
   https://github.com/dmlc/xgboost/issues/2148

   下载https://github.com/criteo-forks/xgboost-jars/releases  包中的xgboost4j.dll 放到mvn库中的包中

   ​

5.   xgboost执行报错

   ```java
   18/09/05 11:48:58 WARN ServletHandler: Error for /api/v1/applications/local-1536119331281/executors
   java.lang.NoSuchMethodError: org.glassfish.jersey.server.ApplicationHandler.<init>(Ljavax/ws/rs/core/Application;Lorg/glassfish/jersey/internal/inject/Binder;Ljava/lang/Object;)V
   at org.glassfish.jersey.servlet.WebComponent.<init>(WebComponent.java:335)
   at org.glassfish.jersey.servlet.ServletContainer.init(ServletContainer.java:178)
   at org.glassfish.jersey.servlet.ServletContainer.init(ServletContainer.java:370)
   at javax.servlet.GenericServlet.init(GenericServlet.java:244)
   at org.spark_project.jetty.servlet.ServletHolder.initServlet(ServletHolder.java:643)
   at org.spark_project.jetty.servlet.ServletHolder.getServlet(ServletHolder.java:499)
   at org.spark_project.jetty.servlet.ServletHolder.ensureInstance(ServletHolder.java:791)
   at org.spark_project.jetty.servlet.ServletHolder.prepare(ServletHolder.java:776)
   at org.spark_project.jetty.servlet.ServletHandler.doHandle(ServletHandler.java:579)
   at org.spark_project.jetty.server.handler.ContextHandler.doHandle(ContextHandler.java:1180)
   at org.spark_project.jetty.servlet.ServletHandler.doScope(ServletHandler.java:512)
   at org.spark_project.jetty.server.handler.ContextHandler.doScope(ContextHandler.java:1112)
   at org.spark_project.jetty.server.handler.ScopedHandler.handle(ScopedHandler.java:141)
   at org.spark_project.jetty.server.handler.gzip.GzipHandler.handle(GzipHandler.java:493)
   at org.spark_project.jetty.server.handler.ContextHandlerCollection.handle(ContextHandlerCollection.java:213)
   at org.spark_project.jetty.server.handler.HandlerWrapper.handle(HandlerWrapper.java:134)
   at org.spark_project.jetty.server.Server.handle(Server.java:534)
   at org.spark_project.jetty.server.HttpChannel.handle(HttpChannel.java:320)
   at org.spark_project.jetty.server.HttpConnection.onFillable(HttpConnection.java:251)
   at org.spark_project.jetty.io.AbstractConnection$ReadCallback.succeeded(AbstractConnection.java:283)
   at org.spark_project.jetty.io.FillInterest.fillable(FillInterest.java:108)
   at org.spark_project.jetty.io.SelectChannelEndPoint$2.run(SelectChannelEndPoint.java:93)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.executeProduceConsume(ExecuteProduceConsume.java:303)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.produceConsume(ExecuteProduceConsume.java:148)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.run(ExecuteProduceConsume.java:136)
   at org.spark_project.jetty.util.thread.QueuedThreadPool.runJob(QueuedThreadPool.java:671)
   at org.spark_project.jetty.util.thread.QueuedThreadPool$2.run(QueuedThreadPool.java:589)
   at java.lang.Thread.run(Thread.java:745)
    
   ```
   解决方法

   ```xml
   <dependency>
               <groupId>org.glassfish.jersey.media</groupId>
               <artifactId>jersey-media-moxy</artifactId>
               <version>2.8</version>
           </dependency>
   ```

   ​

6.  ​

   ```
   18/09/05 13:57:04 WARN ServletHandler: Error for /api/v1/applications/local-1536127019336/executors

   java.lang.NoClassDefFoundError: org/glassfish/jersey/internal/inject/Binder
   at org.glassfish.jersey.servlet.ServletContainer.init(ServletContainer.java:178)
   at org.glassfish.jersey.servlet.ServletContainer.init(ServletContainer.java:370)
   at javax.servlet.GenericServlet.init(GenericServlet.java:244)
   at org.spark_project.jetty.servlet.ServletHolder.initServlet(ServletHolder.java:643)
   at org.spark_project.jetty.servlet.ServletHolder.getServlet(ServletHolder.java:499)
   at org.spark_project.jetty.servlet.ServletHolder.ensureInstance(ServletHolder.java:791)
   at org.spark_project.jetty.servlet.ServletHolder.prepare(ServletHolder.java:776)
   at org.spark_project.jetty.servlet.ServletHandler.doHandle(ServletHandler.java:579)
   at org.spark_project.jetty.server.handler.ContextHandler.doHandle(ContextHandler.java:1180)
   at org.spark_project.jetty.servlet.ServletHandler.doScope(ServletHandler.java:512)
   at org.spark_project.jetty.server.handler.ContextHandler.doScope(ContextHandler.java:1112)
   at org.spark_project.jetty.server.handler.ScopedHandler.handle(ScopedHandler.java:141)
   at org.spark_project.jetty.server.handler.gzip.GzipHandler.handle(GzipHandler.java:493)
   at org.spark_project.jetty.server.handler.ContextHandlerCollection.handle(ContextHandlerCollection.java:213)
   at org.spark_project.jetty.server.handler.HandlerWrapper.handle(HandlerWrapper.java:134)
   at org.spark_project.jetty.server.Server.handle(Server.java:534)
   at org.spark_project.jetty.server.HttpChannel.handle(HttpChannel.java:320)
   at org.spark_project.jetty.server.HttpConnection.onFillable(HttpConnection.java:251)
   at org.spark_project.jetty.io.AbstractConnection$ReadCallback.succeeded(AbstractConnection.java:283)
   at org.spark_project.jetty.io.FillInterest.fillable(FillInterest.java:108)
   at org.spark_project.jetty.io.SelectChannelEndPoint$2.run(SelectChannelEndPoint.java:93)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.executeProduceConsume(ExecuteProduceConsume.java:303)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.produceConsume(ExecuteProduceConsume.java:148)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.run(ExecuteProduceConsume.java:136)
   at org.spark_project.jetty.util.thread.QueuedThreadPool.runJob(QueuedThreadPool.java:671)
   at org.spark_project.jetty.util.thread.QueuedThreadPool$2.run(QueuedThreadPool.java:589)
   at java.lang.Thread.run(Thread.java:745)
   ```
   Caused by: java.lang.ClassNotFoundException: org.glassfish.jersey.internal.inject.Binder
   ```
   at java.net.URLClassLoader.findClass(URLClassLoader.java:381)
   at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
   at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:331)
   at java.lang.ClassLoader.loadClass(ClassLoader.java:357)
   ... 27 more
   ```

7.  ​

8.  sdsdf

9.  ​
   ```java
   18/09/05 14:23:08 WARN ServletHandler: Error for /api/v1/applications/local-1536128583001/executors

   java.lang.NoSuchMethodError: javax.ws.rs.core.Application.getProperties()Ljava/util/Map;

   at org.glassfish.jersey.server.ApplicationHandler.<init>(ApplicationHandler.java:308)
   at org.glassfish.jersey.servlet.WebComponent.<init>(WebComponent.java:335)
   at org.glassfish.jersey.servlet.ServletContainer.init(ServletContainer.java:178)
   at org.glassfish.jersey.servlet.ServletContainer.init(ServletContainer.java:370)
   at javax.servlet.GenericServlet.init(GenericServlet.java:244)
   at org.spark_project.jetty.servlet.ServletHolder.initServlet(ServletHolder.java:643)
   at org.spark_project.jetty.servlet.ServletHolder.getServlet(ServletHolder.java:499)
   at org.spark_project.jetty.servlet.ServletHolder.ensureInstance(ServletHolder.java:791)
   at org.spark_project.jetty.servlet.ServletHolder.prepare(ServletHolder.java:776)
   at org.spark_project.jetty.servlet.ServletHandler.doHandle(ServletHandler.java:579)
   at org.spark_project.jetty.server.handler.ContextHandler.doHandle(ContextHandler.java:1180)
   at org.spark_project.jetty.servlet.ServletHandler.doScope(ServletHandler.java:512)
   at org.spark_project.jetty.server.handler.ContextHandler.doScope(ContextHandler.java:1112)
   at org.spark_project.jetty.server.handler.ScopedHandler.handle(ScopedHandler.java:141)
   at org.spark_project.jetty.server.handler.gzip.GzipHandler.handle(GzipHandler.java:493)
   at org.spark_project.jetty.server.handler.ContextHandlerCollection.handle(ContextHandlerCollection.java:213)
   at org.spark_project.jetty.server.handler.HandlerWrapper.handle(HandlerWrapper.java:134)
   at org.spark_project.jetty.server.Server.handle(Server.java:534)
   at org.spark_project.jetty.server.HttpChannel.handle(HttpChannel.java:320)
   at org.spark_project.jetty.server.HttpConnection.onFillable(HttpConnection.java:251)
   at org.spark_project.jetty.io.AbstractConnection$ReadCallback.succeeded(AbstractConnection.java:283)
   at org.spark_project.jetty.io.FillInterest.fillable(FillInterest.java:108)
   at org.spark_project.jetty.io.SelectChannelEndPoint$2.run(SelectChannelEndPoint.java:93)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.executeProduceConsume(ExecuteProduceConsume.java:303)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.produceConsume(ExecuteProduceConsume.java:148)
   at org.spark_project.jetty.util.thread.strategy.ExecuteProduceConsume.run(ExecuteProduceConsume.java:136)
   at org.spark_project.jetty.util.thread.QueuedThreadPool.runJob(QueuedThreadPool.java:671)
   at org.spark_project.jetty.util.thread.QueuedThreadPool$2.run(QueuedThreadPool.java:589)
   at java.lang.Thread.run(Thread.java:745)
   ```

10.  https://www.jianshu.com/p/dd11e1bb58c3

11.  参考：https://www.jianshu.com/p/dd11e1bb58c3



服务器上运行xgboost报错





	18/09/06 10:17:49 INFO RabitTracker: Tracker Process ends with exit code 0

	Exception in thread "main" java.lang.NoSuchMethodError: org.apache.spark.SparkContext.removeSparkListener(Lorg/apache/spark/scheduler/SparkListenerInterface;)V

	at org.apache.spark.SparkParallelismTracker.safeExecute(SparkParallelismTracker.scala:85)
	at org.apache.spark.SparkParallelismTracker.execute(SparkParallelismTracker.scala:109)
	at ml.dmlc.xgboost4j.scala.spark.XGBoost$$anonfun$trainDistributed$4.apply(XGBoost.scala:238)
	at ml.dmlc.xgboost4j.scala.spark.XGBoost$$anonfun$trainDistributed$4.apply(XGBoost.scala:222)
	at scala.collection.TraversableLike$$anonfun$map$1.apply(TraversableLike.scala:234)
	at scala.collection.TraversableLike$$anonfun$map$1.apply(TraversableLike.scala:234)
	at scala.collection.immutable.List.foreach(List.scala:381)
	at scala.collection.TraversableLike$class.map(TraversableLike.scala:234)
	at scala.collection.immutable.List.map(List.scala:285)
	at ml.dmlc.xgboost4j.scala.spark.XGBoost$.trainDistributed(XGBoost.scala:221)
	at ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier.train(XGBoostClassifier.scala:191)
	at ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier.train(XGBoostClassifier.scala:48)
	at org.apache.spark.ml.Predictor.fit(Predictor.scala:96)
	at com.es.XGBoost$.main(XGBoost.scala:62)
	at com.es.XGBoost.main(XGBoost.scala)
	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.lang.reflect.Method.invoke(Method.java:497)
	at org.apache.spark.deploy.SparkSubmit$.org$apache$spark$deploy$SparkSubmit$$runMain(SparkSubmit.scala:745)
	at org.apache.spark.deploy.SparkSubmit$.doRunMain$1(SparkSubmit.scala:187)
	at org.apache.spark.deploy.SparkSubmit$.submit(SparkSubmit.scala:212)
	at org.apache.spark.deploy.SparkSubmit$.main(SparkSubmit.scala:126)
	at org.apache.spark.deploy.SparkSubmit.main(SparkSubmit.scala)
18/09/06 10:17:49 INFO SparkContext: Invoking stop() from shutdown hook





13:  jpmml-sparkml-xgboost 工程 mvn clean install 

```
[INFO] ------------------------------------------------------------------------
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.6.1:compile (default-compile) on project jpmml-sparkml-xgboost: Compilation failure: Compilation failure:
[ERROR] /D:/esspace/jpmml-sparkml-xgboost/src/main/java/org/jpmml/sparkml/xgboost/XGBoostRegressionModelConverter.java:[37,40] 找不到符号
[ERROR] 符号:   方法 booster()
[ERROR] 位置: 类型为ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel的变量 model
[ERROR] /D:/esspace/jpmml-sparkml-xgboost/src/main/java/org/jpmml/sparkml/xgboost/XGBoostClassificationModelConverter.java:[37,40] 找不到符号
[ERROR] 符号:   方法 booster()
[ERROR] 位置: 类型为ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel的变量 model
[ERROR] -> [Help 1]
[ERROR]
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR]
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException
```





14：

```
Exception in thread "main" java.lang.IllegalArgumentException: Fields org.dmg.pmml.OutputField@1dbf727a and org.dmg.pmml.OutputField@270ab7bc have the same name probability(Iris-setosa)
	at org.jpmml.model.visitors.FieldUtil.nameMap(FieldUtil.java:36)
	at org.jpmml.model.visitors.FieldUtil.selectAll(FieldUtil.java:50)
	at org.jpmml.model.visitors.FieldUtil.selectAll(FieldUtil.java:45)
	at org.jpmml.model.visitors.FieldDependencyResolver.process(FieldDependencyResolver.java:188)
	at org.jpmml.model.visitors.FieldDependencyResolver.visit(FieldDependencyResolver.java:104)
	at org.dmg.pmml.OutputField.accept(OutputField.java:365)
	at org.dmg.pmml.PMMLObject.traverse(PMMLObject.java:86)
	at org.dmg.pmml.Output.accept(Output.java:81)
	at org.dmg.pmml.PMMLObject.traverse(PMMLObject.java:68)
	at org.dmg.pmml.regression.RegressionModel.accept(RegressionModel.java:327)
	at org.dmg.pmml.PMMLObject.traverse(PMMLObject.java:55)
	at org.dmg.pmml.mining.Segment.accept(Segment.java:172)
	at org.dmg.pmml.PMMLObject.traverse(PMMLObject.java:86)
	at org.dmg.pmml.mining.Segmentation.accept(Segmentation.java:123)
	at org.dmg.pmml.PMMLObject.traverse(PMMLObject.java:47)
	at org.dmg.pmml.mining.MiningModel.accept(MiningModel.java:305)
	at org.dmg.pmml.PMMLObject.traverse(PMMLObject.java:86)
	at org.dmg.pmml.PMML.accept(PMML.java:208)
	at org.jpmml.model.visitors.AbstractVisitor.applyTo(AbstractVisitor.java:272)
	at org.jpmml.model.visitors.FieldResolver.applyTo(FieldResolver.java:56)
	at org.jpmml.model.visitors.FieldDependencyResolver.applyTo(FieldDependencyResolver.java:51)
	at org.jpmml.model.visitors.DeepFieldResolver.applyTo(DeepFieldResolver.java:17)
	at org.jpmml.converter.ModelEncoder.encodePMML(ModelEncoder.java:55)
	at org.jpmml.sparkml.PMMLBuilder.build(PMMLBuilder.java:208)
	at org.jpmml.sparkml.PMMLBuilder.buildByteArray(PMMLBuilder.java:278)
	at org.jpmml.sparkml.PMMLBuilder.buildByteArray(PMMLBuilder.java:274)
	at XGBoostTest$.main(XGBoostTest.scala:85)
	at XGBoostTest.main(XGBoostTest.scala)
	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.lang.reflect.Method.invoke(Method.java:498)
	at com.intellij.rt.execution.application.AppMain.main(AppMain.java:144)
18/09/06 16:00:24 INFO SparkContext: Invoking stop() from shutdown hook
```



```scala
    val xgbClassifier = new XGBoostClassifier(xgbParam).
      setFeaturesCol("features").
      setLabelCol("classIndex").setProbabilityCol("pp")   //改一个名字不能用probability
```

