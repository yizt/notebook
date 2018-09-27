



## 特征提取



## 特征转换



## 特征选择

### RRormula

官网地址： [RFormula](http://spark.apache.org/docs/latest/ml-features.html#rformula) 

​              RFormula通过R模型公式来选择列。支持R操作中的部分操作，包括‘~’, ‘.’, ‘:’, ‘+’以及‘-‘，基本操作如下：

1、 **~**分隔目标和对象

2、**+**合并对象，“+ 0”意味着删除空格

3、 **:**交互（数值相乘，类别二值化）

4、**.** 除了目标外的全部列

假设a和b为两列：

　　1、y ~ a + b表示模型y ~ w0 + w1 * a +w2 * b其中w0为截距，w1和w2为相关系数。

　　2、 y ~a + b + a:b – 1表示模型y ~ w1* a + w2 * b + w3 * a * b，其中w1，w2，w3是相关系数。

　　RFormula产生一个向量特征列以及一个double或者字符串标签列。如果类别列是字符串类型，它将通过StringIndexer转换为double类型。如果标签列不存在，则输出中将通过规定的响应变量创造一个标签列。

​        实际上是将所有特征转为feature列，类别转为label列

```scala
    val rFormula = new RFormula
    val formula: RFormula = rFormula.setFormula("Species ~ SepalLength + SepalWidth + PetalLength + PetalWidth")
```

结果如下：

```tiki wiki
|SepalLength|SepalWidth|PetalLength|PetalWidth|Species|         features|label| rawPrediction|  probability|prediction|
+-----------+----------+-----------+----------+-------+-----------------+-----+--------------+-------------+----------+
|        5.1|       3.5|        1.4|       0.2| setosa|[5.1,3.5,1.4,0.2]|  2.0|[0.0,0.0,50.0]|[0.0,0.0,1.0]|       2.0|
|        4.9|       3.0|        1.4|       0.2| setosa|[4.9,3.0,1.4,0.2]|  2.0|[0.0,0.0,50.0]|[0.0,0.0,1.0]|       2.0|
|        4.7|       3.2|        1.3|       0.2| setosa|[4.7,3.2,1.3,0.2]|  2.0|[0.0,0.0,50.0]|[0.0,0.0,1.0]|       2.0|
```

