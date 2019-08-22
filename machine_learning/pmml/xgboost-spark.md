

a) 修改pom版本号

```
<dependency>
			<groupId>ml.dmlc</groupId>
			<artifactId>xgboost4j-spark</artifactId>
			<version>0.80</version>
			<scope>provided</scope>
		</dependency>

		<dependency>
			<groupId>org.apache.spark</groupId>
			<artifactId>spark-core_2.11</artifactId>
			<version>2.3.0</version>
			<scope>provided</scope>
		</dependency>
		<dependency>
			<groupId>org.apache.spark</groupId>
			<artifactId>spark-mllib_2.11</artifactId>
			<version>2.3.0</version>
			<scope>provided</scope>
			<exclusions>
				<exclusion>
					<groupId>org.jpmml</groupId>
					<artifactId>pmml-model</artifactId>
				</exclusion>
			</exclusions>
		</dependency>
```



b)  修改src/main/java/org/jpmml/sparkml/xgboost/XGBoostClassificationModelConverter.java

```java
-               Booster booster = model.booster();
+               Booster booster = model.nativeBooster();
```



c) 修改src/main/java/org/jpmml/sparkml/xgboost/XGBoostRegressionModelConverter.java

```java
-               Booster booster = model.booster();
+               Booster booster = model.nativeBooster();

```



d) 本地安装

```
mvn clean install
```





```xml
  <repositories>
    <repository>
      <id>XGBoost4J-Spark Snapshot Repo</id>
      <name>XGBoost4J-Spark Snapshot Repo</name>
      <url>https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/</url>
    </repository>
  </repositories>
```

