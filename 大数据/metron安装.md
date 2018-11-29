[TOC]

### 依赖

```
yum install bzip2
```





```
curl -sL https://rpm.nodesource.com/setup_11.x | bash -
yum install -y nodejs

## 镜像替换
npm config set registry http://registry.npm.taobao.org/
yarn config set registry http://registry.npm.taobao.org/
npm get registry
```



## 修改

a)

b)



```

git checkout Metron_0.6.0

mvn clean install -PHDP-2.6.5.0

mvn clean install -PHDP-2.6.5.0 -DskipTests -Drat.skip=true
```



rpm打包

依赖docker

```
cd metron-deployment/packaging/docker/rpm-docker
mvn clean package -DskipTests -Pbuild-rpms
```



结果如下：

```
[root@master rpm-docker]# ll RPMS/noarch
total 865380
-rw-r--r-- 1 root root   2471044 Nov  9 08:56 metron-alerts-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root  44029768 Nov  9 08:54 metron-common-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root   4550548 Nov  9 08:56 metron-config-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root 100447324 Nov  9 08:55 metron-data-management-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root 115671208 Nov  9 08:55 metron-elasticsearch-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root  88394708 Nov  9 08:55 metron-enrichment-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root     35044 Nov  9 08:55 metron-indexing-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root  18098944 Nov  9 08:56 metron-maas-service-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root  25183480 Nov  9 08:55 metron-metron-management-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root  90001768 Nov  9 08:54 metron-parsers-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root  82714448 Nov  9 08:56 metron-pcap-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root   2055536 Nov  9 08:55 metron-performance-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root  87716636 Nov  9 08:56 metron-profiler-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root 133512244 Nov  9 08:56 metron-rest-0.6.0-201811090053.noarch.rpm
-rw-r--r-- 1 root root  91234760 Nov  9 08:55 metron-solr-0.6.0-201811090053.noarch.rpm
```



```
mkdir /localrepo
cp -rp /root/metron/metron-deployment/packaging/docker/rpm-docker/RPMS/noarch/*.rpm /localrepo/
yum install createrepo
cd /localrepo
createrepo .
```





mpack

```

cd metron-deployment/packaging/ambari/metron-mpack
mvn clean package -Pmpack -DskipTests
```



结果如下：

```
[root@master metron-mpack]# ll target/
total 76
drwxr-xr-x 2 root root  4096 Nov  9 09:09 archive-tmp
drwxr-xr-x 4 root root  4096 Nov  9 09:09 classes
-rw-r--r-- 1 root root 62454 Nov  9 09:09 metron_mpack-0.6.0.0.tar.gz
-rw-r--r-- 1 root root  2326 Nov  9 09:09 mpack.json
```



安装

```
ambari-server install-mpack --mpack=metron_mpack-0.6.0.0.tar.gz --verbose

ambari-server uninstall-mpack --mpack-name=metron-ambari.mpack --verbose
```

结果如下：

```
... ...
process_pid=22386
INFO: about to run command: chown  -R -L root /var/lib/ambari-server/resources/common-services
INFO: 
process_pid=22387
INFO: about to run command: chown  -R -L root /var/lib/ambari-server/resources/mpacks
INFO: 
process_pid=22388
INFO: about to run command: chown  -R -L root /var/lib/ambari-server/resources/mpacks/cache
INFO: 
process_pid=22389
INFO: about to run command: chown  -R -L root /var/lib/ambari-server/resources/dashboards
INFO: 
process_pid=22390
INFO: Management pack metron-ambari.mpack-0.6.0.0 successfully installed! Please restart ambari-server.
INFO: Loading properties from /etc/ambari-server/conf/ambari.properties
Ambari Server 'install-mpack' completed successfully.

```



pg用户创建

```
sudo -u postgres psql
alter user postgres password '123456';
create database metron;
CREATE USER metron WITH PASSWORD '123456';
GRANT ALL PRIVILEGES ON DATABASE metron TO metron;
```

pg创建表

```
CREATE TABLE users(
   username varchar(20) NOT NULL,
   password varchar(20) NOT NULL,
   enabled boolean NOT NULL DEFAULT FALSE,
   primary key(username)
);


create table authorities (
	username varchar(50) not null,
	authority varchar(50) not null,
	FOREIGN KEY (username) REFERENCES users (username)
);
create unique index ix_auth_username on authorities (username,authority);

create table user_roles (
  user_role_id SERIAL PRIMARY KEY,
  username varchar(20) NOT NULL,
  role varchar(20) NOT NULL,
  UNIQUE (username,role),
  FOREIGN KEY (username) REFERENCES users (username)
);
```

参考：https://docs.spring.io/spring-security/site/docs/5.0.4.RELEASE/reference/htmlsingle/#user-schema

https://grokonez.com/spring-framework/spring-security/use-spring-security-jdbc-authentication-postgresql-spring-boot



```
jdbc:postgresql://localhost:5432/metron
org.postgresql.Driver
metron
123456
postgresql
/usr/lib/ambari-server/postgresql-9.3-1101-jdbc4.jar
```







build rpm

参考：https://github.com/apache/metron/tree/master/metron-deployment/packaging/docker/rpm-docker



MPack

参考：https://github.com/apache/metron/tree/master/metron-deployment/packaging/ambari/metron-mpack





## 使用

### snort

```
yum install https://www.snort.org/downloads/snort/snort-2.9.12-1.centos7.x86_64.rpm
ln -s /usr/lib64/libdnet.so.1 /usr/lib64/libdnet.1
```



### squid

```

bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic squid

/usr/metron/0.6.0/bin/zk_load_configs.sh --mode PUSH -i /usr/metron/0.6.0/config/zookeeper -z localhost:2181

```





## 问题

1：jcenter仓库下载特别慢

解决方法：修改pom.xml文件

```
        <repository>
          <id>jcenter</id>
          <!--<url>https://jcenter.bintray.com/</url> -->
		  <url>http://maven.aliyun.com/repository/jcenter</url>
        </repository>
        
        <id>apache.snapshots</id>
            <name>Apache Development Snapshot Repository</name>
            <url><!--https://repository.apache.org/content/repositories/snapshots/-->
			https://maven.aliyun.com/repository/spring-plugin
		</url>
```



2: 编译mvn install -PHDP-2.6.5.0 -DskipTests提示

```
[WARNING] The requested profile "HDP-2.6.5.0" could not be activated because it does not exist.
```





3: 编译mvn install -PHDP-2.6.5.0 -DskipTests报错

```
[WARNING] The requested profile "HDP-2.6.5.0" could not be activated because it does not exist.
[ERROR] Failed to execute goal org.apache.rat:apache-rat-plugin:0.13-SNAPSHOT:check (default) on project metron-interface: Too many files with unapproved license: 5534 See RAT report in: /root/metron/metron-interface/target/rat.txt -> [Help 1]
[ERROR] 
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR] 
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/MojoFailureException

```



解决方法：增加 -Drat.skip=true



4：mvn编译报错

```
[WARNING] The requested profile "HDP-2.6.5.0" could not be activated because it does not exist.
[ERROR] Failed to execute goal on project metron-rest-client: Could not resolve dependencies for project org.apache.metron:metron-rest-client:jar:0.6.0: Could not find artifact io.confluent:kafka-avro-serializer:jar:1.0 in nexus-aliyun (http://maven.aliyun.com/nexus/content/groups/public) -> [Help 1]
[ERROR] 
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR] 
[ERROR] For more information about the errors and possible solutions, please read the following articles:
[ERROR] [Help 1] http://cwiki.apache.org/confluence/display/MAVEN/DependencyResolutionException
[ERROR] 
[ERROR] After correcting the problems, you can resume the build with the command
[ERROR]   mvn <goals> -rf :metron-rest-client

```



```
   <mirror>
        <id>nexus-aliyun</id>
        <mirrorOf>nexus</mirrorOf>
        <name>Nexus aliyun</name>
        <url>http://maven.aliyun.com/nexus/content/groups/public</url>
    </mirror>
    <mirror>
        <id>central-aliyun</id>
        <mirrorOf>central</mirrorOf>
        <name>central aliyun</name>
        <url>https://maven.aliyun.com/repository/central</url>
    </mirror>
    <mirror>
        <id>apache-aliyun</id>
        <mirrorOf>apache snapshots</mirrorOf>
        <name>apache snapshots aliyun</name>
        <url>https://maven.aliyun.com/repository/apache-snapshots</url>
    </mirror>
    <mirror>
        <id>snapshots-aliyun</id>
        <mirrorOf>snapshots</mirrorOf>
        <name>apache snapshots aliyun</name>
        <url>https://maven.aliyun.com/repository/snapshots</url>
    </mirror>
    <mirror>
        <id>releases-aliyun</id>
        <mirrorOf>releases</mirrorOf>
        <name>apache snapshots aliyun</name>
        <url>https://maven.aliyun.com/repository/releases</url>
    </mirror>
       
    <mirror>
        <id>public-aliyun</id>
        <mirrorOf>public</mirrorOf>
        <name>apache snapshots aliyun</name>
        <url>https://maven.aliyun.com/repository/public</url>
    </mirror>
    <mirror>
        <id>spring-aliyun</id>
        <mirrorOf>spring</mirrorOf>
        <name>apache snapshots aliyun</name>
        <url>https://maven.aliyun.com/repository/spring</url>
    </mirror>
    <mirror>
      <!--This is used to direct the public snapshots repo in the 
          profile below over to a different nexus group -->
      <id>nexus-public-snapshots</id>
      <mirrorOf>public-snapshots</mirrorOf> 
      <url>http://maven.aliyun.com/nexus/content/repositories/snapshots/</url>
    </mirror>


```



5: Metron Indexing Install报错

err:

```
Traceback (most recent call last):
  File "/var/lib/ambari-agent/cache/common-services/METRON/0.6.0/package/scripts/indexing_master.py", line 19, in <module>
    import requests
ImportError: No module named requests
```

out:

```
2018-11-09 02:12:29,388 - Repository['HDP-2.6-repo-1'] {'append_to_file': False, 'base_url': 'http://public-repo-1.hortonworks.com/HDP/centos7/2.x/updates/2.6.5.0', 'action': ['create'], 'components': [u'HDP', 'main'], 'repo_template': '[{{repo_id}}]\nname={{repo_id}}\n{% if mirror_list %}mirrorlist={{mirror_list}}{% else %}baseurl={{base_url}}{% endif %}\n\npath=/\nenabled=1\ngpgcheck=0', 'repo_file_name': 'ambari-hdp-1', 'mirror_list': None}
2018-11-09 02:12:29,393 - File['/etc/yum.repos.d/ambari-hdp-1.repo'] {'content': '[HDP-2.6-repo-1]\nname=HDP-2.6-repo-1\nbaseurl=http://public-repo-1.hortonworks.com/HDP/centos7/2.x/updates/2.6.5.0\n\npath=/\nenabled=1\ngpgcheck=0'}
2018-11-09 02:12:29,393 - Writing File['/etc/yum.repos.d/ambari-hdp-1.repo'] because contents don't match
2018-11-09 02:12:29,394 - Repository['HDP-2.6-GPL-repo-1'] {'append_to_file': True, 'base_url': 'http://public-repo-1.hortonworks.com/HDP-GPL/centos7/2.x/updates/2.6.5.0', 'action': ['create'], 'components': [u'HDP-GPL', 'main'], 'repo_template': '[{{repo_id}}]\nname={{repo_id}}\n{% if mirror_list %}mirrorlist={{mirror_list}}{% else %}baseurl={{base_url}}{% endif %}\n\npath=/\nenabled=1\ngpgcheck=0', 'repo_file_name': 'ambari-hdp-1', 'mirror_list': None}
2018-11-09 02:12:29,396 - File['/etc/yum.repos.d/ambari-hdp-1.repo'] {'content': '[HDP-2.6-repo-1]\nname=HDP-2.6-repo-1\nbaseurl=http://public-repo-1.hortonworks.com/HDP/centos7/2.x/updates/2.6.5.0\n\npath=/\nenabled=1\ngpgcheck=0\n[HDP-2.6-GPL-repo-1]\nname=HDP-2.6-GPL-repo-1\nbaseurl=http://public-repo-1.hortonworks.com/HDP-GPL/centos7/2.x/updates/2.6.5.0\n\npath=/\nenabled=1\ngpgcheck=0'}
2018-11-09 02:12:29,396 - Writing File['/etc/yum.repos.d/ambari-hdp-1.repo'] because contents don't match
2018-11-09 02:12:29,396 - Repository['HDP-UTILS-1.1.0.22-repo-1'] {'append_to_file': True, 'base_url': 'http://public-repo-1.hortonworks.com/HDP-UTILS-1.1.0.22/repos/centos7', 'action': ['create'], 'components': [u'HDP-UTILS', 'main'], 'repo_template': '[{{repo_id}}]\nname={{repo_id}}\n{% if mirror_list %}mirrorlist={{mirror_list}}{% else %}baseurl={{base_url}}{% endif %}\n\npath=/\nenabled=1\ngpgcheck=0', 'repo_file_name': 'ambari-hdp-1', 'mirror_list': None}
2018-11-09 02:12:29,399 - File['/etc/yum.repos.d/ambari-hdp-1.repo'] {'content': '[HDP-2.6-repo-1]\nname=HDP-2.6-repo-1\nbaseurl=http://public-repo-1.hortonworks.com/HDP/centos7/2.x/updates/2.6.5.0\n\npath=/\nenabled=1\ngpgcheck=0\n[HDP-2.6-GPL-repo-1]\nname=HDP-2.6-GPL-repo-1\nbaseurl=http://public-repo-1.hortonworks.com/HDP-GPL/centos7/2.x/updates/2.6.5.0\n\npath=/\nenabled=1\ngpgcheck=0\n[HDP-UTILS-1.1.0.22-repo-1]\nname=HDP-UTILS-1.1.0.22-repo-1\nbaseurl=http://public-repo-1.hortonworks.com/HDP-UTILS-1.1.0.22/repos/centos7\n\npath=/\nenabled=1\ngpgcheck=0'}
2018-11-09 02:12:29,399 - Writing File['/etc/yum.repos.d/ambari-hdp-1.repo'] because contents don't match
2018-11-09 02:12:29,399 - Repository['METRON-0.6.0-repo-1'] {'append_to_file': True, 'base_url': 'file:///localrepo', 'action': ['create'], 'components': [u'METRON', 'main'], 'repo_template': '[{{repo_id}}]\nname={{repo_id}}\n{% if mirror_list %}mirrorlist={{mirror_list}}{% else %}baseurl={{base_url}}{% endif %}\n\npath=/\nenabled=1\ngpgcheck=0', 'repo_file_name': 'ambari-hdp-1', 'mirror_list': None}
2018-11-09 02:12:29,401 - File['/etc/yum.repos.d/ambari-hdp-1.repo'] {'content': ...}
2018-11-09 02:12:29,402 - Writing File['/etc/yum.repos.d/ambari-hdp-1.repo'] because contents don't match
2018-11-09 02:12:29,402 - Package['unzip'] {'retry_on_repo_unavailability': False, 'retry_count': 5}
2018-11-09 02:12:29,447 - Skipping installation of existing package unzip
2018-11-09 02:12:29,447 - Package['curl'] {'retry_on_repo_unavailability': False, 'retry_count': 5}
2018-11-09 02:12:29,452 - Skipping installation of existing package curl
2018-11-09 02:12:29,452 - Package['hdp-select'] {'retry_on_repo_unavailability': False, 'retry_count': 5}
2018-11-09 02:12:29,458 - Skipping installation of existing package hdp-select
2018-11-09 02:12:29,460 - The repository with version 2.6.5.0-292 for this command has been marked as resolved. It will be used to report the version of the component which was installed
2018-11-09 02:12:29,464 - Skipping stack-select on METRON because it does not exist in the stack-select package structure.

Command failed after 1 tries
```



解决方法: 先安装pip然后用pip安装requests

```
pip install requests
```



6: metron启动报错

```
Traceback (most recent call last):
  File "/var/lib/ambari-agent/cache/common-services/METRON/0.6.0/package/scripts/enrichment_master.py", line 121, in <module>
    Enrichment().execute()
  File "/usr/lib/ambari-agent/lib/resource_management/libraries/script/script.py", line 375, in execute
    method(env)
  File "/var/lib/ambari-agent/cache/common-services/METRON/0.6.0/package/scripts/enrichment_master.py", line 62, in start
    self.configure(env)
  File "/usr/lib/ambari-agent/lib/resource_management/libraries/script/script.py", line 120, in locking_configure
    original_configure(obj, *args, **kw)
  File "/var/lib/ambari-agent/cache/common-services/METRON/0.6.0/package/scripts/enrichment_master.py", line 54, in configure
    metron_service.refresh_configs(params)
  File "/var/lib/ambari-agent/cache/common-services/METRON/0.6.0/package/scripts/metron_service.py", line 209, in refresh_configs
    check_indexer_parameters()
  File "/var/lib/ambari-agent/cache/common-services/METRON/0.6.0/package/scripts/metron_service.py", line 592, in check_indexer_parameters
    raise Fail("Missing required indexing parameters(s): indexer={0}, missing={1}".format(indexer, missing))
resource_management.core.exceptions.Fail: Missing required indexing parameters(s): indexer=Elasticsearch, missing=['metron-env/es_hosts']
```



解决方法：设置es主机



8：metron启动报错

```
-4644420460712153391:-5909208420417427394
1317 [main] INFO  o.a.s.u.NimbusClient - Found leader nimbus : sandbox-hdp.hortonworks.com:6627
1325 [main] INFO  o.a.s.s.a.AuthUtils - Got AutoCreds []
1327 [main] INFO  o.a.s.u.NimbusClient - Found leader nimbus : sandbox-hdp.hortonworks.com:6627
java.lang.RuntimeException: Topology with name `yaf` already exists on cluster
	at org.apache.storm.StormSubmitter.submitTopologyAs(StormSubmitter.java:240)
	at org.apache.storm.StormSubmitter.submitTopology(StormSubmitter.java:390)
	at org.apache.storm.StormSubmitter.submitTopology(StormSubmitter.java:162)
	at org.apache.metron.parsers.topology.ParserTopologyCLI.main(ParserTopologyCLI.java:610)
```



```
storm kill enrichment
```



9: Metron启动报错

```
Caused by: java.lang.ClassNotFoundException: org.apache.atlas.storm.hook.StormAtlasHook
	at java.net.URLClassLoader.findClass(URLClassLoader.java:381)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
	at sun.misc.Launcher$AppClassLoader.loadClass(Launcher.java:349)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:357)
	at java.lang.Class.forName0(Native Method)
	at java.lang.Class.forName(Class.java:264)
	at org.apache.storm.StormSubmitter.invokeSubmitterHook(StormSubmitter.java:361)
	... 4 more
```



解决方法：

```
Do you see the "Enable Atlas Hook" Checkbox is enabled?
```

参考：https://community.hortonworks.com/questions/142306/storm-atlas-integration-error-stormatlashook.html



10: metron rest启动报错

```
Caused by: org.postgresql.util.PSQLException: ERROR: relation "users" does not exist
  Position: 13
	at org.postgresql.core.v3.QueryExecutorImpl.receiveErrorResponse(QueryExecutorImpl.java:2161)
	at org.postgresql.core.v3.QueryExecutorImpl.processResults(QueryExecutorImpl.java:1890)
	at org.postgresql.core.v3.QueryExecutorImpl.execute(QueryExecutorImpl.java:255)
	at org.postgresql.jdbc2.AbstractJdbc2Statement.execute(AbstractJdbc2Statement.java:559)
	at org.postgresql.jdbc2.AbstractJdbc2Statement.executeWithFlags(AbstractJdbc2Statement.java:417)
	at org.postgresql.jdbc2.AbstractJdbc2Statement.executeUpdate(AbstractJdbc2Statement.java:363)
	at com.zaxxer.hikari.pool.ProxyPreparedStatement.executeUpdate(ProxyPreparedStatement.java:61)
	at com.zaxxer.hikari.pool.HikariProxyPreparedStatement.executeUpdate(HikariProxyPreparedStatement.java)
	at org.springframework.jdbc.core.JdbcTemplate.lambda$update$0(JdbcTemplate.java:855)
	at org.springframework.jdbc.core.JdbcTemplate.execute(JdbcTemplate.java:605)
	... 51 more

```



参考：https://github.com/apache/metron/tree/master/metron-interface/metron-rest#security

https://docs.spring.io/spring-security/site/docs/5.0.4.RELEASE/reference/htmlsingle/#user-schema

https://grokonez.com/spring-framework/spring-security/use-spring-security-jdbc-authentication-postgresql-spring-boot