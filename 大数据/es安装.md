[TOC]



启用30天试用

http://192.168.3.191:9200/_xpack/license/trial_status

POST:  http://192.168.3.191:9200/_xpack/license/start_trial?acknowledge=true



http://10.221.123.45:9200/_xpack/license/trial_status



修改用户密码

bin/elasticsearch-setup-passwords interactive -u "http://10.221.123.45:9200"



elastic,kibana,logstash_system,beats_system.





192.168.3.191:9200/_xpack/security/user/remote_monitor



告警配置

https://www.elastic.co/guide/en/elasticsearch/reference/6.2/notification-settings.html



sql可以测试

https://www.elastic.co/guide/en/elasticsearch/reference/6.4/sql-getting-started.html



### ES

模板

```
{
  "index_patterns": ["log*"],
  "settings": {
    "number_of_shards": 1
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": false
      },
      "properties": {
        "fb_offset": {
          "type": "long"
        },
        "date": {
          "type": "date",
          "format": "dd/MM/yy HH:mm:ss:SSS"
        }
      }
    }
  }
}
```



```
{
  "index_patterns": ["log*"],
  "settings": {
    "number_of_shards": 32
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": true
      },
      "properties": {
        "fb_offset": {
          "type": "long"
        },
        "date": {
          "type": "date",
          "format": "dd/MM/yy HH:mm:ss:SSS"
        },
  "time": {
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss,SSS||dd/MMM/yyyy/HH:mm:ss||dd/MMM/yyyy:HH:mm:ss Z" 
        }
      }
    }
  }
}
```





### logstash

需要增加的

```
%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} 
```





```json
input {
    # 从文件读取日志信息
    file {
        path => "/opt/logtest/elasticsearch-6.4.2/logs/log_cluster.log"
        type => "plain"
        start_position => "beginning"
    }
    
}

# filter {
#
# }

output {
    # 输出到 elasticsearch
    elasticsearch {
        user => "logstash_internal"
        password => "123456"
        hosts => ["192.168.3.191:9200"]
        index => "log-test-%{+YYYY.MM.dd}"
    }
}
```



```
filter {

		grok {
			match => ["[fields][message]","%{DATA:remoteIp} %{DATA:RFC} %{DATA:clientFlag} \[%{HTTPDATE:time}\] \"%{DATA:reqLine}\" %{DATA:status} %{DATA:reqBytes} %{DATA:serverName}\:%{DATA:serverPort} \"%{DATA:referer}\" \"%{DATA:cookie}\" \"%{DATA:user-Agent}\" %{DATA:reqHBytes} %{DATA:respHBytes} \- %{DATA:clientIp}\:%{DATA:clientPort} %{NUMBER:RT}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"access"		
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","dd/MMM/yyyy:HH:mm:ss Z","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}

	
	
	mutate {
		add_field => { 
			"topic"=>"accesslog"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}
```







### filebeat



./filebeat -e -c filebeat.yml

```
output.console:
  codec.format:
    string: '%{[hostname]} %{[@timestamp]} %{[source]} %{[offset]} %{[message]}'
```



kafka数据如下：

```
{"@timestamp":"2018-10-24T12:36:44.100Z","@metadata":{"beat":"filebeat","type":"doc","version":"6.4.2","topic":"accesslog"},"input":{"type":"log"},"beat":{"name":"airflow-01.embrace.com","hostname":"airflow-01.embrace.com","version":"6.4.2"},"host":{"name":"airflow-01.embrace.com"},"offset":0,"message":"10.6.141.21 - - [12/Aug/2015:12:53:20 +0800] \"STATUS / HTTP/1.1\" 200 65 vm-vmw2233-app.localdomain:6666 \"-\" \"-\" \"ClusterListener/1.0\" 158 390 - 122.119.123.31:6666 2076","source":"/opt/logtest/logs/accesslog.log","prospector":{"type":"log"}}
{"@timestamp":"2018-10-24T12:36:44.100Z","@metadata":{"beat":"filebeat","type":"doc","version":"6.4.2","topic":"accesslog"},"beat":{"name":"airflow-01.embrace.com","hostname":"airflow-01.embrace.com","version":"6.4.2"},"host":{"name":"airflow-01.embrace.com"},"source":"/opt/logtest/logs/accesslog.log","offset":169,"message":"10.6.141.21 - - [12/Aug/2015:12:53:20 +0800] \"STATUS / HTTP/1.1\" 200 65 vm-vmw2233-app.localdomain:6666 \"-\" \"-\" \"ClusterListener/1.0\" 158 390 - 122.119.123.31:6666 1349","prospector":{"type":"log"},"input":{"type":"log"}}
{"@timestamp":"2018-10-24T12:36:44.100Z","@metadata":{"beat":"filebeat","type":"doc","version":"6.4.2","topic":"accesslog"},"beat":{"name":"airflow-01.embrace.com","hostname":"airflow-01.embrace.com","version":"6.4.2"},"host":{"name":"airflow-01.embrace.com"},"source":"/opt/logtest/logs/accesslog.log","offset":338,"message":"10.6.141.21 - - [12/Aug/2015:12:53:20 +0800] \"STATUS / HTTP/1.1\" 200 59 vm-vmw2233-app.localdomain:6666 \"-\" \"-\" \"ClusterListener/1.0\" 153 384 - 122.119.123.31:6666 1943","prospector":{"type":"log"},"input":{"type":"log"}}
```





### kafka

```
nohup bin/zookeeper-server-start.sh config/zookeeper.properties > logs/zookeeper.log 2>&1 &
nohup bin/kafka-server-start.sh config/server.properties > logs/kafka.log 2>&1 &
```



```
awk -F':' '{ system("./bin/kafka-topics.sh --delete --zookeeper localhost:2181 --topic=" $1 )}' /tmp/topics.txt

./bin/kafka-topics.sh --delete --zookeeper localhost:2181 --topic was_system_out
```



```
awk -F':' '{ system("./bin/kafka-topics.sh --create --zookeeper localhost:2181 --topic=" $1 " --partitions=" 24 " --replication-factor=" 1) }' /tmp/topics.txt

bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic was_system_out
```



消费：

```
bin/kafka-console-consumer.sh --bootstrap-server 10.221.123.1:6667 --topic amq --from-beginning
```





```
apche_accesslog
amq
apache_error
jboss_access
jboss_server
jcf
jcf_adapt
jcf_app_server
jcf_operation
jcf_reg
jcf_regbk
redis
sih_err
sih_info
sih_running
tode
tuxedo
was_gc
was_http_access
was_system_err
was_system_out
wmq_mqm
wmq_mqm_fdc
wmq_qm
```







## 不满足点

c)-5：事务合并查询，基线分析查询

c)-7: 线性回归图，雷达图

e)-1:日志聚类功能挖掘海量日志中异常模式、可以将上百、上千条日志记录聚类成数条

e)-2: 算法提供二次开发接口，根据需要可以把算法加入；聚类算法参数通过界面可调

j)-2: 平台提供二次开发接口，可以扩展新的动态基线和异常检测算法











## 问题记录

1: logstash 启动报错

```
[2018-10-24T14:32:41,483][ERROR][logstash.config.sourceloader] Could not fetch all the sources {:exception=>LogStash::Outputs::ElasticSearch::HttpClient::Pool::BadResponseCodeError, :message=>"Got response code '403' contacting Elasticsearch at URL 'http://192.168.3.191:9200/.logstash/doc/_mget'", :backtrace=>["/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/logstash-output-elasticsearch-9.2.1-java/lib/logstash/outputs/elasticsearch/http_client/manticore_adapter.rb:80:in `perform_request'", "/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/logstash-output-elasticsearch-9.2.1-java/lib/logstash/outputs/elasticsearch/http_client/pool.rb:291:in `perform_request_to_url'", "/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/logstash-output-elasticsearch-9.2.1-java/lib/logstash/outputs/elasticsearch/http_client/pool.rb:278:in `block in perform_request'", "/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/logstash-output-elasticsearch-9.2.1-java/lib/logstash/outputs/elasticsearch/http_client/pool.rb:373:in `with_connection'", "/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/logstash-output-elasticsearch-9.2.1-java/lib/logstash/outputs/elasticsearch/http_client/pool.rb:277:in `perform_request'", "/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/logstash-output-elasticsearch-9.2.1-java/lib/logstash/outputs/elasticsearch/http_client/pool.rb:285:in `block in Pool'", "/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/logstash-output-elasticsearch-9.2.1-java/lib/logstash/outputs/elasticsearch/http_client.rb:167:in `post'", "/opt/logtest/logstash-6.4.2/x-pack/lib/config_management/elasticsearch_source.rb:137:in `fetch_config'", "/opt/logtest/logstash-6.4.2/x-pack/lib/config_management/elasticsearch_source.rb:71:in `pipeline_configs'", "/opt/logtest/logstash-6.4.2/logstash-core/lib/logstash/config/source_loader.rb:61:in `block in fetch'", "org/jruby/RubyArray.java:2481:in `collect'", "/opt/logtest/logstash-6.4.2/logstash-core/lib/logstash/config/source_loader.rb:60:in `fetch'", "/opt/logtest/logstash-6.4.2/logstash-core/lib/logstash/agent.rb:142:in `converge_state_and_update'", "/opt/logtest/logstash-6.4.2/logstash-core/lib/logstash/agent.rb:110:in `block in execute'", "/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/stud-0.0.23/lib/stud/interval.rb:18:in `interval'", "/opt/logtest/logstash-6.4.2/logstash-core/lib/logstash/agent.rb:99:in `execute'", "/opt/logtest/logstash-6.4.2/logstash-core/lib/logstash/runner.rb:362:in `block in execute'", "/opt/logtest/logstash-6.4.2/vendor/bundle/jruby/2.3.0/gems/stud-0.0.23/lib/stud/task.rb:24:in `block in initialize'"]}

```



参考：https://discuss.elastic.co/t/logstash-x-pack-error-conencting-to-es/128398



```
POST _xpack/security/role/logstash_writer
{
  "cluster": ["manage_index_templates", "monitor"],
  "indices": [
    {
      "names": [ "logstash-*",".logstash" ], 
      "privileges": ["write","delete","create_index"]
    }
  ]
}
```



```
POST _xpack/security/user/logstash_internal
{
  "password" : "123456",
  "roles" : [ "logstash_writer"],
  "full_name" : "Internal Logstash User"
}
```

创建logstash_admin用户

http://10.221.123.45:9200/_xpack/security/user/logstash_admin

{
  "password" : "123456",
  "roles" : [ "logstash_reader", "logstash_admin","logstash_writer"], 
  "full_name" : "Kibana User for Logstash"
}	



```
PUT _xpack/security/user/logstash_system/_enable
```



2：kibana界面保存报错：

```
Advanced Setting Error

Unable to save advanced setting.

Check that Kibana and Elasticsearch are still running, then try again.
```



日志错误如下：

```
{"type":"response","@timestamp":"2018-10-24T07:14:09Z","tags":[],"pid":4452,"method":"post","statusCode":503,"req":{"url":"/api/kibana/settings","method":"post","headers":{"host":"192.168.3.191:5601","connection":"keep-alive","content-length":"37","accept":"application/json","origin":"http://192.168.3.191:5601","kbn-version":"6.4.2","user-agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36","content-type":"application/json","referer":"http://192.168.3.191:5601/app/kibana","accept-encoding":"gzip, deflate","accept-language":"zh-CN,zh;q=0.9,en;q=0.8"},"remoteAddress":"192.168.1.100","userAgent":"192.168.1.100","referer":"http://192.168.3.191:5601/app/kibana"},"res":{"statusCode":503,"responseTime":290,"contentLength":9},"message":"POST /api/kibana/settings 503 290ms - 9.0B"}
{"type":"response","@timestamp":"2018-10-24T07:14:09Z","tags":[],"pid":4452,"method":"get","statusCode":304,"req":{"url":"/bundles/4b5a84aaf1c9485e060c503a0ff8cadb.woff2","method":"get","headers":{"host":"192.168.3.191:5601","connection":"keep-alive","origin":"http://192.168.3.191:5601","if-none-match":"\"574ea2698c03ae9477db2ea3baf460ee32f1a7ea\"","if-modified-since":"Wed, 26 Sep 2018 13:57:11 GMT","user-agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36","accept":"*/*","referer":"http://192.168.3.191:5601/bundles/commons.style.css","accept-encoding":"gzip, deflate","accept-language":"zh-CN,zh;q=0.9,en;q=0.8"},"remoteAddress":"192.168.1.100","userAgent":"192.168.1.100","referer":"http://192.168.3.191:5601/bundles/commons.style.css"},"res":{"statusCode":304,"responseTime":2,"contentLength":9},"message":"GET /bundles/4b5a84aaf1c9485e060c503a0ff8cadb.woff2 304 2ms - 9.0B"}
```



3：

```
[2018-10-26T11:03:35,846][WARN ][logstash.filters.grok    ] Timeout executing grok '%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{DATA:remoteIp} %{DATA:RFC} %{DATA:clientFlag} \[%{HTTPDATE:time}\] .*? %{DATA:reqResource} %{DATA:status} %{DATA:reqBytes} %{DATA:RT}ms' against field 'message' with value 'Value too large to output (507 bytes)! First 255 chars are: airflow-01.embrace.com 2018-10-26T10:28:47.161Z /opt/logtest/logs/was_http_access.log 0 122.119.122.57 - - [04/Aug/2015:13:59:18 +0800] "POST /dz/servlet/GetCheckCode HTTP/1.1" 200 - "http://b2c.donghaiair.com/dz/FlightSearch.do" "Mozilla/5.0 (iPad; U; CPU'!

```



4: logstash测试

```
jcf_app_server 数据或格式有问题：放到一个字段中
wmq_mqm_fdc 数据格式不明：放到一个字段中

已处理好：
was_gc：格式不对
was_http_access 格式不对
wmq_qm  格式不对
wmq_mqm 格式不对
amq 数据有问题
```

匹配多行到msg字段中

```
(?<msg>(.|\\n)*)
```



24

20right;  1个AMQ*.FDC格式不明、1个jcf_app_server数据或格式有问题；was_gc 没有调通



5: 

```
2018-10-27T13:28:45.322+0800	ERROR	kafka/client.go:233	Kafka (topic=was_system_out): dropping too large message of size 666.
2018-10-27T13:28:45.322+0800	ERROR	kafka/client.go:233	Kafka (topic=was_system_out): dropping too large message of size 666.
2018-10-27T13:28:45.322+0800	ERROR	kafka/client.go:233	Kafka (topic=was_system_out): dropping too large message of size 666.
2018-10-27T13:28:45.322+0800	ERROR	kafka/client.go:233	Kafka (topic=was_system_out): dropping too large message of size 666.
2018-10-27T13:28:45.322+0800	ERROR	kafka/client.go:233	Kafka (topic=was_system_out): dropping too large message of size 666.
2018-10-27T13:28:45.322+0800	ERROR	kafka/client.go:233	Kafka (topic=was_system_out): dropping too large message of size 666.
```

修改kafka配置



6：邮件告警错误

```
[2018-10-28T13:32:31,078][ERROR][o.e.x.w.a.e.ExecutableEmailAction] [node1] failed to execute action [_inlined_/email_1]
javax.mail.MessagingException: failed to send email with subject [Watch [watch1] has exceeded the threshold] via account [sitechyizt]
	at org.elasticsearch.xpack.watcher.notification.email.EmailService.send(EmailService.java:153) ~[?:?]
	at org.elasticsearch.xpack.watcher.notification.email.EmailService.send(EmailService.java:145) ~[?:?]
	at org.elasticsearch.xpack.watcher.actions.email.ExecutableEmailAction.execute(ExecutableEmailAction.java:72) ~[?:?]
	at org.elasticsearch.xpack.core.watcher.actions.ActionWrapper.execute(ActionWrapper.java:144) [x-pack-core-6.4.2.jar:6.4.2]
	at org.elasticsearch.xpack.watcher.execution.ExecutionService.executeInner(ExecutionService.java:464) [x-pack-watcher-6.4.2.jar:6.4.2]
	at org.elasticsearch.xpack.watcher.execution.ExecutionService.execute(ExecutionService.java:295) [x-pack-watcher-6.4.2.jar:6.4.2]
	at org.elasticsearch.xpack.watcher.transport.actions.execute.TransportExecuteWatchAction$1.doRun(TransportExecuteWatchAction.java:154) [x-pack-watcher-6.4.2.jar:6.4.2]
	at org.elasticsearch.common.util.concurrent.AbstractRunnable.run(AbstractRunnable.java:37) [elasticsearch-6.4.2.jar:6.4.2]
	at java.util.concurrent.Executors$RunnableAdapter.call(Executors.java:511) [?:1.8.0_131]
	at java.util.concurrent.FutureTask.run(FutureTask.java:266) [?:1.8.0_131]
	at org.elasticsearch.common.util.concurrent.ThreadContext$ContextPreservingRunnable.run(ThreadContext.java:624) [elasticsearch-6.4.2.jar:6.4.2]
	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1142) [?:1.8.0_131]
	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:617) [?:1.8.0_131]
	at java.lang.Thread.run(Thread.java:748) [?:1.8.0_131]
Caused by: com.sun.mail.smtp.SMTPSendFailedException: 553 Mail from must equal authorized user

	at com.sun.mail.smtp.SMTPTransport.issueSendCommand(SMTPTransport.java:2267) ~[?:?]
	at com.sun.mail.smtp.SMTPTransport.mailFrom(SMTPTransport.java:1758) ~[?:?]
	at com.sun.mail.smtp.SMTPTransport.sendMessage(SMTPTransport.java:1257) ~[?:?]
	at org.elasticsearch.xpack.watcher.notification.email.Account.send(Account.java:141) ~[?:?]
	at org.elasticsearch.xpack.watcher.notification.email.EmailService.send(EmailService.java:151) ~[?:?]
	... 13 more
Caused by: com.sun.mail.smtp.SMTPSenderFailedException: 553 Mail from must equal authorized user

	at com.sun.mail.smtp.SMTPTransport.mailFrom(SMTPTransport.java:1767) ~[?:?]
	at com.sun.mail.smtp.SMTPTransport.sendMessage(SMTPTransport.java:1257) ~[?:?]
	at org.elasticsearch.xpack.watcher.notification.email.Account.send(Account.java:141) ~[?:?]
	at org.elasticsearch.xpack.watcher.notification.email.EmailService.send(EmailService.java:151) ~[?:?]
	... 13 more

```

https://github.com/zjlywjh001/phrackCTF-Personal-Docker/issues/1

增加

```
email_defaults:
        from: sitechyizt@163.com
```



7：



bulk_max_size: 15000->50000 没有效果 ；100000 -> 2000000

worker：20->30  没有效果

harvester_buffer_size：104857600 -> 504857600  有提升，后由降下来

channel_buffer_size ：20000 -> 10000  有下降 改到100000，有提升



Pipeline workers 24->40 好像有轻微提升

Pipeline batch size 800 ->8000 ;  大幅提升，接近翻倍；->20000 



## 现场查询脚本

```
{
    "query": {
        "range" : {
            "logtime" : {
                "gte" : "2018-11-01 2:16:00,000 Z",
                "lt" :  "2018-11-01 2:16:01,000 Z"
            }
        }
    }
}
```



