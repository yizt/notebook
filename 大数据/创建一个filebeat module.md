go安装

```
https://dl.google.com/go/go1.11.2.linux-amd64.tar.gz
tar -xvf go1.11.2.linux-amd64.tar.gz
export GOROOT=/opt/securetest/go
export GOPATH=/opt/securetest/gopath
export PATH=$PATH:$GOROOT/bin
```



```
mkdir -p ${GOPATH}/src/github.com/elastic
git clone https://github.com/elastic/beats ${GOPATH}/src/github.com/elastic/beats
```



```
cd beats/filebeat

make create-module MODULE=sysaudit
make create-fileset MODULE=sysaudit FILESET=audit
make create-fields MODULE=sysaudit FILESET=audit
make update
```

```
New module was generated, now you can start creating filesets by create-fileset command.

New fileset was generated, please check that module.yml file have proper fileset dashboard settings. After setting up Grok pattern in pipeline.json, please generate fields.yml

```

参考；https://www.elastic.co/guide/en/beats/devguide/master/filebeat-modules-devguide.html



```
[OPERATE USER:root][LOGIN USER:root][LOGIN SOURCE IP:(192.168.1.100)][LOGIN PID:6595][LOGIN TIME:2018-11-21 10:28 .][OPERATE TIME AND COMMAND: 2018-11-21 11:04:55 source /etc/bashrc ]
```



```
\[OPERATE USER:%{DATA:sysaudit.audit.user}\]\[LOGIN USER:%{DATA:sysaudit.audit.login_user}\]\[LOGIN SOURCE IP:\(%{IPORHOST:sysaudit.audit.client_ip}\)\]\[LOGIN PID:%{NUMBER:sysaudit.audit.login_pid}\]\[LOGIN TIME:%{TIMESTAMP_ISO8601:sysaudit.audit.login_time}.*\]\[OPERATE TIME AND COMMAND: %{TIMESTAMP_ISO8601:sysaudit.audit.timestamp} %{GREEDYDATA:sysaudit.audit.cmd} \]
```

```
\\[OPERATE USER:%{DATA:sysaudit.audit.user}\\]\\[LOGIN USER:%{DATA:sysaudit.audit.login_user}\\]\\[LOGIN SOURCE IP:\\(%{IPORHOST:sysaudit.audit.client_ip}\\)\\]\\[LOGIN PID:%{NUMBER:sysaudit.audit.login_pid}\\]\\[LOGIN TIME:%{TIMESTAMP_ISO8601:sysaudit.audit.login_time}.*\\]\\[OPERATE TIME AND COMMAND: %{TIMESTAMP_ISO8601:sysaudit.audit.timestamp} %{GREEDYDATA:sysaudit.audit.cmd} \\]
```



修改模板 

PUT http://192.168.3.191:9200/_template/filebeat-6.4.2

```
        "@timestamp": {
          "type": "date",
          "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ss"
        },
```



修改kibana配置

```
Discover: blocked by: [FORBIDDEN/12/index read-only / allow delete (api)];: [cluster_block_exception] blocked by: [FORBIDDEN/12/index read-only / allow delete (api)];
```



```
PUT .kibana/_settings
{
"index": {
"blocks": {
"read_only_allow_delete": "false"
}
}
}
```





## hadoop日志解析



```
PUT _ingest/pipeline/filebeat-6.4.2-hadoop-pipeline
{
  "description": "Pipeline for parsing sysaudit audit logs",
  "processors": [
      {
          "rename": {
              "field": "@timestamp",
              "target_field": "event.created"
          }
      },
      {
          "grok": {
              "field": "message",
              "pattern_definitions" : {
                  "GREEDYMULTILINE" : "(.|\\n)*",
                  "INDEXNAME": "[a-zA-Z0-9_.-]*"
              },
              "patterns": [
                  "%{TIMESTAMP_ISO8601:op_time} (?<msg>(.|\n)*)"
              ]
          }
      },
      {
          "date": {
              "field": "op_time",
			  "target_field": "@timestamp",
			  "formats": ["ISO8601","yyyy-MM-dd HH:mm:ss,SSS"],
			  "timezone": "Asia/Shanghai"
              }
      }
    ],
  "on_failure" : [{
    "set" : {
      "field" : "error.message",
      "value" : "{{ _ingest.on_failure_message }}"
    }
  }]
}

```

