[TOC]



## Query查询

###  查询和过滤

```
{
  "query": { 
    "bool": { 
      "must": [
        { "match": { "title":   "Search"        }}, 
        { "match": { "content": "Elasticsearch" }}  
      ],
      "filter": [ 
        { "term":  { "status": "published" }}, 
        { "range": { "publish_date": { "gte": "2015-01-01" }}} 
      ]
    }
  }
}
```



### 全文检索

#### match

```
{
    "query": {
        "match" : {
            "message" : "this is a test"
        }
    }
}

{
    "query": {
        "match" : {
            "message" : {
                "query" : "this is a test",
                "operator" : "and"
            }
        }
    }
}

```



#### match_phrase

短语查询

```
{
    "query": {
        "match_phrase" : {
            "message" : "this is a test"
        }
    }
}
```



### 字段级查询

#### term query

```
{
  "query": {
    "term" : { "user" : "Kimchy" } 
  }
}
```



#### terms query

```
{
    "query": {
        "terms" : { "user" : ["kimchy", "elasticsearch"]}
    }
}
```



#### range query

```
{
    "query": {
        "range" : {
            "age" : {
                "gte" : 10,
                "lte" : 20,
                "boost" : 2.0
            }
        }
    }
}
```

  日期类型

```
{
    "query": {
        "range" : {
            "date" : {
                "gte" : "now-1d/d",
                "lt" :  "now/d"
            }
        }
    }
}
```



#### regexp query

```
{
    "query": {
        "regexp":{
            "name.first": "s.*y"
        }
    }
}
```



### 组合查询

#### bool query

- must: 文档必须完全匹配条件
- should: should下面会带一个以上的条件，至少满足一个条件，这个文档就符合should
- must_not: 文档必须不匹配条件

```json
{
  "query": {
    "bool" : {
      "must" : {
        "term" : { "user" : "kimchy" }
      },
      "filter": {
        "term" : { "tag" : "tech" }
      },
      "must_not" : {
        "range" : {
          "age" : { "gte" : 10, "lte" : 20 }
        }
      },
      "should" : [
        { "term" : { "tag" : "wow" } },
        { "term" : { "tag" : "elasticsearch" } }
      ],
      "minimum_should_match" : 1,
      "boost" : 1.0
    }
  }
}
```



#### Dis Max Query

If the query is "albino elephant" this ensures that "albino" matching one field and "elephant" matching another gets a higher score than "albino" matching both fields. 

```
{
    "query": {
        "dis_max" : {
            "tie_breaker" : 0.7,
            "boost" : 1.2,
            "queries" : [
                {
                    "term" : { "age" : 34 }
                },
                {
                    "term" : { "age" : 35 }
                }
            ]
        }
    }
}
```



#### Function Scores Query

```
{
    "query": {
        "function_score": {
          "query": { "match_all": {} },
          "boost": "5", 
          "functions": [
              {
                  "filter": { "match": { "test": "bar" } },
                  "random_score": {}, 
                  "weight": 23
              },
              {
                  "filter": { "match": { "test": "cat" } },
                  "weight": 42
              }
          ],
          "max_boost": 42,
          "score_mode": "max",
          "boost_mode": "multiply",
          "min_score" : 42
        }
    }
}
```



#### Boosting Query

The `boosting` query can be used to effectively demote results that match a given query. Unlike the "NOT" clause in bool query, this still selects documents that contain undesirable terms, but reduces their overall score



```
{
    "query": {
        "boosting" : {
            "positive" : {
                "term" : {
                    "field1" : "value1"
                }
            },
            "negative" : {
                 "term" : {
                     "field2" : "value2"
                }
            },
            "negative_boost" : 0.2
        }
    }
}s
```



### 联合查询

#### nested query

```

```



#### Has Child Query

```
{
    "query": {
        "has_child" : {
            "type" : "blog_tag",
            "query" : {
                "term" : {
                    "tag" : "something"
                }
            }
        }
    }
}
```



### 其它

​     

```json
"size": 10,
  "from": 0,
  "sort": {
    "_score": {
      "order": "desc"
    }
  },
  "highlight": {
    "fields": {
      "ygroup_name": {}
    },
    "pre_tags": [
      "<font color=red>"
    ],
    "post_tags": [
      "</font>"
    ]
  }
```



### Span Query

参考：https://www.cnblogs.com/xing901022/p/4982698.html



## 聚合

### Metrics Aggregation

#### avg

```
{
    "aggs" : {
        "avg_grade" : { "avg" : { "field" : "grade" } }
    }
}
```



#### weighted avg

```
{
    "size": 0,
    "aggs" : {
        "weighted_grade": {
            "weighted_avg": {
                "value": {
                    "field": "grade"
                },
                "weight": {
                    "field": "weight"
                }
            }
        }
    }
}
```



#### cardinality

A `single-value` metrics aggregation that calculates an approximate count of distinct values. Values can be extracted either from specific fields in the document or generated by a script.

```
{
    "aggs" : {
        "type_count" : {
            "cardinality" : {
                "field" : "type"
            }
        }
    }
}
```



#### sum

A `single-value` metrics aggregation that sums up numeric values that are extracted from the aggregated documents. These values can be extracted either from specific numeric fields in the documents, or be generated by a provided script.

```
{
    "query" : {
        "constant_score" : {
            "filter" : {
                "match" : { "type" : "hat" }
            }
        }
    },
    "aggs" : {
        "hat_prices" : { "sum" : { "field" : "price" } }
    }
}
```



#### value count

```
{
    "aggs" : {
        "types_count" : { "value_count" : { "field" : "type" } }
    }
}
```



#### stats

A `multi-value` metrics aggregation that computes stats over numeric values extracted from the aggregated documents. These values can be extracted either from specific numeric fields in the documents, or be generated by a provided script.

The stats that are returned consist of: `min`, `max`, `sum`, `count` and `avg`.

```
{
    "aggs" : {
        "grades_stats" : { "stats" : { "field" : "grade" } }
    }
}
```



#### percentiles

A `multi-value` metrics aggregation that calculates one or more percentiles over numeric values extracted from the aggregated documents. These values can be extracted either from specific numeric fields in the documents, or be generated by a provided script.

```
{
    "size": 0,
    "aggs" : {
        "load_time_outlier" : {
            "percentiles" : {
                "field" : "load_time" 
            }
        }
    }
}
```



#### Percentile rank

A `multi-value` metrics aggregation that calculates one or more percentile ranks over numeric values extracted from the aggregated documents. These values can be extracted either from specific numeric fields in the documents, or be generated by a provided script.

```
{
    "size": 0,
    "aggs" : {
        "load_time_ranks" : {
            "percentile_ranks" : {
                "field" : "load_time", 
                "values" : [500, 600]
            }
        }
    }
}
```

结果

```
{
    ...

   "aggregations": {
      "load_time_ranks": {
         "values" : {
            "500.0": 55.00000000000001,
            "600.0": 64.0
         }
      }
   }
}
```



### bucket aggregation

#### adjacency matrix

A bucket aggregation returning a form of [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix). The request provides a collection of named filter expressions, similar to the `filters` aggregation request. Each bucket in the response represents a non-empty cell in the matrix of intersecting filters.

```
{
  "size": 0,
  "aggs" : {
    "interactions" : {
      "adjacency_matrix" : {
        "filters" : {
          "grpA" : { "terms" : { "accounts" : ["hillary", "sidney"] }},
          "grpB" : { "terms" : { "accounts" : ["donald", "mitt"] }},
          "grpC" : { "terms" : { "accounts" : ["vladimir", "nigel"] }}
        }
      }
    }
  }
}
```



结果

```
{
  "took": 9,
  "timed_out": false,
  "_shards": ...,
  "hits": ...,
  "aggregations": {
    "interactions": {
      "buckets": [
        {
          "key":"grpA",
          "doc_count": 2
        },
        {
          "key":"grpA&grpB",
          "doc_count": 1
        },
        {
          "key":"grpB",
          "doc_count": 2
        },
        {
          "key":"grpB&grpC",
          "doc_count": 1
        },
        {
          "key":"grpC",
          "doc_count": 1
        }
      ]
    }
  }
}
```



#### composite aggregation

A multi-bucket aggregation that creates composite buckets from different sources.

Unlike the other `multi-bucket` aggregation the `composite` aggregation can be used to paginate **all**buckets from a multi-level aggregation efficiently. This aggregation provides a way to stream **all**buckets of a specific aggregation similarly to what [scroll](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-scroll.html) does for documents.

For instance the following document:

```
{
    "keyword": ["foo", "bar"],
    "number": [23, 65, 76]
}
```

... creates the following composite buckets when `keyword` and `number` are used as values source for the aggregation:

```
{ "keyword": "foo", "number": 23 }
{ "keyword": "foo", "number": 65 }
{ "keyword": "foo", "number": 76 }
{ "keyword": "bar", "number": 23 }
{ "keyword": "bar", "number": 65 }
{ "keyword": "bar", "number": 76 }
```

a) terms

```
{
    "aggs" : {
        "my_buckets": {
            "composite" : {
                "sources" : [
                    { "product": { "terms" : { "field": "product" } } }
                ]
            }
        }
     }
}
```

b) histogram

```
{
    "aggs" : {
        "my_buckets": {
            "composite" : {
                "sources" : [
                    { "histo": { "histogram" : { "field": "price", "interval": 5 } } }
                ]
            }
        }
    }
}
```



c) data histogram

```
{
    "aggs" : {
        "my_buckets": {
            "composite" : {
                "sources" : [
                    { "date": { "date_histogram" : { "field": "timestamp", "interval": "1d","format": "yyyy-MM-dd" } } }
                ]
            }
        }
    }
}
```



d) mixed

```
{
    "aggs" : {
        "my_buckets": {
            "composite" : {
                "sources" : [
                    { "date": { "date_histogram": { "field": "timestamp", "interval": "1d" } } },
                    { "product": { "terms": {"field": "product" } } }
                ]
            }
        }
    }
}
```



e) sub

```
{
    "aggs" : {
        "my_buckets": {
            "composite" : {
                 "sources" : [
                    { "date": { "date_histogram": { "field": "timestamp", "interval": "1d", "order": "desc" } } },
                    { "product": { "terms": {"field": "product" } } }
                ]
            },
            "aggregations": {
                "the_avg": {
                    "avg": { "field": "price" }
                }
            }
        }
    }
}
```



#### histogram

```
{
    "aggs" : {
        "prices" : {
            "histogram" : {
                "field" : "price",
                "interval" : 50
            }
        }
    }
}
```



```
{
    "aggs" : {
        "prices" : {
            "histogram" : {
                "field" : "price",
                "interval" : 50,
                "min_doc_count" : 1
            }
        }
    }
}
```



#### date histogram

```
{
    "aggs" : {
        "sales_over_time" : {
            "date_histogram" : {
                "field" : "date",
                "interval" : "month"
            }
        }
    }
}
```



#### range

```
{
    "aggs" : {
        "price_ranges" : {
            "range" : {
                "field" : "price",
                "ranges" : [
                    { "to" : 100.0 },
                    { "from" : 100.0, "to" : 200.0 },
                    { "from" : 200.0 }
                ]
            }
        }
    }
}
```



```
{
    "aggs" : {
        "price_ranges" : {
            "range" : {
                "field" : "price",
                "keyed" : true,
                "ranges" : [
                    { "key" : "cheap", "to" : 100 },
                    { "key" : "average", "from" : 100, "to" : 200 },
                    { "key" : "expensive", "from" : 200 }
                ]
            }
        }
    }
}
```



#### date range

```
{
    "aggs": {
        "range": {
            "date_range": {
                "field": "date",
                "format": "MM-yyy",
                "ranges": [
                    { "to": "now-10M/M" }, 
                    { "from": "now-10M/M" } 
                ]
            }
        }
    }
}
```



```
{
   "aggs": {
       "range": {
           "date_range": {
               "field": "date",
               "time_zone": "CET",
               "ranges": [
                  { "to": "2016/02/01" }, 
                  { "from": "2016/02/01", "to" : "now/d" },
                  { "from": "now/d" }
              ]
          }
      }
   }
}
```



#### diversified sampler

Like the `sampler` aggregation this is a filtering aggregation used to limit any sub aggregations’ processing to a sample of the top-scoring documents. The `diversified_sampler` aggregation adds the ability to limit the number of matches that share a common value such as an "author".



#### filter

Defines a single bucket of all the documents in the current document set context that match a specified filter. Often this will be used to narrow down the current aggregation context to a specific set of documents.

```
{
    "aggs" : {
        "t_shirts" : {
            "filter" : { "term": { "type": "t-shirt" } },
            "aggs" : {
                "avg_price" : { "avg" : { "field" : "price" } }
            }
        }
    }
}
```



#### filters

Defines a multi bucket aggregation where each bucket is associated with a filter. Each bucket will collect all documents that match its associated filter.

```
{
  "size": 0,
  "aggs" : {
    "messages" : {
      "filters" : {
        "filters" : {
          "errors" :   { "match" : { "body" : "error"   }},
          "warnings" : { "match" : { "body" : "warning" }}
        }
      }
    }
  }
}
```



#### terms

A multi-bucket value source based aggregation where buckets are dynamically built - one per unique value.

```
{
    "aggs" : {
        "genres" : {
            "terms" : { "field" : "genre" }
        }
    }
}
```





## machine learning

The first consideration is that it must be time series data. The machine learning features are designed to model and detect anomalies in time series data.



it is information that contains key performance indicators (KPIs) for the health, security, or success of your business or system. 





You can also use multi-metric jobs to split a single time series into multiple time series based on a categorical field. For example, you can split the data based on its hostnames, locations, or users. Each time series is modeled independently. By looking at temporal patterns on a per entity basis, you might spot things that might have otherwise been hidden in the lumped view.



## filebeat

es插件

```
bin/elasticsearch-plugin install ingest-user-agent
```



```
./filebeat modules list
./filebeat modules enable apache2 elasticsearch postgresql system nginx
./filebeat setup --dashboards

```

```
apache2
auditd
elasticsearch
icinga
iis
kafka
kibana
logstash
mongodb
mysql
nginx
osquery
postgresql
redis
system
traefik

```



系统审计

```
mkdir /usermonitor
chmod 777 -R /usermonitor
```





## metricbeat

```
bin/elasticsearch -E node.name=node191 -E network.host=192.168.3.191 -E node.master=true -E node.data=false -d
bin/elasticsearch -E node.name=node192 -E network.host=192.168.3.192 -E node.master=false -E node.data=true -d
bin/elasticsearch -E node.name=node193 -E network.host=192.168.3.193 -E node.master=false -E node.data=true -d
```





https://www.elastic.co/guide/en/beats/metricbeat/current/defining-processors.html





https://artifacts.elastic.co/downloads/beats/metricbeat/metricbeat-6.4.2-linux-x86_64.tar.gz

https://artifacts.elastic.co/downloads/beats/packetbeat/packetbeat-6.4.2-linux-x86_64.tar.gz









```
./metricbeat modules enable kibana postgresql elasticsearch
./metricbeat setup --dashboards

./metricbeat -e -c metricbeat.yml
```



```
Enabled:
elasticsearch
kibana
postgresql
system

Disabled:
aerospike
apache
ceph
couchbase
docker
dropwizard
envoyproxy
etcd
golang
graphite
haproxy
http
jolokia
kafka
kubernetes
kvm
logstash
memcached
mongodb
munin
mysql
nginx
php_fpm
prometheus
rabbitmq
redis
traefik
uwsgi
vsphere
windows
zookeeper

```





```
{
  "_index": "metricbeat-6.4.2-2018.11.20",
  "_type": "doc",
  "_id": "RJrcMGcBiKUpeSHNXV8m",
  "_version": 1,
  "_score": null,
  "_source": {
    "@timestamp": "2018-11-20T11:23:26.494Z",
    "metricset": {
      "name": "activity",
      "module": "postgresql",
      "host": "192.168.3.51:5432",
      "rtt": 50729
    },
    "postgresql": {
      "activity": {
        "state_change": "2018-11-20T11:23:28.211Z",
        "pid": 1654,
        "application_name": "",
        "query_start": "2018-11-20T11:23:28.211Z",
        "backend_start": "2018-11-20T11:23:28.195Z",
        "transaction_start": "2018-11-20T11:23:28.211Z",
        "state": "active",
        "client": {
          "hostname": "",
          "port": 37619,
          "address": "192.168.3.191"
        },
        "query": "SELECT * FROM pg_stat_bgwriter",
        "database": {
          "oid": 13212,
          "name": "postgres"
        },
        "user": {
          "name": "postgres",
          "id": 10
        }
      }
    },
    "beat": {
      "name": "airflow-01.embrace.com",
      "hostname": "airflow-01.embrace.com",
      "version": "6.4.2"
    },
    "host": {
      "name": "airflow-01.embrace.com"
    }
  },
  "fields": {
    "postgresql.activity.transaction_start": [
      "2018-11-20T11:23:28.211Z"
    ],
    "@timestamp": [
      "2018-11-20T11:23:26.494Z"
    ],
    "postgresql.activity.query_start": [
      "2018-11-20T11:23:28.211Z"
    ],
    "postgresql.activity.backend_start": [
      "2018-11-20T11:23:28.195Z"
    ],
    "postgresql.activity.state_change": [
      "2018-11-20T11:23:28.211Z"
    ]
  },
  "highlight": {
    "postgresql.activity.query": [
      "@kibana-highlighted-field@SELECT * FROM pg_stat_bgwriter@/kibana-highlighted-field@"
    ],
    "metricset.host": [
      "@kibana-highlighted-field@192.168.3.51:5432@/kibana-highlighted-field@"
    ]
  },
  "sort": [
    1542713006494
  ]
}
```





## packetbeat



必须是root用户;区分字段type

```shell
./packetbeat setup --dashboards
./packetbeat -e -c packetbeat.yml
```



es插件安装

```
bin/elasticsearch-plugin install ingest-geoip
```





通用包之外的

- [ICMP](https://www.elastic.co/guide/en/beats/packetbeat/current/packetbeat-icmp-options.html)

- [DNS](https://www.elastic.co/guide/en/beats/packetbeat/current/packetbeat-dns-options.html)

- [HTTP](https://www.elastic.co/guide/en/beats/packetbeat/current/packetbeat-http-options.html)

- [AMQP](https://www.elastic.co/guide/en/beats/packetbeat/current/packetbeat-amqp-options.html)

- [Cassandra](https://www.elastic.co/guide/en/beats/packetbeat/current/configuration-cassandra.html)

- [Memcache](https://www.elastic.co/guide/en/beats/packetbeat/current/packetbeat-memcache-options.html)

- [MySQL and PgSQL](https://www.elastic.co/guide/en/beats/packetbeat/current/packetbeat-mysql-pgsql-options.html)

- [Thrift](https://www.elastic.co/guide/en/beats/packetbeat/current/configuration-thrift.html)

- [MongoDB](https://www.elastic.co/guide/en/beats/packetbeat/current/configuration-mongodb.html)

- [TLS](https://www.elastic.co/guide/en/beats/packetbeat/current/configuration-tls.html)

  ​



kibana console

```
PUT _ingest/pipeline/geoip-info
{
  "description": "Add geoip info",
  "processors": [
    {
      "geoip": {
        "field": "client_ip",
        "target_field": "client_geoip",
        "properties": ["location"],
        "ignore_failure": true
      }
    }
  ]
}
```







```json
{
  "_index": "packetbeat-6.4.2-2018.11.20",
  "_type": "doc",
  "_id": "4TmCMGcBtwTbeWkIldH9",
  "_version": 1,
  "_score": null,
  "_source": {
    "@timestamp": "2018-11-20T09:45:22.610Z",
    "icmp": {
      "request": {
        "message": "143(0)",
        "type": 143,
        "code": 0
      },
      "version": 6
    },
    "client_ip": "fe80::e9:58da:6f5e:8fda",
    "beat": {
      "name": "airflow-01.embrace.com",
      "hostname": "airflow-01.embrace.com",
      "version": "6.4.2"
    },
    "type": "icmp",
    "status": "OK",
    "ip": "ff02::16",
    "host": {
      "name": "airflow-01.embrace.com"
    },
    "path": "ff02::16",
    "bytes_in": 20
  },
  "fields": {
    "@timestamp": [
      "2018-11-20T09:45:22.610Z"
    ]
  },
  "highlight": {
    "type": [
      "@kibana-highlighted-field@icmp@/kibana-highlighted-field@"
    ]
  },
  "sort": [
    1542707122610
  ]
}
```



```json
{
  "_index": "packetbeat-6.4.2-2018.11.20",
  "_type": "doc",
  "_id": "ADmCMGcBtwTbeWkIQ9HG",
  "_version": 1,
  "_score": null,
  "_source": {
    "@timestamp": "2018-11-20T09:45:01.235Z",
    "proc": "",
    "direction": "in",
    "type": "http",
    "http": {
      "request": {
        "params": "packages=x1.txt%2C+t2.txt",
        "headers": {
          "content-length": 0
        }
      },
      "response": {
        "headers": {
          "content-length": 139,
          "content-type": "application/json"
        },
        "code": 204,
        "phrase": "No Content"
      }
    },
    "beat": {
      "name": "airflow-01.embrace.com",
      "hostname": "airflow-01.embrace.com",
      "version": "6.4.2"
    },
    "client_server": "",
    "ip": "192.168.3.191",
    "client_port": 58954,
    "bytes_in": 825,
    "responsetime": 4,
    "client_ip": "192.168.2.248",
    "path": "/andible/packages/remove",
    "client_proc": "",
    "server": "",
    "host": {
      "name": "airflow-01.embrace.com"
    },
    "status": "OK",
    "query": "DELETE /andible/packages/remove",
    "port": 8000,
    "method": "DELETE",
    "bytes_out": 222
  },
  "fields": {
    "@timestamp": [
      "2018-11-20T09:45:01.235Z"
    ]
  },
  "highlight": {
    "type": [
      "@kibana-highlighted-field@http@/kibana-highlighted-field@"
    ]
  },
  "sort": [
    1542707101235
  ]
}
```



```
{
  "_index": "packetbeat-6.4.2-2018.11.20",
  "_type": "doc",
  "_id": "oTmMMGcBtwTbeWkIduuD",
  "_version": 1,
  "_score": null,
  "_source": {
    "@timestamp": "2018-11-20T09:56:10.000Z",
    "start_time": "2018-11-20T09:55:22.392Z",
    "last_time": "2018-11-20T09:55:22.392Z",
    "type": "flow",
    "host": {
      "name": "airflow-01.embrace.com"
    },
    "transport": "tcp",
    "beat": {
      "version": "6.4.2",
      "name": "airflow-01.embrace.com",
      "hostname": "airflow-01.embrace.com"
    },
    "source": {
      "ip": "192.168.1.100",
      "port": 62337,
      "stats": {
        "net_packets_total": 1,
        "net_bytes_total": 108
      }
    },
    "dest": {
      "ip": "192.168.3.191",
      "port": 22,
      "stats": {
        "net_packets_total": 1,
        "net_bytes_total": 56
      }
    },
    "flow_id": "EAT/////AP//////CP8AAAHAqAFkwKgDv4HzFgA",
    "final": false
  },
  "fields": {
    "start_time": [
      "2018-11-20T09:55:22.392Z"
    ],
    "@timestamp": [
      "2018-11-20T09:56:10.000Z"
    ],
    "last_time": [
      "2018-11-20T09:55:22.392Z"
    ]
  },
  "highlight": {
    "source.ip": [
      "@kibana-highlighted-field@192.168.1.100@/kibana-highlighted-field@"
    ],
    "type": [
      "@kibana-highlighted-field@flow@/kibana-highlighted-field@"
    ]
  },
  "sort": [
    1542707770000
  ]
}
```

