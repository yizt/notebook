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