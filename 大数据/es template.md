[TOC]

### apache

http://10.221.123.45:9200/_template/temp_log_apache

```
{
  "index_patterns": ["log-apache*"],
  "settings": {
    "number_of_shards": 128,
    "number_of_replicas":0,
     "index.refresh_interval" : "3",
    "index.translog.durability": "async",
    "index.translog.sync_interval": "10s",
    "index.translog.flush_threshold_size": "256mb"
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
        "reqHBytes": {
          "type": "long"
        },
        "respHBytes": {
          "type": "long"
        },
        "RT": {
          "type": "long"
        },
        "reqBytes": {
          "type": "long"
        },
 	   "logtime": {
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss,SSS Z||dd/MMM/yyyy/HH:mm:ss Z||dd/MMM/yyyy:HH:mm:ss Z" 
        },
        "fb_time": {
          "type": "keyword"
        }
      }
    }
  }
}
```



### jboss

http://10.221.123.45:9200/_template/temp_log_jboss

```
{
  "index_patterns": ["log-jboss*"],
  "settings": {
    "number_of_shards": 64,
    "number_of_replicas":0,
    "index.refresh_interval" : "10s",
    "index.translog.durability": "async",
    "index.translog.sync_interval": "10s",
    "index.translog.flush_threshold_size": "600MB"
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": true
      },
      "properties": {
     	 "fb_time": {
          "type": "keyword"
        },
  		"logtime": {
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss,SSS Z||dd/MMM/yyyy/HH:mm:ss Z||dd/MMM/yyyy:HH:mm:ss Z" 
        }
      }
    }
  }
}
```



### was

http://10.221.123.45:9200/_template/temp_log_was

```
{
  "index_patterns": ["log-was*"],
  "settings": {
    "number_of_shards": 64,
    "number_of_replicas":0,
     "index.refresh_interval" : "10s",
    "index.translog.durability": "async",
    "index.translog.sync_interval": "10s",
    "index.translog.flush_threshold_size": "600MB"
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": true
      },
      "properties": {
     	 "fb_time": {
          "type": "keyword"
        },
  		"logtime": {
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss,SSS Z||dd/MMM/yyyy/HH:mm:ss Z||dd/MMM/yyyy:HH:mm:ss Z" 
        }
      }
    }
  }
}
```



### was_gc

http://10.221.123.45:9200/_template/temp_log_was_gc

```
{
  "index_patterns": ["log-was_gc*"],
  "settings": {
    "number_of_shards": 64,
    "number_of_replicas":0,
    "index.refresh_interval" : "10s",
    "index.translog.durability": "async",
    "index.translog.sync_interval": "10s",
    "index.translog.flush_threshold_size": "600MB"
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": true
      },
      "properties": {
     	 "fb_time": {
          "type": "keyword"
        }
      }
    }
  }
}
```



### jcf

http://10.221.123.45:9200/_template/temp_log_jcf

```
{
  "index_patterns": ["log-jcf*"],
  "settings": {
    "number_of_shards": 64,
    "number_of_replicas":0,
     "index.refresh_interval" : "10s",
    "index.translog.durability": "async",
    "index.translog.sync_interval": "10s",
    "index.translog.flush_threshold_size": "600MB"
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": true
      },
      "properties": {
     	 "fb_time": {
          "type": "keyword"
        },
        "lineNo": {
          "type": "integer"
        },
  		"logtime": {
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss,SSS Z||dd/MMM/yyyy/HH:mm:ss Z||dd/MMM/yyyy:HH:mm:ss Z" 
        }
      }
    }
  }
}
```





### tode

http://10.221.123.45:9200/_template/temp_log_tode

```
{
  "index_patterns": ["log-tode*"],
  "settings": {
    "number_of_shards": 64,
    "number_of_replicas":0,
        "index.refresh_interval" : "10s",
    "index.translog.durability": "async",
    "index.translog.sync_interval": "10s",
    "index.translog.flush_threshold_size": "600MB"
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": true
      },
      "properties": {
     	 "fb_time": {
          "type": "keyword"
        },
        "lineNo": {
          "type": "integer"
        },
  		"logtime": {
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss,SSS Z||dd/MMM/yyyy/HH:mm:ss Z||dd/MMM/yyyy:HH:mm:ss Z" 
        }
      }
    }
  }
}
```



### tuxedo

http://10.221.123.45:9200/_template/temp_log_tuxedo

```
{
  "index_patterns": ["log-tuxedo*"],
  "settings": {
    "number_of_shards": 64,
    "number_of_replicas":0,
    "index.refresh_interval" : "10s",
    "index.translog.durability": "async",
    "index.translog.sync_interval": "10s",
    "index.translog.flush_threshold_size": "600MB"
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": true
      },
      "properties": {
     	 "fb_time": {
          "type": "keyword"
        },
        "message_number": {
          "type": "integer"
        },
  		"logtime": {
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss,SSS Z||dd/MMM/yyyy/HH:mm:ss Z||dd/MMM/yyyy:HH:mm:ss Z" 
        }
      }
    }
  }
}
```



### wmq、AMQ

http://10.221.123.45:9200/_template/temp_log_wqm

```
{
  "index_patterns": ["log-wqm*"],
  "settings": {
    "number_of_shards": 64,
    "number_of_replicas":0,
     "index.refresh_interval" : "10s",
    "index.translog.durability": "async",
    "index.translog.sync_interval": "10s",
    "index.translog.flush_threshold_size": "600MB"
  },
  "mappings": {
    "doc": {
      "_source": {
        "enabled": true
      },
      "properties": {
     	 "fb_time": {
          "type": "keyword"
        },
  		"logtime": {
          "type": "date",
          "format": "yyyy-MM-dd HH:mm:ss,SSS Z||yyyy-MM-dd HH:mm:ss Z||dd/MMM/yyyy/HH:mm:ss Z||dd/MMM/yyyy:HH:mm:ss Z" 
        }
      }
    }
  }
}
```







