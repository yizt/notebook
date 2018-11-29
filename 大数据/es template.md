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









```
{
    "filebeat-6.4.2": {
        "order": 1,
        "index_patterns": [
            "filebeat-6.4.2-*"
        ],
        "settings": {
            "index": {
                "mapping": {
                    "total_fields": {
                        "limit": "10000"
                    }
                },
                "refresh_interval": "5s",
                "number_of_routing_shards": "30",
                "number_of_shards": "3"
            }
        },
        "mappings": {
            "doc": {
                "_meta": {
                    "version": "6.4.2"
                },
                "date_detection": false,
                "dynamic_templates": [
                    {
                        "fields": {
                            "mapping": {
                                "type": "keyword"
                            },
                            "match_mapping_type": "string",
                            "path_match": "fields.*"
                        }
                    },
                    {
                        "docker.container.labels": {
                            "mapping": {
                                "type": "keyword"
                            },
                            "match_mapping_type": "string",
                            "path_match": "docker.container.labels.*"
                        }
                    },
                    {
                        "kibana.log.meta": {
                            "path_match": "kibana.log.meta.*",
                            "mapping": {
                                "type": "keyword"
                            },
                            "match_mapping_type": "string"
                        }
                    },
                    {
                        "strings_as_keyword": {
                            "match_mapping_type": "string",
                            "mapping": {
                                "ignore_above": 1024,
                                "type": "keyword"
                            }
                        }
                    }
                ],
                "properties": {
                    "beat": {
                        "properties": {
                            "name": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "hostname": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "timezone": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "version": {
                                "type": "keyword",
                                "ignore_above": 1024
                            }
                        }
                    },
                    "icinga": {
                        "properties": {
                            "debug": {
                                "properties": {
                                    "facility": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "severity": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            },
                            "main": {
                                "properties": {
                                    "facility": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "severity": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            },
                            "startup": {
                                "properties": {
                                    "facility": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "severity": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            }
                        }
                    },
                    "osquery": {
                        "properties": {
                            "result": {
                                "properties": {
                                    "name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "action": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "host_identifier": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "unix_time": {
                                        "type": "long"
                                    },
                                    "calendar_time": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            }
                        }
                    },
                    "system": {
                        "properties": {
                            "auth": {
                                "properties": {
                                    "program": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "pid": {
                                        "type": "long"
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "sudo": {
                                        "properties": {
                                            "error": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "tty": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "pwd": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "user": {
                                                "ignore_above": 1024,
                                                "type": "keyword"
                                            },
                                            "command": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            }
                                        }
                                    },
                                    "useradd": {
                                        "properties": {
                                            "name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "uid": {
                                                "type": "long"
                                            },
                                            "gid": {
                                                "type": "long"
                                            },
                                            "home": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "shell": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            }
                                        }
                                    },
                                    "timestamp": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "user": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "ssh": {
                                        "properties": {
                                            "event": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "method": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "ip": {
                                                "type": "ip"
                                            },
                                            "dropped_ip": {
                                                "type": "ip"
                                            },
                                            "port": {
                                                "type": "long"
                                            },
                                            "signature": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "geoip": {
                                                "properties": {
                                                    "continent_name": {
                                                        "type": "keyword",
                                                        "ignore_above": 1024
                                                    },
                                                    "city_name": {
                                                        "type": "keyword",
                                                        "ignore_above": 1024
                                                    },
                                                    "region_name": {
                                                        "type": "keyword",
                                                        "ignore_above": 1024
                                                    },
                                                    "country_iso_code": {
                                                        "type": "keyword",
                                                        "ignore_above": 1024
                                                    },
                                                    "location": {
                                                        "type": "geo_point"
                                                    },
                                                    "region_iso_code": {
                                                        "type": "keyword",
                                                        "ignore_above": 1024
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    "groupadd": {
                                        "properties": {
                                            "name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "gid": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "hostname": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            },
                            "syslog": {
                                "properties": {
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "timestamp": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "hostname": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "program": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "pid": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            }
                        }
                    },
                    "event": {
                        "properties": {
                            "severity": {
                                "type": "long"
                            },
                            "created": {
                                "type": "date"
                            }
                        }
                    },
                    "kibana": {
                        "properties": {
                            "log": {
                                "properties": {
                                    "tags": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "state": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "meta": {
                                        "type": "object"
                                    }
                                }
                            }
                        }
                    },
                    "postgresql": {
                        "properties": {
                            "log": {
                                "properties": {
                                    "thread_id": {
                                        "type": "long"
                                    },
                                    "user": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "database": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "duration": {
                                        "type": "float"
                                    },
                                    "query": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "timestamp": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "timezone": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "level": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            }
                        }
                    },
                    "stream": {
                        "ignore_above": 1024,
                        "type": "keyword"
                    },
                    "process": {
                        "properties": {
                            "program": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "pid": {
                                "type": "long"
                            }
                        }
                    },
                    "http": {
                        "properties": {
                            "response": {
                                "properties": {
                                    "status_code": {
                                        "type": "long"
                                    },
                                    "elapsed_time": {
                                        "type": "long"
                                    },
                                    "content_length": {
                                        "type": "long"
                                    }
                                }
                            },
                            "request": {
                                "properties": {
                                    "method": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            }
                        }
                    },
                    "tags": {
                        "type": "keyword",
                        "ignore_above": 1024
                    },
                    "error": {
                        "properties": {
                            "message": {
                                "type": "text",
                                "norms": false
                            },
                            "code": {
                                "type": "long"
                            },
                            "type": {
                                "type": "keyword",
                                "ignore_above": 1024
                            }
                        }
                    },
                    "host": {
                        "properties": {
                            "os": {
                                "properties": {
                                    "version": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "family": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "platform": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            },
                            "ip": {
                                "type": "ip"
                            },
                            "mac": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "name": {
                                "ignore_above": 1024,
                                "type": "keyword"
                            },
                            "id": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "architecture": {
                                "type": "keyword",
                                "ignore_above": 1024
                            }
                        }
                    },
                    "mongodb": {
                        "properties": {
                            "log": {
                                "properties": {
                                    "severity": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "component": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "context": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            }
                        }
                    },
                    "read_timestamp": {
                        "type": "keyword",
                        "ignore_above": 1024
                    },
                    "log": {
                        "properties": {
                            "level": {
                                "type": "keyword",
                                "ignore_above": 1024
                            }
                        }
                    },
                    "mysql": {
                        "properties": {
                            "error": {
                                "properties": {
                                    "timestamp": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "thread_id": {
                                        "type": "long"
                                    },
                                    "level": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            },
                            "slowlog": {
                                "properties": {
                                    "query_time": {
                                        "properties": {
                                            "sec": {
                                                "type": "float"
                                            }
                                        }
                                    },
                                    "rows_sent": {
                                        "type": "long"
                                    },
                                    "query": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "id": {
                                        "type": "long"
                                    },
                                    "user": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "host": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "ip": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "lock_time": {
                                        "properties": {
                                            "sec": {
                                                "type": "float"
                                            }
                                        }
                                    },
                                    "rows_examined": {
                                        "type": "long"
                                    },
                                    "timestamp": {
                                        "type": "long"
                                    }
                                }
                            }
                        }
                    },
                    "redis": {
                        "properties": {
                            "log": {
                                "properties": {
                                    "role": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "level": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "pid": {
                                        "type": "long"
                                    }
                                }
                            },
                            "slowlog": {
                                "properties": {
                                    "key": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "args": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "cmd": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "duration": {
                                        "properties": {
                                            "us": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "id": {
                                        "type": "long"
                                    }
                                }
                            }
                        }
                    },
                    "source": {
                        "type": "keyword",
                        "ignore_above": 1024
                    },
                    "fileset": {
                        "properties": {
                            "module": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "name": {
                                "type": "keyword",
                                "ignore_above": 1024
                            }
                        }
                    },
                    "meta": {
                        "properties": {
                            "cloud": {
                                "properties": {
                                    "provider": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "instance_id": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "instance_name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "machine_type": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "availability_zone": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "project_id": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "region": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    }
                                }
                            }
                        }
                    },
                    "iis": {
                        "properties": {
                            "access": {
                                "properties": {
                                    "remote_ip": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "request_time_ms": {
                                        "type": "long"
                                    },
                                    "http_version": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "body_sent": {
                                        "properties": {
                                            "bytes": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "url": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "query_string": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "agent": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "user_agent": {
                                        "properties": {
                                            "os_minor": {
                                                "type": "long"
                                            },
                                            "device": {
                                                "ignore_above": 1024,
                                                "type": "keyword"
                                            },
                                            "os": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os_major": {
                                                "type": "long"
                                            },
                                            "name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "major": {
                                                "type": "long"
                                            },
                                            "minor": {
                                                "type": "long"
                                            },
                                            "patch": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            }
                                        }
                                    },
                                    "server_ip": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "hostname": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "win32_status": {
                                        "type": "long"
                                    },
                                    "cookie": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "body_received": {
                                        "properties": {
                                            "bytes": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "port": {
                                        "type": "long"
                                    },
                                    "sub_status": {
                                        "type": "long"
                                    },
                                    "referrer": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "response_code": {
                                        "type": "long"
                                    },
                                    "site_name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "server_name": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "geoip": {
                                        "properties": {
                                            "location": {
                                                "type": "geo_point"
                                            },
                                            "region_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "city_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "region_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "continent_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "country_iso_code": {
                                                "ignore_above": 1024,
                                                "type": "keyword"
                                            }
                                        }
                                    },
                                    "method": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "user_name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            },
                            "error": {
                                "properties": {
                                    "server_ip": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "method": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "url": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "queue_name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "remote_ip": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "server_port": {
                                        "type": "long"
                                    },
                                    "http_version": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "response_code": {
                                        "type": "long"
                                    },
                                    "reason_phrase": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "geoip": {
                                        "properties": {
                                            "region_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "continent_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "country_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "location": {
                                                "type": "geo_point"
                                            },
                                            "region_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "city_name": {
                                                "ignore_above": 1024,
                                                "type": "keyword"
                                            }
                                        }
                                    },
                                    "remote_port": {
                                        "type": "long"
                                    }
                                }
                            }
                        }
                    },
                    "docker": {
                        "properties": {
                            "container": {
                                "properties": {
                                    "id": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "image": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "labels": {
                                        "type": "object"
                                    }
                                }
                            }
                        }
                    },
                    "apache2": {
                        "properties": {
                            "access": {
                                "properties": {
                                    "user_name": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "http_version": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "response_code": {
                                        "type": "long"
                                    },
                                    "referrer": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "agent": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "remote_ip": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "method": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "url": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "body_sent": {
                                        "properties": {
                                            "bytes": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "user_agent": {
                                        "properties": {
                                            "device": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "major": {
                                                "type": "long"
                                            },
                                            "minor": {
                                                "type": "long"
                                            },
                                            "name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os_major": {
                                                "type": "long"
                                            },
                                            "os_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "patch": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os_minor": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "geoip": {
                                        "properties": {
                                            "city_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "region_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "continent_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "country_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "location": {
                                                "type": "geo_point"
                                            },
                                            "region_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            }
                                        }
                                    }
                                }
                            },
                            "error": {
                                "properties": {
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "pid": {
                                        "type": "long"
                                    },
                                    "tid": {
                                        "type": "long"
                                    },
                                    "module": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "level": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "client": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            }
                        }
                    },
                    "kafka": {
                        "properties": {
                            "log": {
                                "properties": {
                                    "message": {
                                        "norms": false,
                                        "type": "text"
                                    },
                                    "component": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "class": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "trace": {
                                        "properties": {
                                            "class": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "message": {
                                                "norms": false,
                                                "type": "text"
                                            },
                                            "full": {
                                                "type": "text",
                                                "norms": false
                                            }
                                        }
                                    },
                                    "timestamp": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "level": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            }
                        }
                    },
                    "nginx": {
                        "properties": {
                            "access": {
                                "properties": {
                                    "url": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "body_sent": {
                                        "properties": {
                                            "bytes": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "referrer": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "response_code": {
                                        "type": "long"
                                    },
                                    "agent": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "user_agent": {
                                        "properties": {
                                            "os": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os_major": {
                                                "type": "long"
                                            },
                                            "os_minor": {
                                                "type": "long"
                                            },
                                            "device": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "patch": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "major": {
                                                "type": "long"
                                            },
                                            "minor": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "geoip": {
                                        "properties": {
                                            "continent_name": {
                                                "ignore_above": 1024,
                                                "type": "keyword"
                                            },
                                            "country_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "location": {
                                                "type": "geo_point"
                                            },
                                            "region_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "city_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "region_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            }
                                        }
                                    },
                                    "remote_ip": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "user_name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "method": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "http_version": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    }
                                }
                            },
                            "error": {
                                "properties": {
                                    "tid": {
                                        "type": "long"
                                    },
                                    "connection_id": {
                                        "type": "long"
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "level": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "pid": {
                                        "type": "long"
                                    }
                                }
                            }
                        }
                    },
                    "offset": {
                        "type": "long"
                    },
                    "prospector": {
                        "properties": {
                            "type": {
                                "type": "keyword",
                                "ignore_above": 1024
                            }
                        }
                    },
                    "service": {
                        "properties": {
                            "name": {
                                "type": "keyword",
                                "ignore_above": 1024
                            }
                        }
                    },
                    "@timestamp": {
                        "type": "date",
                        "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd HH:mm:ss"
                    },
                    "kubernetes": {
                        "properties": {
                            "namespace": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "node": {
                                "properties": {
                                    "name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            },
                            "labels": {
                                "type": "object"
                            },
                            "annotations": {
                                "type": "object"
                            },
                            "container": {
                                "properties": {
                                    "name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "image": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            },
                            "pod": {
                                "properties": {
                                    "name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "uid": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            }
                        }
                    },
                    "auditd": {
                        "properties": {
                            "log": {
                                "properties": {
                                    "acct": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "items": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "item": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "record_type": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "old_auid": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "new_auid": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "sequence": {
                                        "type": "long"
                                    },
                                    "res": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "geoip": {
                                        "properties": {
                                            "region_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "continent_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "city_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "region_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "country_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "location": {
                                                "type": "geo_point"
                                            }
                                        }
                                    },
                                    "old_ses": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "ppid": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "new_ses": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "pid": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "a0": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    }
                                }
                            }
                        }
                    },
                    "elasticsearch": {
                        "properties": {
                            "index": {
                                "properties": {
                                    "name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "id": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    }
                                }
                            },
                            "shard": {
                                "properties": {
                                    "id": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            },
                            "audit": {
                                "properties": {
                                    "layer": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "principal": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "action": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "request": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "event_type": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "origin_type": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "origin_address": {
                                        "type": "ip"
                                    },
                                    "uri": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "request_body": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            },
                            "deprecation": {
                                "properties": {}
                            },
                            "gc": {
                                "properties": {
                                    "jvm_runtime_sec": {
                                        "type": "float"
                                    },
                                    "threads_total_stop_time_sec": {
                                        "type": "float"
                                    },
                                    "stopping_threads_time_sec": {
                                        "type": "float"
                                    },
                                    "tags": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "heap": {
                                        "properties": {
                                            "size_kb": {
                                                "type": "long"
                                            },
                                            "used_kb": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "old_gen": {
                                        "properties": {
                                            "size_kb": {
                                                "type": "long"
                                            },
                                            "used_kb": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "young_gen": {
                                        "properties": {
                                            "size_kb": {
                                                "type": "long"
                                            },
                                            "used_kb": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "phase": {
                                        "properties": {
                                            "duration_sec": {
                                                "type": "float"
                                            },
                                            "scrub_symbol_table_time_sec": {
                                                "type": "float"
                                            },
                                            "scrub_string_table_time_sec": {
                                                "type": "float"
                                            },
                                            "weak_refs_processing_time_sec": {
                                                "type": "float"
                                            },
                                            "parallel_rescan_time_sec": {
                                                "type": "float"
                                            },
                                            "class_unload_time_sec": {
                                                "type": "float"
                                            },
                                            "cpu_time": {
                                                "properties": {
                                                    "sys_sec": {
                                                        "type": "float"
                                                    },
                                                    "real_sec": {
                                                        "type": "float"
                                                    },
                                                    "user_sec": {
                                                        "type": "float"
                                                    }
                                                }
                                            },
                                            "name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            }
                                        }
                                    }
                                }
                            },
                            "server": {
                                "properties": {
                                    "component": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "gc": {
                                        "properties": {
                                            "young": {
                                                "properties": {
                                                    "one": {
                                                        "type": "long"
                                                    },
                                                    "two": {
                                                        "type": "long"
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    "gc_overhead": {
                                        "type": "long"
                                    }
                                }
                            },
                            "slowlog": {
                                "properties": {
                                    "took": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "types": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "source_query": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "extra_source": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "took_millis": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "total_shards": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "routing": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "logger": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "type": {
                                        "ignore_above": 1024,
                                        "type": "keyword"
                                    },
                                    "search_type": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "total_hits": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "id": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "stats": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            },
                            "node": {
                                "properties": {
                                    "name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            }
                        }
                    },
                    "message": {
                        "type": "text",
                        "norms": false
                    },
                    "input": {
                        "properties": {
                            "type": {
                                "type": "keyword",
                                "ignore_above": 1024
                            }
                        }
                    },
                    "syslog": {
                        "properties": {
                            "facility": {
                                "type": "long"
                            },
                            "priority": {
                                "type": "long"
                            },
                            "severity_label": {
                                "type": "keyword",
                                "ignore_above": 1024
                            },
                            "facility_label": {
                                "ignore_above": 1024,
                                "type": "keyword"
                            }
                        }
                    },
                    "fields": {
                        "type": "object"
                    },
                    "logstash": {
                        "properties": {
                            "log": {
                                "properties": {
                                    "module": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "thread": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "log_event": {
                                        "type": "object"
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "level": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    }
                                }
                            },
                            "slowlog": {
                                "properties": {
                                    "plugin_params_object": {
                                        "type": "object"
                                    },
                                    "module": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "plugin_name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "took_in_millis": {
                                        "type": "long"
                                    },
                                    "took_in_nanos": {
                                        "type": "long"
                                    },
                                    "plugin_type": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "plugin_params": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "message": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "level": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "thread": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "event": {
                                        "type": "text",
                                        "norms": false
                                    }
                                }
                            }
                        }
                    },
                    "traefik": {
                        "properties": {
                            "access": {
                                "properties": {
                                    "user_name": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "url": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "body_sent": {
                                        "properties": {
                                            "bytes": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "referrer": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "agent": {
                                        "norms": false,
                                        "type": "text"
                                    },
                                    "geoip": {
                                        "properties": {
                                            "location": {
                                                "type": "geo_point"
                                            },
                                            "region_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "city_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "region_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "continent_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "country_iso_code": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            }
                                        }
                                    },
                                    "frontend_name": {
                                        "type": "text",
                                        "norms": false
                                    },
                                    "backend_url": {
                                        "norms": false,
                                        "type": "text"
                                    },
                                    "remote_ip": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "method": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "http_version": {
                                        "type": "keyword",
                                        "ignore_above": 1024
                                    },
                                    "response_code": {
                                        "type": "long"
                                    },
                                    "user_agent": {
                                        "properties": {
                                            "device": {
                                                "ignore_above": 1024,
                                                "type": "keyword"
                                            },
                                            "minor": {
                                                "type": "long"
                                            },
                                            "patch": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os_minor": {
                                                "type": "long"
                                            },
                                            "os_name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "major": {
                                                "type": "long"
                                            },
                                            "name": {
                                                "type": "keyword",
                                                "ignore_above": 1024
                                            },
                                            "os": {
                                                "ignore_above": 1024,
                                                "type": "keyword"
                                            },
                                            "os_major": {
                                                "type": "long"
                                            }
                                        }
                                    },
                                    "request_count": {
                                        "type": "long"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "aliases": {}
    }
}
```

