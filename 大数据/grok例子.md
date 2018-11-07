[TOC]





```json
# ##  Log Jboss

input{	
    kafka {
		type => "accesslog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"groupa2"
		topics =>"jboss_access"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
    kafka {
		type => "serverlog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"groupa2"
		topics =>"jboss_server"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}
filter {
	# 根据不同 type 制定不同规则
	if [type] == "serverlog" {
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{TIMESTAMP_ISO8601:time} %{DATA:logLevel} \[%{DATA:className}\]\(%{DATA:thread}\)%{GREEDYDATA:msg}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"server"
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyy-MM-dd HH:mm:ss,SSS","yyyyMMddHHmmssSSS"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	if [type] == "accesslog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{DATA:remoteIp} %{DATA:RFC} %{DATA:clientFlag} \[%{HTTPDATE:time}\] .*? %{DATA:reqResource} %{DATA:status} %{DATA:reqBytes} %{DATA:RT}ms"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"access"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","dd/MMM/yyyy/HH:mm:ss","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "time"
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"jboss"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}


output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "accesslog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-jboss-access-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "serverlog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-jboss-server-%{+YYYY.MM.dd}"
			}
			
		}
	}
}



















###  Log Redis

input{	
	kafka {
		type => "redislog"
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_redis"
		topics =>"redis"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}

filter {
	# 根据不同 type 制定不同规则
	if [type] == "redislog" {
	
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \[%{DATA:id}\] %{MONTHDAY:monthday} %{MONTH:month} %{TIME:time1} \# %{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"redis"
			}
		}
		
	}
	grok {
		# 支持以正则方式格式化信息
		match => [
			"@timestamp","^(?<yyyy>\d{1,4})"
		]
	}
	mutate {
		add_field => { 
			"topic"=>"AMQ"
			"time"=>"%{yyyy}/%{month}/%{monthday} %{time1}"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
		remove_field => ["time1"]
	}
	# 日期格式化插件
	date {
		# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
		match => ["time","yyyy/MMM/dd HH:mm:ss,SSS","yyyyMMddHHmmssSSS"]
		# 格式化后输出到哪个字段,一般用于图形显示的索引
		target => "index_time"
	}
}

output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "redislog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-redis-%{+YYYY.MM.dd}"
			}
		}
	}
}


















###Log Tuxedo
input{	
    kafka {
		type => "ulog"
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_tuxedu"
		topics =>"tuxedo"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}


filter {
	# 根据不同 type 制定不同规则
	if [type] == "ulog" {
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{DATA:time}\.%{DATA:hostName}\!%{DATA:serverName}\.%{DATA:pid}\.%{DATA:thread_ID}\.%{DATA:context_ID}\:%{DATA:catalog_name}\:%{DATA:message_number}\:%{DATA:logLevel}\:%{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"ulog"
			}
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"tuxedo"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}

output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "ulog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-tuxedo-%{+YYYY.MM.dd}"
			}
			
		}
	}
}




















#####Log Tode
input{	
    kafka {
		type => "tlog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_tode"
		topics =>"tode"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}


filter {
	# 根据不同 type 制定不同规则
	if [type] == "tlog" {
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \[%{DATA:time}\]%{DATA:logLevel}\:%{DATA:hostName}\:%{DATA:serverName}\:%{DATA:pid}\:\[%{DATA:sourceFileName}\:%{DATA:lineNo}\]\:%{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"tlog"
			}
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"tode"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}
output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "tlog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-tode-%{+YYYY.MM.dd}"
			}
			
		}
	}
}


















#####Log Sih


input{	
	kafka {
		type => "SIH_info_001_log"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_sih"
		topics =>"sih_info"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
	kafka {
		type => "SIH_err_001_log"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_sih1"
		topics =>"sih_err"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
	kafka {
		type => "SIH_running_001_log"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_sih"
		topics =>"sih_running"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}
filter {
	# 根据不同 type 制定不同规则
	if [type] == "SIH_info_001_log" {
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \[\#\%\&\*\^\]%{DATA:time}\:%{DATA:code}\.%{DATA:hostName}\!SIH\:%{DATA:Program}\.%{DATA:PID}\,%{DATA:code1}\:%{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"SIH_info_001"
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyyMMddhhmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	if [type] == "SIH_err_001_log" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \[\#\%\&\*\^\]%{DATA:time}\:%{DATA:code}\.%{DATA:hostName}\!SIH\:%{DATA:Program}\.%{DATA:PID}\,%{DATA:code1}\:%{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"SIH_err_001"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyyMMddhhmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	if [type] == "SIH_running_001_log" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \[\#\%\&\*\^\]%{DATA:time}\:%{DATA:code}\.%{DATA:hostName}\!SIH\:%{DATA:Program}\.%{DATA:PID}\,%{DATA:code1}\:RUNNING\> AppName\=%{DATA:AppName}\;OtherSide=%{DATA:OtherSide}\;SihProcessTime=%{DATA:SihProcessTime}\;CoreProcessTime=%{DATA:CoreProcessTime}\;AppProcessTime=%{DATA:AppProcessTime}\;AppResponseTime=%{DATA:AppResponseTime}\;SihProcessMaxTime=%{DATA:SihProcessMaxTime}\;CoreProcessMaxTime=%{DATA:CoreProcessMaxTime}\;AppProcessMaxTime=%{DATA:AppProcessMaxTime}\;AppResponseMaxTime=%{DATA:AppResponseMaxTime}\;TpsThres=%{DATA:TpsThres}\;SvrReq=%{DATA:SvrReq}\;SvrReqFromOtherNode=%{DATA:SvrReqFromOtherNode}\;SvrReqTps=%{DATA:SvrReqTps}\;SvrRes=%{DATA:SvrRes}\;ClientReq=%{DATA:ClientReq}\;ClientRes=%{DATA:ClientRes}\;ClientResFromOtherNode=%{DATA:ClientResFromOtherNode}\;ClientResTps=%{DATA:ClientResTps}\;SvrReqAvMsgLen=%{DATA:SvrReqAvMsgLen}\;ClientReqAvMsgLen=%{DATA:ClientReqAvMsgLen}\;SvrResAvMsgLen=%{DATA:SvrResAvMsgLen}\;ClientResAvMsgLen=%{DATA:ClientResAvMsgLen}\;Discard=%{DATA:Discard}\;MsgType=%{GREEDYDATA:MsgType}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"SIH_running_001"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyyMMddhhmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"SIH"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}
output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "SIH_info_001_log" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-sih_info-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "SIH_err_001_log" {
			
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-sih_err-%{+YYYY.MM.dd}"
			}
		}
		if [type] == "SIH_running_001_log" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-sih_running-%{+YYYY.MM.dd}"
			}
		}
	}
}







































##########Log Wmq

input{	
	kafka {
		type => "active_AMQERR01log"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_wmq3"
		topics =>"wmq_qm"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
	kafka {
		type => "errors_AMQERR01log"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_wmq3"
		topics =>"wmq_mqm"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
	kafka {
		type => "errors_AMQERR01fdc"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_wmq4"
		topics =>"wmq_mqm_fdc"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}
filter {
	# 根据不同 type 制定不同规则
	if [type] == "active_AMQERR01log" {
	
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{DATESTAMP:date} %{DATA:aa} \- Process\(%{DATA:process}\) User\(%{DATA:user}\) Program\(%{DATA:program}\) \nHost\(%{DATA:host}\) Installation\(%{DATA:installation}\) \nVRMF\(%{DATA:VRMF}\)\s+%{DATA:ErrCode}\:%{DATA:ErrMsg}\s+EXPLANATION\:\s+(?<EXPLANATION>(.|\n)*)ACTION\:\s+(?<ACTION>(.|\n)*)"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"active_AMQERR01"
				"time"=>"%{date} %{aa}"
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","MM/dd/yyyy hh:mm:ss aa","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	if [type] == "errors_AMQERR01log" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{DATESTAMP:date} %{DATA:aa} \- Process\(%{DATA:process}\) User\(%{DATA:user}\) Program\(%{DATA:program}\) \nHost\(%{DATA:host}\) Installation\(%{DATA:installation}\) \nVRMF\(%{DATA:VRMF}\)\s+%{DATA:ErrCode}\:%{DATA:ErrMsg}\s+EXPLANATION\:\s+(?<EXPLANATION>(.|\n)*)ACTION\:\s+(?<ACTION>(.|\n)*)"
			]
		}
		
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"active_AMQERR01"
				"time"=>"%{date} %{aa}"
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","MM/dd/yyyy hh:mm:ss aa","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	
	if [type] == "errors_AMQERR01fdc" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{GREEDYDATA:msg}"
			]
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"WMQ"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}
output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "active_AMQERR01log" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-wqm_qm-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "errors_AMQERR01log" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-wqm_mqm-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "errors_AMQERR01fdc" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-wqm_mqm_fdc-%{+YYYY.MM.dd}"
			}
			
		}
	}
}




















######## Log Jcf
input{	
    kafka {
		type => "admin_jcflog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_jcf1"
		topics =>"jcf"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
    kafka {
		type => "operationlog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_jcf1"
		topics =>"jcf_operation"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
	kafka {
		type => "adapter_jcflog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_jcf1"
		topics =>"jcf_adapt"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
    kafka {
		type => "reg_jcflog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_jcf1"
		topics =>"jcf_reg"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
	kafka {
		type => "regbk_jcflog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_jcf2"
		topics =>"jcf_regbk"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
    kafka {
		type => "App_jcflog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_jcf4"
		topics =>"jcf_app_server"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}


filter {
	# 根据不同 type 制定不同规则
	if [type] == "admin_jcflog" {
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{TIMESTAMP_ISO8601:time} %{DATA:logLevel}\:%{DATA:message} className\:\[%{DATA:className}\] methodName\:\[%{DATA:methodName}\] lineNo\:\[%{NUMBER:lineNo}\]"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"admin_jcf"
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyy-MM-dd HH:mm:ss,SSS","yyyyMMddHHmmssSSS"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	if [type] == "operationlog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{DATA:logLevel} \[%{DATA:otype}\:%{TIMESTAMP_ISO8601:time} %{GREEDYDATA:message}\]"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"operation"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyy-MM-dd HH:mm:ss,SSS","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "time"
		}
	}
	if [type] == "adapter_jcflog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{TIMESTAMP_ISO8601:time} \| %{DATA:logLevel} \| %{DATA:threadName} \| %{DATA:className} \| .*? \| %{DATA:lineNo} \- %{DATA:packageName} \- %{DATA:version} \| %{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"adapter_jcf"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyy-MM-dd HH:mm:ss,SSS","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "time"
		}
	}
	if [type] == "reg_jcflog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{TIMESTAMP_ISO8601:time} \| %{DATA:logLevel} \| %{DATA:threadName} \| %{DATA:className} \| .*? \| %{DATA:lineNo} \- %{DATA:packageName} \- %{DATA:version} \| %{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"reg_jcf"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyy-MM-dd HH:mm:ss,SSS","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "time"
		}
	}
	if [type] == "regbk_jcflog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{TIMESTAMP_ISO8601:time} \| %{DATA:logLevel} \| %{DATA:threadName} \| %{DATA:className} \| .*? \| %{DATA:lineNo} \- %{DATA:packageName} \- %{DATA:version} \| %{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"regbk_jcf"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyy-MM-dd HH:mm:ss,SSS","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "time"
		}
	}
	if [type] == "App_jcflog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{GREEDYDATA:msg}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"regbk_jcf"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyy-MM-dd HH:mm:ss,SSS","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "time"
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"jcf"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}

output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	
	if "_grokparsefailure" not in [tags] {
		if [type] == "admin_jcflog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-jcf-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "operationlog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-jcf_operation-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "adapter_jcflog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-jcf_adapt-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "reg_jcflog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-jcf_reg-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "regbk_jcflog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-jcf_regbk-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "App_jcflog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-jcf_app_server-%{+YYYY.MM.dd}"
			}
		}
	}
}














#########Log Was
input{	
    kafka {
		type => "systemOutlog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"groupa5"
		topics =>"was_system_out"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
    kafka {
		type => "systemErrlog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"groupa2"
		topics =>"was_system_err"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
	kafka {
		type => "http_accesslog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_was"
		topics =>"was_http_access"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
    kafka {
		type => "gc_serverlog"	
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_was5"
		topics =>"was_gc"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}

filter {
	# 根据不同 type 制定不同规则
	if [type] == "systemOutlog" {
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \[%{DATESTAMP:date} %{DATA:z}\] %{DATA:threadId} %{DATA:shortName} %{DATA:eventType} %{DATA:className} %{DATA:methodName} (?<msg>(.|\n)*)"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"systemOut"
				"datetime"=>"%{date} %{z}"
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["datetime","yy-MM-dd HH:mm:ss:SSS z","yyyyMMddHHmmssSSS"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	if [type] == "systemErrlog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \[%{DATESTAMP:date} %{DATA:z}\] %{DATA:threadId} %{DATA:shortName} %{DATA:eventType} %{DATA:className} %{DATA:methodName} %{GREEDYDATA:message}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"systemErr"
				"datetime"=>"%{date} %{z}"
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["datetime","yy-MM-dd HH:mm:ss:SSS z","yyyyMMddHHmmssSSS"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	if [type] == "gc_serverlog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \<gc type=\"%{DATA:type}\" id=\"%{DATA:id}\" totalid=\"%{DATA:totalid}\" intervalms=\"%{DATA:gc_intervalms}\"\>\s+\<classunloading classloaders=\"%{DATA:classloaders}\" classes=\"%{DATA:classes}\" timevmquiescems=\"%{DATA:timevmquiescems}\" timetakenms=\"%{DATA:timetakenms}\" \/\>\s+\<finalization objectsqueued=\"%{DATA:objectsqueued}\" \/\>\s+\<timesms mark=\"%{DATA:mark}\" sweep=\"%{DATA:sweep}\" compact=\"%{DATA:compact}\" total=\"%{DATA:total}\" \/\>\s+\<tenured freebytes=\"%{DATA:tenured_freebytes}\" totalbytes=\"%{DATA:tenured_totalbytes}\" percent=\"%{DATA:tenured_percent}\" \>\s+\<soa freebytes=\"%{DATA:soa_freebytes}\" totalbytes=\"%{DATA:soa_totalbytes}\" percent=\"%{DATA:soa_percent}\" \/\>\s+\<loa freebytes=\"%{DATA:loa_freebytes}\" totalbytes=\"%{DATA:loa_totalbytes}\" percent=\"%{DATA:loa_percent}\" \/\>\s+\<\/tenured\>\s+\<\/gc\>.*?"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"gc_server"
			}
		}
	}
	if [type] == "http_accesslog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{DATA:remoteIp} %{DATA:RFC} %{DATA:clientFlag} \[%{HTTPDATE:time}\] .*? %{DATA:reqResource} %{DATA:protocol}\" %{DATA:status} %{DATA:reqBytes} %{GREEDYDATA:RT}"
			]
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"WAS"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}

output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "systemOutlog" {
		elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-was_system_out-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "systemErrlog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-was_system_err-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "gc_serverlog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-was_gc-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "http_accesslog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-was_http_access-%{+YYYY.MM.dd}"
			}
			
		}
	}

}




























###########Log Apache

input{	
    kafka {
		type => "accesslog"	
        bootstrap_servers=> "192.168.3.193:9092"
		topics =>"apche_accesslog"
		group_id =>"groupa5"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
    kafka {
		type => "errorlog"	
        bootstrap_servers=> "192.168.3.193:9092"
		topics =>"apache_error"
		group_id =>"groupa3"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}
filter {
	# 根据不同 type 制定不同规则
	if [type] == "accesslog" {
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{DATA:remoteIp} %{DATA:RFC} %{DATA:clientFlag} \[%{HTTPDATE:time}\] \"%{DATA:reqLine}\" %{DATA:status} %{DATA:reqBytes} %{DATA:serverName}\:%{DATA:serverPort} \"%{DATA:referer}\" \"%{DATA:cookie}\" \"%{DATA:user-Agent}\" %{DATA:reqHBytes} %{DATA:respHBytes} \- %{DATA:clientIp}\:%{DATA:clientPort} %{NUMBER:RT}"
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
	}
	if [type] == "errorlog" {
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} \[.*? %{MONTH:month} %{MONTHNUM:monthNum} %{TIME:time1} %{YEAR:year}\] \[%{DATA:errLevel}\] %{GREEDYDATA:errMsg}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"error"		
			}
		}
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","dd/MMM/yyyy/HH:mm:ss","yyyyMMddHHmmss"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"apache"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}


output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "accesslog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-apache-%{+YYYY.MM.dd}"
			}
			
		}
		if [type] == "errorlog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-apache-%{+YYYY.MM.dd}"
			}
			
		}
	}
}






























##############Log Amq

input{	
	kafka {
		type => "activemqlog"
        bootstrap_servers=> "192.168.3.193:9092"
        group_id=>"group_amq2"
		topics =>"amq"
		consumer_threads =>1
		auto_offset_reset => earliest
    }
}
filter {
	# 根据不同 type 制定不同规则
	if [type] == "activemqlog" {
	
		# grok 格式化信息插件，默认输入每条信息为  'message'
		grok {
			match => ["message","%{IPORHOST:fb_ip} %{DATA:fb_time} %{DATA:fb_filename} %{DATA:fb_offset} %{TIMESTAMP_ISO8601:time} \| %{DATA:logLevel} \| %{DATA:message} \| %{DATA:className} \| %{GREEDYDATA:threadName}"
			]
		}
		mutate {
			# add_field 新增字段，可以是多个. %{field_name} 动态取值
			add_field => {	
				"fileName"=>"activemq"
			}
		}
		# 日期格式化插件
		date {
			# 输入日期字段及格式  time:20180420090435 => yyyyMMddHHmmss
			match => ["time","yyyy-MM-dd HH:mm:ss,SSS","yyyyMMddHHmmssSSS"]
			# 格式化后输出到哪个字段,一般用于图形显示的索引
			target => "index_time"
		}
	}
	
	mutate {
		add_field => { 
			"topic"=>"AMQ"
		}
		# 删除不需要的字段
		remove_field => ["message"]
		remove_field => ["path"]
	}
}
output {
	# logstash对未解析成功的数据会存放在 tags 模块里, 这里进行过滤
	if "_grokparsefailure" not in [tags] {
		if [type] == "activemqlog" {
			elasticsearch {
            user => "logstash_internal"
            password => "123456"
            hosts => ["192.168.3.191:9200"]
            index => "log-amq-%{+YYYY.MM.dd}"
			}
			
		}
	}
}
```



现场调试结果：

21:

2018-10-30 16:53:23,366 DEBUG [dataClear.OtherServerLogData](Thread-5)server JBoss

31:

```
<%{TIMESTAMP_ISO8601:time}><%{DATA:pid}><%{DATA:component}><%{DATA:msg_type}><%{DATA:class_name}><%{DATA:function}><%{GREEDYDATA:msg}>
```

32:



-rw-r--r-- 1 root root   826831940 Oct 30 19:01 /opt/app/logproduce/log/41/jcf0.log
-rw-r--r-- 1 root root    26789412 Oct 30 19:01 /opt/app/logproduce/log/45/jcf0.log
-rw-r--r-- 1 root root    43518925 Oct 30 19:01 /opt/app/logproduce/log/51/jcf0.log
-rw-r--r-- 1 root root    27913499 Oct 30 19:01 /opt/app/logproduce/log/61/jcf0.log
-rw-r--r-- 1 root root   160774695 Oct 30 19:01 /opt/app/logproduce/log/71/jcf0.log
-rw-r--r-- 1 root root    71339113 Oct 30 19:01 /opt/app/logproduce/log/81/jcf0.log



#### 32-wag_gc

```
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<finalization objectsqueued=1537 /> 
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<finalization objectsqueued=1537 /> 
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<finalization objectsqueued=1537 /> 
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<finalization objectsqueued=1537 /> 
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<finalization objectsqueued=1537 /> 
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
<finalization objectsqueued=1537 /> 
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<finalization objectsqueued=1537 /> 
<timesms mark=80.257 sweep=11.954 compact=0.000 total=95.159 />
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
<finalization objectsqueued=1537 /> 
```

```
<gc type=global id=1 totalid=1 intervalms=0.000> <classunloading classloaders=0 classes=0 timevmquiescems=0.000 timetakenms=2.734 />
```





```
\<gc type=%{DATA:type} id=%{DATA:id} totalid=%{DATA:totalid} intervalms=%{DATA:gc_intervalms}\>\s+\<classunloading classloaders=%{DATA:classloaders} classes=%{DATA:classes} timevmquiescems=%{DATA:timevmquiescems} timetakenms=%{DATA:timetakenms} \/\>\s+(\<finalization objectsqueued=%{DATA:objectsqueued} \/\>\s+)?(\<timesms mark=%{DATA:mark} sweep=%{DATA:sweep} compact=%{DATA:compact} total=%{DATA:total} \/\>)?
```





#### 41-jcf

```
2018-10-30 11:02:19,045 INFO:session no user 1540897339045 className:SessionUtils] methodName:[getUserName88] lineNo:[941]
2018-10-30 11:02:19,045 FATAL:session no user 1540897339045 className:SessionUtils] methodName:[getUserName66] lineNo:[11]
2018-10-30 11:02:19,045 ERROR:session no user 1540897339045 className:SessionUtils] methodName:[getUserName5] lineNo:[553]
2018-10-30 11:02:19,046 WARN:session no user 1540897339046 className:SessionUtils] methodName:[getUserName61] lineNo:[628]
2018-10-30 11:02:19,046 FATAL:session no user 1540897339046 className:SessionUtils] methodName:[getUserName94] lineNo:[746]
2018-10-30 11:02:19,046 INFO:session no user 1540897339046 className:SessionUtils] methodName:[getUserName89] lineNo:[538]
```

```
%{TIMESTAMP_ISO8601:time} %{DATA:logLevel}\:%{DATA:msg} className\:%{DATA:className}\] methodName\:\[%{DATA:methodName}\] lineNo\:\[%{NUMBER:lineNo}\]
```



#### 45-jcf_regbk

```
2018-10-30 11:02:19,030 | 2 | Thread-10 | RegBkServer | 锛燂紵 |386 - test - 3.0 | 0.0627303663277109
2018-10-30 11:02:19,032 | 4 | Thread-10 | RegBkServer | 锛燂紵 |908 - test - 2.4 | 0.13982705779490712
2018-10-30 11:02:19,033 | 0 | Thread-10 | RegBkServer | 锛燂紵 |104 - test - 1.4 | 0.4089680813421863
2018-10-30 11:02:19,034 | 2 | Thread-10 | RegBkServer | 锛燂紵 |401 - test - 3.3 | 0.8206166834583875
2018-10-30 11:02:19,035 | 0 | Thread-10 | RegBkServer | 锛燂紵 |374 - test - 0.3 | 0.7016273680268268
2018-10-30 11:02:19,036 | 1 | Thread-10 | RegBkServer | 锛燂紵 |177 - test - 3.9 | 0.8382875515184536
2018-10-30 11:02:19,037 | 0 | Thread-10 | RegBkServer | 锛燂紵 |573 - test - 2.3 | 0.44562622348248504
2018-10-30 11:02:19,038 | 4 | Thread-10 | RegBkServer | 锛燂紵 |253 - test - 2.8 | 0.30070393870904033
2018-10-30 11:02:19,039 | 4 | Thread-10 | RegBkServer | 锛燂紵 |743 - test - 1.13 | 0.15035660109816618
2018-10-30 11:02:19,040 | 1 | Thread-10 | RegBkServer | 锛燂紵 |464 - test - 1.2 | 0.1999089669388775
2018-10-30 11:02:19,041 | 4 | Thread-10 | RegBkServer | 锛燂紵 |27 - test - 2.19 | 0.10232274453122792
2018-10-30 11:02:19,042 | 0 | Thread-10 | RegBkServer | 锛燂紵 |913 - test - 3.19 | 0.16921315018936867
```

```
%{TIMESTAMP_ISO8601:time} \| %{DATA:logLevel} \| %{DATA:threadName} \| %{DATA:className} \| .*? \|%{DATA:lineNo} \- %{DATA:packageName} \- %{DATA:version} \| %{GREEDYDATA:msg}
```



#### 51-tode

```
[2018-10-30 11:02:19,046]INFO:mirrors.toushibao.com:bbl:5393:[dataClear.OtherServerLogData:511]: java.lang.RuntimeException
[2018-10-30 11:02:19,046]ERROR:mirrors.toushibao.com:bbl:5393:[dataClear.OtherServerLogData:759]: java.lang.NullPointerException
[2018-10-30 11:02:19,046]WARN:mirrors.toushibao.com:bbl:5393:[dataClear.OtherServerLogData:882]: java.lang.RuntimeException
[2018-10-30 11:02:19,046]ERROR:mirrors.toushibao.com:bbl:5393:[dataClear.OtherServerLogData:679]: java.lang.ThreadDeath
[2018-10-30 11:02:19,046]ERROR:mirrors.toushibao.com:bbl:5393:[dataClear.OtherServerLogData:82]: java.lang.RuntimeException
[2018-10-30 11:02:19,046]DEBUG:mirrors.toushibao.com:bbl:5393:[dataClear.OtherServerLogData:449]: java.lang.NullPointerException
```

```
\[%{DATA:time}\]%{DATA:logLevel}\:%{DATA:hostName}\:%{DATA:serverName}\:%{DATA:pid}\:\[%{DATA:sourceFileName}\:%{DATA:lineNo}\]\:%{GREEDYDATA:msg}
```

对的



#### 61-tuxedo



```
1540897339029.mirrors.toushibao.com!bbl:5491.42.91:getTuxedoRunLogData:Tuxedo.log:663:WARN: java.lang.ThreadDeath
1540897339032.mirrors.toushibao.com!bbl:5491.42.44:getTuxedoRunLogData:Tuxedo.log:30:INFO: java.lang.FileNotFoundException
1540897339035.mirrors.toushibao.com!bbl:5491.42.70:getTuxedoRunLogData:Tuxedo.log:722:INFO: java.lang.RuntimeException
1540897339036.mirrors.toushibao.com!bbl:5491.42.67:getTuxedoRunLogData:Tuxedo.log:492:FATAL: java.lang.SQLException
1540897339038.mirrors.toushibao.com!bbl:5491.42.23:getTuxedoRunLogData:Tuxedo.log:53:INFO: java.lang.ThreadDeath
1540897339039.mirrors.toushibao.com!bbl:5491.42.84:getTuxedoRunLogData:Tuxedo.log:16:TRACE: java.lang.NullPointerException
1540897339041.mirrors.toushibao.com!bbl:5491.42.64:getTuxedoRunLogData:Tuxedo.log:559:INFO: java.lang.ArrayIndexOutOfBoundsException
```

```
%{DATA:time}\.%{DATA:hostName}\!%{DATA:serverName}:%{DATA:pid}\.%{DATA:thread_ID}\.%{DATA:context_ID}\:%{DATA:catalog_name}\:%{DATA:message_number}\:%{DATA:logLevel}\: %{GREEDYDATA:msg}
```

```
UNIX_MS
```



#### 71-wmq

```
2018-10-30 11:02:19 - Thread-7 root Program mirrors.toushibao.com Installation(info) 25.171-b10 OtherServerLogData otherserverlogdata = new OtherServerLogData: java.lang.NullPointerException
2018-10-30 11:02:19 - Thread-7 root Program mirrors.toushibao.com Installation(info) 25.171-b10 OtherServerLogData otherserverlogdata = new OtherServerLogData: java.lang.FileNotFoundException
2018-10-30 11:02:19 - Thread-7 root Program mirrors.toushibao.com Installation(info) 25.171-b10 OtherServerLogData otherserverlogdata = new OtherServerLogData: java.lang.FileNotFoundException
2018-10-30 11:02:19 - Thread-7 root Program mirrors.toushibao.com Installation(info) 25.171-b10 OtherServerLogData otherserverlogdata = new OtherServerLogData: java.lang.NoClassDefFoundError
2018-10-30 11:02:19 - Thread-7 root Program mirrors.toushibao.com Installation(info) 25.171-b10 OtherServerLogData otherserverlogdata = new OtherServerLogData: java.lang.NullPointerException
2018-10-30 11:02:19 - Thread-7 root Program mirrors.toushibao.com Installation(info) 25.171-b10 OtherServerLogData otherserverlogdata = new OtherServerLogData: java.lang.ThreadDeath
```

```
%{DATESTAMP:date} \- %{DATA:thread} %{DATA:user} %{DATA:program} %{DATA:host} %{DATA:installInfo}\s+%{DATA:version} %{DATA:ErrCode} %{GREEDYDATA:ErrMsg}
```





#### 81-amq



```
2018-10-30 11:02:19,046 | FATAL | java.lang.NoClassDefFoundError | dataClear.OtherServerLogData | Thread-2

2018-10-30 11:02:19,046 | INFO | java.lang.RuntimeException | dataClear.OtherServerLogData | Thread-2

2018-10-30 11:02:19,046 | ERROR | java.lang.FileNotFoundException | dataClear.OtherServerLogData | Thread-2

2018-10-30 11:02:19,046 | INFO | java.lang.ClassCastException | dataClear.OtherServerLogData | Thread-2

2018-10-30 11:02:19,047 | ERROR | java.lang.SQLException | dataClear.OtherServerLogData | Thread-2

2018-10-30 11:02:19,047 | FATAL | java.lang.NullPointerException | dataClear.OtherServerLogData | Thread-2
```

```
%{TIMESTAMP_ISO8601:time} \| %{DATA:logLevel} \| %{DATA:msg} \| %{DATA:className} \| %{GREEDYDATA:threadName}
```

