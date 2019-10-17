

离线包下载地址

```
https://pypi.doubanio.com/simple
```



例子

```shell
wget https://pypi.doubanio.com/packages/fc/49/82d64d705ced344ba458197dadab30cfa745f9650ee22260ac2b275d288c/SQLAlchemy-1.3.8.tar.gz#md5=2eb0a5a3da5054a9f8be4efacc4ffd95
tar -xvf SQLAlchemy-1.3.8.tar.gz
cd SQLAlchemy-1.3.8
python setup.py install
```



```shell
wget https://pypi.doubanio.com/packages/0c/ba/e521b9dfae78dc88d3e88be99c8d6f8737a69b65114c5e4979ca1209c99f/psycopg2-2.7.7-cp37-cp37m-manylinux1_x86_64.whl#md5=ac467b457304857e5cf33116f2d8cddd

pip install psycopg2-2.7.7-cp37-cp37m-manylinux1_x86_64.whl
```



```sql
CREATE TABLE tb_cmd_cfg (
	seq float8 NOT NULL,
	func_id varchar(40) NOT NULL,
	cfg_key varchar(40) NULL,
	memo varchar(100) NULL,
	exec_cmd text NULL,
	"enable" varchar(1) NULL,
	PRIMARY KEY (seq, func_id)
);


CREATE TABLE tb_cmd_cfg (
	seq float8 NOT NULL,
	func_id varchar(40) NOT NULL,
	cfg_key varchar(40) NULL,
	PRIMARY KEY (seq, func_id)
);
```

