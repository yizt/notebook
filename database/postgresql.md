

安装包下载

```shell
wget https://get.enterprisedb.com/postgresql/postgresql-9.5.19-1-linux-x64.run
```

配置文件/opt/PostgreSQL/9.5/data/postgresql.conf
```shell
# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
host    all     all     10.33.2.0/24    trust
host    all     all     10.33.3.0/24    trust
```

启停

```shell
/etc/init.d/postgresql-9.5 start
/etc/init.d/postgresql-9.5 stop
```





mac下

```
brew install postgresql
wget https://get.enterprisedb.com/postgresql/postgresql-9.5.19-1-osx.dmg
```



第三方包下载

```
export PATH=/Library/PostgreSQL/9.5/bin:$PATH
pip install psycopg2
```

```shell
sudo mv libpq.dylib libpq.dylib.bak
sudo ln -s /Library/PostgreSQL/9.5/lib/libpq.5.8.dylib libpq.dylib
sudo ln -s /Library/PostgreSQL/9.5/lib/libpq.5.8.dylib libpq.5.dylib
```

