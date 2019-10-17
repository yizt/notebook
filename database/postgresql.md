

安装包下载

```shell
wget https://get.enterprisedb.com/postgresql/postgresql-9.5.19-1-linux-x64.run
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

