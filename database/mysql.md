安装

```shell
brew install mysql
```

```shell
wget https://cdn.mysql.com/archives/mysql-8.0/mysql-server_8.0.16-2ubuntu18.04_amd64.deb-bundle.tar
```



安全设置和密码(12345678)

```shell
mysql_secure_installation
```



启停

```sh
brew services start mysql
brew services stop mysql
```



命令行登录

```shell
mysql -uroot -p12345678
```



数据库创建

```mysql
create database cmd_db character set utf8;
```



创建用户及授权

```mysql
create user 'yizt'@'%' identified by '12345678';
grant all privileges on cmd_db.* to 'yizt'@'%';
flush privileges;
```



```mysql
ALTER user 'yizt'@'%' IDENTIFIED WITH mysql_native_password BY '12345678';
flush privileges;
```





```shell
sqlacodegen --outfile db.py --tables user mysql+mysqlconnector://yizt:12345678@127.0.0.1:3306/cmd_db?auth_plugin=mysql_native_password


sqlacodegen --outfile db.py --tables TABLES mysql+mysqlconnector://yizt:12345678@127.0.0.1:3306/information_schema?auth_plugin=mysql_native_password
```

## 其它

1：sqlalchemy升级

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade sqlalchemy --ignore-installed
```







