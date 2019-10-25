# -*- coding: utf-8 -*-
"""
Created on 2019/10/2 上午10:21

@author: mick.yi

"""
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    # 表的名字:
    __tablename__ = 'user'

    # 表的结构:
    id = Column(String(20), primary_key=True)
    name = Column(String(20))


# 初始化数据库连接:
engine = create_engine('mysql+mysqlconnector://yizt:12345678@127.0.0.1:3306/cmd_db?auth_plugin=mysql_native_password')

#Base.metadata.create_all(engine)

# 创建DBSession类型:
DBSession = sessionmaker(bind=engine)

# 创建session对象:
session = DBSession()
# # 创建新User对象:
new_user = User(id='8')
# # 添加到session:
session.add(new_user)
# x = session.execute("update user set name='Bob3' where id ='6' ")
# print(x)
# 提交即保存到数据库:
session.commit()
# 关闭session:
session.close()





