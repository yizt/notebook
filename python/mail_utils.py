# -*- coding: utf-8 -*-
"""
 @File    : mail_utils.py
 @Time    : 2021/10/19 下午5:10
 @Author  : yizuotian
 @Description    :
"""

import smtplib
from email.header import Header
from email.mime.text import MIMEText

def qq_send(subject, msg):
    """

    :param subject: 邮件标题
    :param msg: 邮件内容
    :return:
    """
    # 第三方 SMTP 服务
    mail_host = "smtp.qq.com"  # 设置服务器
    mail_user = "315108378"  # 用户名
    mail_pass = "nayzfemwvqofcaeb"  # 口令

    sender = '315108378<315108378@qq.com>'
    receivers = [# 'sitechyizt@163.com'
         # ,
         'yizt<csuyzt@163.com>'
        #,'308145208@qq.com'
    ]  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

    message = MIMEText(msg, 'plain', 'utf-8')
    message['From'] = Header('315108378<315108378@qq.com>', 'utf-8')
    # message['To'] = Header(",".join(receivers), 'utf-8')
    message['To'] = Header(",".join(receivers), 'utf-8')

    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtp = smtplib.SMTP()
        smtp.connect(mail_host, 25)  # 25 为 SMTP 端口号
        smtp.login(mail_user, mail_pass)
        smtp.sendmail(sender,
                      # ['csuyzt@163.com'], #
                      receivers,
                      message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e)
        print("Error: 无法发送邮件")


def send(subject, msg):
    """

    :param subject: 邮件标题
    :param msg: 邮件内容
    :return:
    """
    # 第三方 SMTP 服务
    mail_host = "smtp.163.com"  # 设置服务器
    mail_user = "sitechyizt"  # 用户名
    mail_pass = "XXQHWPMTYFQUFXZH"  # 口令

    sender = 'sitechyizt<sitechyizt@163.com>'
    receivers = [# 'sitechyizt@163.com'
         # ,
         'yizt<csuyzt@163.com>'
        #,'308145208@qq.com'
    ]  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱

    message = MIMEText(msg, 'plain', 'utf-8')
    message['From'] = Header('sitechyizt<sitechyizt@163.com>', 'utf-8')
    # message['To'] = Header(",".join(receivers), 'utf-8')
    message['To'] = Header(",".join(receivers), 'utf-8')

    message['Subject'] = Header(subject, 'utf-8')

    try:
        smtp = smtplib.SMTP()
        smtp.connect(mail_host, 25)  # 25 为 SMTP 端口号
        smtp.login(mail_user, mail_pass)
        smtp.sendmail(sender,
                      # ['csuyzt@163.com'], #
                      receivers,
                      message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e)
        print("Error: 无法发送邮件")


if __name__ == '__main__':
    s = """仿射变换原理是?"""
    qq_send('问题咨询', s)