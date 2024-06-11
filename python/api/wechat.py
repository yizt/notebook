# -*- coding: utf-8 -*-
"""
 @File    : wechat.py
 @Time    : 2020/9/3 上午9:16
 @Author  : yizuotian
 @Description    :
"""

import cv2
import numpy as np
import requests
from requests_toolbelt import MultipartEncoder



# Secret = "5nf5BbhprxWu25P9FMbuM38Bo4Daze6oL8OYxtKANJE"
# corpid = 'wxbb4d69853cac0ac2'
# url = 'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={}&corpsecret={}'
#
# getr = requests.get(url=url.format(corpid, Secret))
#
# access_token = getr.json().get('access_token')
#
# print(access_token)


def upload(token, im_path):
    url_addr = "https://qyapi.weixin.qq.com/cgi-bin/media" \
               "/upload?access_token={}&type={}".format(token, 'image')
    m = MultipartEncoder(
        fields={"filename": ('file', open(im_path, 'rb'), 'application/octet-stream')},
    )
    r = requests.post(url=url_addr, data=m, headers={'Content-Type': m.content_type})
    print(r.json())
    return r


def download(tocken, media_id):
    url_addr = "https://qyapi.weixin.qq.com/cgi-bin/media" \
               "/get?access_token={}&media_id={}".format(tocken, media_id)

    r = requests.get(url=url_addr)
    print(r)
    return r


# r = upload(access_token, '/Users/yizuotian/Pictures/chepai.jpg')
# media_id=r.json()['media_id']
access_token='feKcNVRLzAOsGYCe88GA1oP-hJtGT7ABIeZPOpORSniYlyf2A5nq8Zu0Ow2x6mbBdUf05MWZ7DPbFPEFxS2bt5jYGFSqnFBPFzlGi2G74UxFf1vyVyFC9p55A6cW9UzwRm8TC48qPLdwDhUrCyAe62978WyJHUz6klo4RNS6yLw6dceVpcNELyzOp75DyS9p9Qk9KoahR8N8daqF6N1kZg'
media_id = '3UsVbLfepZNteELlLM09dn4vO1jQEUmJQgmqO3VHXjVhpLuiFncGLO4wXbPh9bJ-8'
r = download(access_token, media_id)

# with open('a.jpg', mode='wb') as w:
#     w.write(r.content)

response_byte = r.content
# f = open('a.jpg', 'rb')
# response_byte = f.read()
cv2.imshow('im', cv2.imdecode(np.fromstring(response_byte, 'uint8'), 1))
# bytes_stream = BytesIO(response_byte)
# capture_img = Image.open(bytes_stream)
# capture_img = cv2.cvtColor(np.asarray(capture_img), cv2.COLOR_RGB2BGR)
# cv2.imshow("capture_img", capture_img)
cv2.waitKey(0)
