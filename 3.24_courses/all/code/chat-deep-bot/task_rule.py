# -*- coding: UTF-8 -*-
from microsoftbotframework import ReplyToActivity
import tensorflow as tf
import re
import requests
import io
import sys
#改变标准输出的默认编码
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
locations = ["杭州", "重庆", "上海", "北京"]
use_rule = True
def echo_response(message):
    if message["type"] == "message":
        if  use_rule:
            answer = str(actWeather(message["text"]))
            ReplyToActivity(fill=message, text=answer).send()
        else: 
            ReplyToActivity(fill=message, text=message["text"] + "").send()


def actWeather(inputs_strs):
    location = "北京"
    outstrs = []
    for i in locations:
        if (i in inputs_strs) and ("天气" in inputs_strs):
            location = i 
            outstrs.append("地点： %s\n     气温： %s\n     注意： %s" % (location, "10度", "多穿衣服"))
            return outstrs
    print(location)
    return "我怎样能帮您"

def actRegex(inputs_strs):
    # 参考：https://www.cnblogs.com/huxi/archive/2010/07/04/1771073.html
    location = "北京"
    p = re.compile(r'\d+')
    result = p.findall('one1two2three3four4')
    print(result)
    print(len(result))
    return "我怎样能帮您"

# print(actWeather("北京"))
print(actRegex("test"))