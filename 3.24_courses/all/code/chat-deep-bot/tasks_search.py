from microsoftbotframework import ReplyToActivity
import tensorflow as tf
import re
import requests
# import io
# import sys
# #改变标准输出的默认编码
# sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
use_search = True
locations = ["杭州", "重庆", "上海", "北京"]

def echo_response_test(message):
    if message["type"] == "message":
        if  use_search:
            answer = str(actWeather(message["text"]))
            ReplyToActivity(fill=message, text=answer).send()
        else: 
            ReplyToActivity(fill=message, text=message["text"] + "").send()


def actWeather(inputs_strs):
    location = "北京"
    for i in locations:
        if (i in inputs_strs) and ("天气" in inputs_strs):
            location = i 
            page = requests.get("http://wthrcdn.etouch.cn/weather_mini?city=%s" %(location))
            data = page.json()
            temperature = data['data']['wendu']
            notice = data['data']['ganmao']
            outstrs = []
            outstrs.append("地点： %s\n     气温： %s\n     注意： %s" % (location, temperature, notice))
            return outstrs
    print(location)
    return "我怎样能帮您"

print(actWeather("北京"))