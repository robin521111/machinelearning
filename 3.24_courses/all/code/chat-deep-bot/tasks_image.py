from microsoftbotframework import ReplyToActivity
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pylab 

def echo_response(message):
    if message["type"] == "message":
        # 处理上传的为附件的数据
        if  "attachments" in message: 
            # 处理图像数据
            if "image" in message["attachments"][0]["contentType"]:
                url = message["attachments"][0]["contentUrl"]
                name = message["attachments"][0]["name"]
                import requests
                from io import BytesIO
                from PIL import Image
                # 获取图像二进制数据
                response = requests.get(url)
                # 将图像读取
                img = Image.open(BytesIO(response.content))
                # 处理...
                # 将结果图像保存本地
                img.save("D:\\test.jpg")
                # 返回图像，重要是告诉结果图像路径
                ReplyToActivity(fill=message, attachments=[{
                    "contentType": "image/jpeg",
                    "contentUrl": "D:\\test.jpg",
                    "name": "test.jpg"}]).send()
            else:
                # 返回文本 
                ReplyToActivity(fill=message, text = "test").send()
        else: 
            # 返回文本 
            ReplyToActivity(fill=message, text = "test").send()
        