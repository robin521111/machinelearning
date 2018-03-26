from microsoftbotframework import ReplyToActivity
import tensorflow as tf
import re
from chatbot import chatbot
use_tf = True
bot = chatbot.Chatbot()
#bot.main(['--test', 'daemon', '--modelTag', 'xiaohuang'])
bot.main(['--test', 'daemon', '--modelTag', 'cornell'])

def echo_response(message):
    if message["type"] == "message":
        if  use_tf:
            question = message["text"]
            answer = bot.daemonPredict(question)
            ReplyToActivity(fill=message, text=answer).send()
        else: 
            ReplyToActivity(fill=message, text=message["text"] + "").send()