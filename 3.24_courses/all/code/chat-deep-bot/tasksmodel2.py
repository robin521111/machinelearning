from microsoftbotframework import ReplyToActivity
import tensorflow as tf
import re
from chatbot import chatbot
from a3_inference import InferenceObject
use_tf = True

inference = InferenceObject()
inference.main(args = ['--test', '--modelTag', 'xiaohuang','--vocabularySize', '1000','--corpus', 'xiaohuang',  "--numEpochs", "10", "--maxLength", "10", "--filterVocab", "0"])

def echo_response(message):
    if message["type"] == "message":
        if  use_tf:
            question = message["text"]
            answer = inference.daemonPredict("hi")
            ReplyToActivity(fill=message, text=answer).send()
        else: 
            ReplyToActivity(fill=message, text=message["text"] + "").send()