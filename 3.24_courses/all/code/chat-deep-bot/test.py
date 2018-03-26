from microsoftbotframework import ReplyToActivity
import tensorflow as tf
import re
from chatbot import chatbot

bot = chatbot.Chatbot()
bot.main(['--test', 'daemon'])
question = "hello"
answer = bot.daemonPredict(question)
print(answer)
