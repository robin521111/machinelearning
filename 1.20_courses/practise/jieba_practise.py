import jieba

words = list(jieba.cut("如果我们考虑重复词语的情况，也就是说，重复的词语我们视为其出现多次，直接按条件独立假设的方式推导，这样的模型叫作多项式模型。
                  "))
print(' '.join(words))

