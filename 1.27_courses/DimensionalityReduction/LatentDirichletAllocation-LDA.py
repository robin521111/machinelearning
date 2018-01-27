#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
对文本进行降维
课上详述过程
"""
print(__doc__)


import os 
import numpy as np
try:
    import jieba
    def get_file_list(dirs):
        """
        dirs:输入文件夹
        output：获取文件夹中所有文件和文件名
        """
        file_names = []
        names = []
        for root, dirs, files in os.walk(dirs):
            for file in files:
                file_names.append(root+"/"+file)
                names.append(file.split("_")[0])
        return file_names, names
    def get_words(dirs):
        """
        dirs:输入文件夹
        out：获取分词结果，并进行保存
        """
        file_names, file_class = get_file_list(dirs)
        words = []
        print("Seging...")
        for itr in file_names:
            file = open(itr, "r", encoding="utf-8")
            txt = file.read()
            seg_list = jieba.cut(txt, cut_all=False)
            words.append(" ".join(seg_list))
        print("Finished!")
        np.savez("data/words.npz", data=np.array(words), label=np.array(file_class))
        return words, file_class
except:
    pass
finally:
    pass
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
def vector():
    """
    向量化并用LDA进行降维
    """
    data_files = np.load("data/words.npz")
    topic = set(data_files['label'])
    n_topic = len(topic)
    print(n_topic)

    vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=10000)
    print("Vectoring....")
    vect = vectorizer.fit_transform(data_files['data'])
    print("LDA...")
    lda = LatentDirichletAllocation(n_components=n_topic, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lines = lda.fit_transform(vect)
    feature_names = vectorizer.get_feature_names()
    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        ttt = "Topic #%d: " % topic_idx
        ttt += ' '.join([feature_names[i]
                             for i in topic.argsort()[:20 - 1:-1]])
        print(ttt)
        message = ([feature_names[i]
                             for i in topic.argsort()[:200 - 1:-1]])
        topic_words.append(message)
        np.savez("data/words-cloud.npz", topic=np.array(topic_words), data=lines)
import treeview
def view_topic():
    """
    将主题进行绘图
    """
    files = np.load("data/words-cloud.npz")
    for idx, itr in enumerate(files['topic']):
        treeview.view_cloud(itr, "figure/topic-%d.html"%idx)
    data = []

if __name__ == "__main__":
    #vector()
    view_topic()
    data_files = np.load("data/words.npz")
    data_files = data_files['data']
    outfile = open("figure/article.txt", "w")
    for itr in range(6):
        outfile.write("\n%s%d%s\n"%("="*6,itr,"="*6))
        outfile.write(data_files[itr].replace(" ", ""))
           
    data = np.load("data/words-cloud.npz")['data']
    treeview.view_weigh(data, "figure/article.html")
