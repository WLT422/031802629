# -*- coding: utf-8 -*-
import sys
import jieba
from gensim import corpora,models,similarities
from collections import defaultdict
#import psutil
#import os

#读取文档
with open(sys.argv[1],'r',encoding='UTF-8') as s1:
    txt1 = s1.read()
with open(sys.argv[2],'r',encoding='UTF-8') as s2:
    txt2 = s2.read()
#建立停用词，并处理文本
endflag = ['？','。','！','······']
texts = []
for line in txt1:
    words = ' '.join(jieba.cut(line)).split(' ')    # 利用jieba工具进行中文分词
    text = []
    # 过滤停用词，只保留不属于停用词的词语
    for word in words:
        if word not in endflag:
            text.append(word)
    texts.append(text)

words = ' '.join(jieba.cut(txt2)).split(' ')
new_text = []
for word in words:
    if word not in endflag:
        new_text.append(word)


#计算词语的频率
fre = defaultdict(int)
for text in texts:
    for word in text:
        fre[word]+=1
#建立词典
dictionary = corpora.Dictionary(texts)
#将要对比的文档通过doc2bow转化为稀疏向量
new_xs = dictionary.doc2bow(new_text)
#建立语料库，将文档转化为向量
corpus = [dictionary.doc2bow(text)for text in texts]
#初始化一个tfidf模型用来转换向量
tfidf = models.TfidfModel(corpus)
#将整个词库转换为tfidf表示
corpus_tfidf = tfidf[corpus]
#利用上一步得到的带有tfidf的语料库建立索引
index = similarities.MatrixSimilarity(corpus_tfidf)
#计算相似度并返回最大值
sim = index[tfidf[new_xs]]
#print('%.2f'% max(sim))
res = '%.2f'% max(sim)
#写入文件
s3 = open(sys.argv[3],'w',encoding='UTF-8')
s3.write(str(res))
s3.close()

print(0)

#性能分析
#print(u'当前进程的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024) )
#print(u'当前进程的使用的CPU时间：%.4f s' % (psutil.Process(os.getpid()).cpu_times().user) )

     
