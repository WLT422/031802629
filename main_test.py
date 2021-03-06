# -*- coding: utf-8 -*-
import sys
import jieba
from gensim import corpora,models,similarities
from collections import defaultdict
import os
#import psutil




def list_dir(text_list,dir_path):
    dir_files = os.listdir(dir_path)  # 得到该文件夹下所有的文件
    for file in dir_files:
        file_path = os.path.join(dir_path, file)  # 路径拼接成绝对路径
        if os.path.isfile(file_path):  # 如果是文件，就打印这个文件路径
            if file_path.endswith(".txt"):
                text_list.append(file_path)
        if os.path.isdir(file_path):  # 如果目录，就递归子目录
            list_dir(text_list,file_path)
    return text_list


if __name__ == '__main__':
    all_txt = []
    thesaurus_path = r"e:/yey/sim_0.8"
    text_list = list_dir(all_txt,thesaurus_path)
 










#读取文档
with open('e:/yey/sim_0.8/orig.txt','r',encoding='UTF-8') as s1:
    txt1 = s1.read()
for i in range(len(text_list)):

    with open(text_list[i],'r',encoding='UTF-8') as s2:
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
    #通过token2id得到特征数
    featurenum=len(dictionary.token2id.keys())
    #12、稀疏矩阵相似度，从而建立索引
    index=similarities.SparseMatrixSimilarity(tfidf[corpus],num_features=featurenum)
    #得出结果
    sim=index[tfidf[new_xs]]
    print(text_list[i])
    print('相似度为%s','%.2f'% max(sim))
    #res = '%.2f'% max(sim)
    #写入文件
    #s3 = open(sys.argv[3],'w',encoding='UTF-8')
    #s3.write(str(res))
    #s3.close()

    #性能分析
    #print(u'当前进程的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024) )
    #print(u'当前进程的使用的CPU时间：%.4f s' % (psutil.Process(os.getpid()).cpu_times().user) )

    print(0)


