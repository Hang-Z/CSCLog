from collections import defaultdict
import math
import operator
import pandas as pd
import re
import torch
import json

from transformers import BertTokenizer, BertModel, AutoTokenizer

def get_keys(sentence):
    stopwords = ['in','on','with','by','for','at','about','under','of','to','from']
    line = sentence.lower()
    line = re.sub('[^\w\u4e00-\u9fff]+',' ',line)
    return [str(x) for x in line.split() if x not in stopwords]
    
def feature_select(list_words):
    #总词频统计
    doc_frequency=defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i]+=1
 
    #计算每个词的TF值
    word_tf={}  #存储每个词的tf值
    for i in doc_frequency:
        word_tf[i]=doc_frequency[i]/sum(doc_frequency.values())
 
    #计算每个词的IDF值
    doc_num=len(list_words)
    word_idf={} #存储每个词的idf值
    word_doc=defaultdict(int) #存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i]+=1
    for i in doc_frequency:
        word_idf[i]=math.log(doc_num/(word_doc[i]+1))
 
    #计算每个词的TF*IDF的值
    word_tf_idf={}
    for i in doc_frequency:
        word_tf_idf[i]=word_tf[i]*word_idf[i]
 
    # 对字典按值由大到小排序
    dict_feature_select=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    return dict_feature_select

def word_vec(keys, tokenizer, model):
    encode_keys = {}
    for word, weight in keys:
        encoded_input = tokenizer(word, return_tensors='pt')
        output = model(**encoded_input)
        encode_keys[word] = torch.mul(output[0].mean(dim = 1, keepdim = False)[0], weight)
    
    return encode_keys

def sentence_vec(sentences, keys, tokenizer, model):
    encode_keys = word_vec(keys, tokenizer, model)
    first_ = list(encode_keys.values())[0]
    encode_sentence = {}
    for line in sentences.items():
        line_vec = torch.zeros_like(first_)
        for word in line[1]:
            line_vec += encode_keys[word]
        encode_sentence[line[0]] = line_vec.tolist()
    return encode_sentence

name = '../data/HDFS/result_30s/HDFS_30s'
input_dir = name + '.log_templates.csv'
output_dir = name + '_sentences_emb.json'
model_path = "../model/bert"

# word Embedding编码
datas = pd.read_csv(input_dir)

# loading model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

sentences = {line[0]:get_keys(line[1]) for line in zip(datas['EventId'], datas['EventTemplate'])}
keys = feature_select([item[1] for item in sentences.items()])
encode_sentence = sentence_vec(sentences, keys, tokenizer, model)

save_js = json.dumps(encode_sentence)
f2 = open(output_dir, 'w')
f2.write(save_js)
f2.close()
print('Embedding Done')