import numpy as np
import os
import pickle
from gensim.models.word2vec import Word2Vec

def patchMatrix(doc,dst_size=(0,0),default=0):
    '''fill items with default value to unify each dimension of them
    '''
    if dst_size[1]==0:
        max_len=np.max([len(item) for item in doc])
    else:
        max_len=dst_size[1]
    for i in range(len(doc)): #len() 方法返回列表元素个数。
        length=len(doc[i])
        if length<max_len:
            doc[i]=doc[i]+[default]*(max_len-length)#用0填充doc中长度不足的条
    if dst_size[0]>len(doc):
        doc.extend([[default]*max_len]*(dst_size[0]-len(doc))) #用全0doc条填充整个doc
    return doc

def get_batch_docs(in_docs):
    '''Input raw document list.
    Output numpy array(unify numbers of each document's sentences).
    '''
    #padding的长度是按每一批中长度最大的微博进行的
    max_sen_num=np.max([len(doc) for doc in in_docs])#总集in_docs中doc的最大长度
    max_word_num=np.max([np.max([len(sen) for sen in doc]) for doc in in_docs])#doc中sen的最大长度
    out_docs=np.array([patchMatrix(doc,dst_size=(max_sen_num,max_word_num)) for doc in in_docs],dtype=np.int32) #batch_size * 
    return out_docs

def get_patched_docs(docs,labels,ids,batch_size=32):
    '''Input raw document list, y values and id of documents.
    Output the list of numpy array. Each list has {batch_size} elements. 
    '''
    from keras.utils.np_utils import to_categorical
    data=zip(docs,labels,ids)#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    data=sorted(data,key=lambda x:len(x[0]),reverse=True) 
    docs,labels,ids=zip(*data)#解压
    #合并成batch
    n_batch=int((len(docs)-1)/batch_size)+1
    #patched_docs=[get_batch_docs(docs[i*batch_size:(i+1)*batch_size]) for i in range(n_batch)]
    #patched_xs=[xs[i*batch_size:(i+1)*batch_size] for i in range(n_batch)]
    
    patched_docs=[get_batch_docs(docs[i*batch_size:(i+1)*batch_size]) for i in range(n_batch)]
    ys=to_categorical(labels)
    patched_ys=[ys[i*batch_size:(i+1)*batch_size] for i in range(n_batch)]
    patched_ids=[ids[i*batch_size:(i+1)*batch_size] for i in range(n_batch)]
    return patched_docs,patched_ys,patched_ids

import gensim
class SmpCupText(object):
    '''give the path stored data
    use get_data method to get data and you need give it the parameter 'task'
    task 0: gender
    task 1: age
    task 2: location
    # Input Parameters:
        path: the directory stored 'SmpCup.txt'
    '''
    def __init__(self,path,
                 text_name='SmpCup.txt',
                 word2vec_name='SmpCup.word2vec.pkl',
                 y_name='SmpCup.y.pkl',stopwords_name='stopwords.txt',fill_empty=True):
        self.path=path
        self.train_num=3200
        self.is_transformed=False
        self.split_char='\$\$yuml\$\$'
        self.fill_empty=fill_empty
        self._load_text(os.path.join(self.path,text_name))
        self.ys=pickle.load(open(os.path.join(self.path,y_name),'rb'))
        self.word2vec=pickle.load(open(os.path.join(self.path,word2vec_name),'rb'))
        
        '''load stopwords from file'''
        if stopwords_name and os.path.exists(os.path.join(self.path,stopwords_name)):
            with open(os.path.join(self.path,stopwords_name),encoding='utf8') as f:
                self.stopwords=[item.strip() for item in f.readlines()]
        self._transform()
        
    def _load_text(self,filename):
        import re
        with open(filename,encoding='utf8') as f:
            data=[item.strip().split(',',maxsplit=4) for item in f.readlines()]
            self.ids=[item[0] for item in data]
            self.answers=['%s,%s,%s'%(item[1],item[2],item[3]) for item in data]
            self.contents=[re.split(self.split_char,item[4]) for item in data]
            
    def _transform(self):
        '''transform text to integer'''
        voc={}  #按词出现的顺序排列（python字典好像是无序的），并没有按词频排列。而且词典使用了全长并没有截断。
        id2word=['']
        docs=[]
        weights=[np.zeros((300,))]  #词向量矩阵,embedding_matrix，第一个用于mask!
        for doc in self.contents:
            dst_doc=[]
            for sen in doc:
                dst_sen=[]
                for word in sen.split(' '):
                    if word not in voc:
                        if word in self.word2vec or self.fill_empty:
                            voc[word]=len(voc)+1
                            if word in self.word2vec:
                                weights.append(self.word2vec[word])
                            elif self.fill_empty:
                                weights.append(np.random.uniform(low=-0.1,high=0.1,size=(300,)))
                            id2word.append(word)
                    if word in voc:
                        dst_sen.append(voc[word])
                dst_doc.append(dst_sen)
            docs.append(dst_doc)
        self.weights=np.array(weights,dtype=np.float32)
        self.voc=voc
        self.docs=docs
        self.id2word=id2word


    def get_data(self,task=2,batch_size=32,train_index=None,test_index=None):
        '''task can be 0->gender,1->age,2->location'''
        if self.is_transformed==False:
            self._transform()
        if train_index==None:
            train_num=self.train_num
            train_data=get_patched_docs(self.docs[:train_num],self.ys[:train_num,task], self.ids[:train_num], batch_size)
            test_data=get_patched_docs(self.docs[train_num:],self.ys[train_num:,task], self.ids[train_num:], batch_size)
        else:
            
            train_data=get_patched_docs([self.docs[i] for i in train_index],
                                        self.ys[train_index,task], 
                                        [self.ids[i] for i in train_index], batch_size)
            test_data=get_patched_docs([self.docs[i] for i in test_index],
                                       self.ys[test_index,task],
                                       [self.ids[i] for i in test_index], batch_size)

        return train_data,test_data