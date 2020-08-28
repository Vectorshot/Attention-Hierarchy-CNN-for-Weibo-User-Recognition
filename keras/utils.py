import numpy as np
from keras.preprocessing.sequence import pad_sequences
def patchMatrix(doc,dst_size=(0,0),default=0):
    '''fill items with default value to unify each dimension of them
    '''
    if dst_size[1]==0:
        max_len=np.max([len(item) for item in doc])
    else:
        max_len=dst_size[1]
    for i in range(len(doc)):
        length=len(doc[i])
        if length<max_len:
            doc[i]=doc[i]+[default]*(max_len-length)
    if dst_size[0]>len(doc):
        doc.extend([[default]*max_len]*(dst_size[0]-len(doc)))
    return np.array(doc)#同database.py

class Worder(object):
    '''用于处理深度学习模型的输入
    1. 将句子中的词变为id
    2. 获取word2vec等词向量模型的weights，输入Embedding层中
    > 使用示例：  worder=Worder()
            train_data.content.apply(worder.fit)
            worder.build_weight(word2vec_model)
            train_data['word_xs']=train_data.content.apply(worder.transform)
    其中train_data的content为句子集合，每个句子中的词使用空格分割
    '''
    def __init__(self):
        self.vocabs=['\s']
        self.vocab2id={'\s':0}
        
    def fit(self,sentences):
        vocabs=self.vocabs
        vocab2id=self.vocab2id
        for item in sentences:
            for word in item.split(' '):
                if word not in vocab2id:
                    vocab2id[word]=len(vocabs)
                    vocabs.append(word)
    
    def build_weight(self,word2vec_model,dim,is_fast_text=False):
        if is_fast_text:
            weights=[word2vec_model[word] for word in self.vocabs]
        else:
            weights=[np.random.uniform(low=-0.1,high=0.1,size=(dim,)) 
                     if word not in word2vec_model 
                     else word2vec_model[word] for word in self.vocabs]
        exist_cnt=len([word for word in self.vocabs if word in word2vec_model])
        print('exist %d/%d, rate: %.2f'%(exist_cnt,len(self.vocabs),exist_cnt/len(self.vocabs)))
        self.weights=np.array(weights)
        
    def transform(self,sentences,patch_zeros=False):
        data=[]
        for item in sentences:
            record=[self.vocab2id[word] for word in item.split(' ')]
            data.append(record)
        if patch_zeros:
            return patchMatrix(data)
        else:
            return data
        
        
def pad_tensor2d(doc,dst_size=(0,0),default=0):
    '''fill items with default value to unify each dimension of them
    '''
    data=[]
    if dst_size[1]==0:
        max_len=np.max([len(item) for item in doc])
    else:
        max_len=dst_size[1]
    for i in range(len(doc)):
        length=len(doc[i])
        if length<max_len:
            tmp=[default]*(max_len-length)#不足补0
            data.append(tmp+doc[i])
        else:
            data.append(doc[i][:max_len])#除去过长部分
    if dst_size[0]>len(data):
        data.extend([[default]*max_len]*(dst_size[0]-len(doc)))
    elif dst_size[0]>0 and dst_size[0]<len(data):
        data=data[:dst_size[0]]
    return np.array(data)#同database.py的patchMatrix

def pad_zeros_2d(xs,ids):
    '''
    xs shapes: sentence id, char id, char one-hot
    ids: list of sentence id
    '''
    xas=[xs[id] for id in ids]
    return pad_sequences(xas,padding='post')#结尾补0


def pad_tensor3d(items,max_doc_len=None,max_sentence_len=None):
    '''
    用于文档级的Embedding层，word id矩阵补零
    '''
    l1=np.max([len(b) for b in items])
    if l1==0:
        return np.zeros((len(items),1,1))
    if max_doc_len is not None and l1>max_doc_len:
        l1=max_doc_len
    l2=0  
    for item in items:
        vals=[len(b) for b in item]
        if len(vals)>0:
            l2=max(l2,np.max(vals))
    if max_sentence_len is not None and l2>max_sentence_len:
        l2=max_sentence_len
    #l1最大句子数，l2每个句子的最大词数
    return np.array([pad_tensor2d(item,dst_size=(l1,l2)) for item in items])



def pad_zeros_3d(xs,ids):
    xas=[xs[id] for id in ids]
    return pad_tensor3d(xas)


def apply_generator(func,generator,steps):
    '''用于神经网络模型中输出中间层的结果
    func是K.function的返回值
    generator 样本生成器，与model.fit_generator传入的值相同
    steps 是 1+int((样本数-1)/batch_size)
    '''
    vals=[]
    for i,item in enumerate(generator):
        val=func(item[0]+[0])
        vals.append(val)
        if i+1==steps:
            break
    # 合并神经网络的各输出值
    cnt=len(vals[0])
    data=[]
    for i in range(cnt):
        data.append(np.concatenate([item[i] for item in vals]))
    return data