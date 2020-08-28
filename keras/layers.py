from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras.layers.merge import concatenate
from keras.backend.common import normalize_data_format
import numpy as np

class AttentionPoolingLayer(Layer):
    '''The layer need mask. 
    The attention_dim can be any value. It's only used inside the layer.
    If set return_sequences=False, the second last dimension will be removed(average pooling). 
    e.g. Input: 32(batch)x50(sentences)x100(words)x300(vectors)
        Output: 32x50x300
        That means you input each words' vector and the layer outputs the sentence's vector.
        The dimensions of mask should be 32x50x100
        The mask is used to record how many words in each sentence.
    '''
    def __init__(self, attention_dim=128,multiple_inputs=False,return_sequences=False,**kwargs):
        self.supports_masking = True
        self.attention_dim = attention_dim
        self.multiple_inputs=multiple_inputs
        self.return_sequences=return_sequences
        super(AttentionPoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.multiple_inputs:
            n_in = input_shape[1][-1]
        else:
            n_in=input_shape[-1]
        n_out = self.attention_dim
        lim = np.sqrt(6. / (n_in + n_out))
        #self.W = self.add_weight(name='{}_W', shape=(n_in, n_out), initializer='uniform', trainable=True)
        W = K.random_uniform_variable((n_in, n_out), -lim, lim, name='{}_W'.format(self.name))  #均匀分布
        b = K.zeros((n_out,), name='{}_b'.format(self.name))#self.name代替{}
        self.W = W
        self.b = b

        self.v = K.random_normal_variable(shape=(n_out,1),mean=0, scale=0.1, name='{}_v'.format(self.name))  #正态分布
        self.trainable_weights = [self.W, self.v, self.b]
       
        
    def call(self, x, mask=None):
        # x shape: 32(batch)x50(sentences)x100(words)x300(vector)
        # self.W: 300x{attention_dim}
        # 多输入的意思就是一个用于计算attention（xa），然后一个用于获取加权后的表示向量（xt）
        if self.multiple_inputs==False:
            xt=x
            xa=x
        else:
            xt=x[0]
            xa=x[1]
            mask=mask[0]
        atten = K.tanh(K.dot(xa, self.W) + self.b)  # 32x50x100x{attention_dim}
        # self.v: {attention_dim}*1 正态分布和为1
        atten=K.dot(atten,self.v) # 32x50x100*1
        atten=K.sum(atten,axis=-1) # 32*50*100  相当于K.squeeze(atten)
        atten = self.softmask(atten, mask)  # 32x50x100

        self.attention=atten
        atten=K.expand_dims(atten) # 32x50x100x1
        output = atten * xt
        if self.return_sequences==False:
            output = K.sum(output, axis=-2)  # sum the second last dimension
        return output
    
    def compute_output_shape(self, input_shape):
        if self.multiple_inputs:
            input_shape=input_shape[0]
        if self.return_sequences:
            return input_shape
        else:
            shape = list(input_shape)
            return tuple(shape[:-2] + shape[-1:])
    
    def compute_mask(self, x, mask=None):
        if self.multiple_inputs:
            mask=mask[0]
        if self.return_sequences==True:
            return mask
        elif mask is not None and K.ndim(mask)>2:#ndim:返回维数
            return K.equal(K.all(K.equal(mask,False),axis=-1),False)#equal逐个元素对比，axis=-1 倒数第一维 all:逻辑and
        else:
            return None
        
    def softmask(self,x, mask,axis=-1):
        '''
        softmax with mask, used in attention mechanism others
        '''
        y = K.exp(x)
        if mask is not None:
            mask=K.cast(mask,'float32')
            y = y * mask
        sumx = K.sum(y, axis=axis, keepdims=True) + 1e-6
        x = y / sumx
        return x
    
    
class ConcatLayer(Layer):
    def __init__(self,**kwargs):
        super(ConcatLayer, self).__init__(**kwargs)
       
    def build(self, input_shape):
       
        self.built = True

    def call(self, inputs):
        title_xs,tag_x=inputs
        shape=K.shape(title_xs)
        if K.ndim(title_xs)==4:
            tx=K.expand_dims(tag_x,axis=-2)
            tx=K.expand_dims(tx,axis=-2)
            tx=K.tile(tx,[1,shape[1],shape[2],1])
        elif K.ndim(title_xs)==3:
            tx=K.expand_dims(tag_x,axis=-2)
            tx=K.tile(tx,[1,shape[1],1])
        outputs=concatenate([title_xs,tx],)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[0])[:-1]+[input_shape[0][-1]+input_shape[1][-1]])
    
    def compute_mask(self, inputs, mask=None):
        return mask[0]
    
class Embedding(Layer):
    '''支持篇章级（4维）输入的词嵌入层
    '''
    def __init__(self, input_dim, output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        kwargs['dtype'] = 'int32'
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(Embedding, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.input_length = input_length

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return tuple(list(input_shape)+[self.output_dim])

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.embeddings, inputs)
        return out

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero,
                  'input_length': self.input_length}
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class ConvDense(Layer):
    '''
    使用二维卷积的方法，对3维Tensor中的一维进行变换
    例如，当units=128时：
    假设输入 Tensor 32*200*300
    输出 Tensor 32*200*128
    即对最后一维进行变换
    '''
    def __init__(self, units,activation=None,**kwargs):
        super(ConvDense, self).__init__(**kwargs)
        self.filters = units
        rank=2
        strides=(1,1)
        padding='valid'
        data_format=None
        dilation_rate=(1, 1)
        use_bias=True
        kernel_initializer='glorot_uniform'
        kernel_regularizer=None
        bias_regularizer=None
        activity_regularizer=None
        kernel_constraint=None
        bias_constraint=None
        bias_initializer='zeros'
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
    def build(self, input_shape):
        kernel_shape = (1,input_shape[-1])+(1, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs=K.expand_dims(inputs)
        
        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            outputs=self.activation(outputs)
        
        outputs=K.sum(outputs,axis=-2)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1]+[self.filters])
    
    def compute_mask(self, inputs, mask=None):
        return mask
    