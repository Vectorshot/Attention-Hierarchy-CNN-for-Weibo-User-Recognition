import numpy as np
from keras.models import Model

class YModel(Model):
    def __init__(self, **kwargs):
        super(YModel, self).__init__(**kwargs)
        
    def ytrain(self,xs,ys,epoch=500):

        n_batch=len(xs)
        #print('n_batch 一共分为batch:%d' % n_batch)
        
        #np.random.seed(27) 添加后模型效果下降 3个点左右，因为选种子是比较重要的过程
        indexes=np.random.randint(low=0,high=n_batch,size=(epoch,))
            
        loss,acc,total=0,0,0
        for n,i in enumerate(indexes):
            val=self.train_on_batch(xs[i],ys[i])#返回：数据块在现有模型中的误差率或者元组
            num=len(xs[i]) #一个batch中有多少样本
            #print('这批样本有%d个' % num)
            #print(xs[i][0].shape)
            #print(ys[i][0].shape)
            loss,acc,total=loss+val[0]*num,acc+val[1]*num,total+num
            if (n+1)%10==0:
                print('\r%d/%d'%(n+1,epoch),val[0],val[1],end='')
            
        loss,acc=loss/total,acc/total
        return loss,acc

    def ytest(self,xs,ys):
        loss,acc,total=0,0,0
        for x_test,y_test in zip(xs,ys):
            val=self.test_on_batch(x_test,y_test)
            num=len(x_test)
            loss,acc,total=loss+val[0]*num,acc+val[1]*num,total+num
        loss,acc=loss/total,acc/total
        return loss,acc

    def ypredict(self,xs,xids):
        results=[]
        for x_test,ids in zip(xs,xids):
            val=self.predict_on_batch(x_test)
            results.extend(zip(val,ids))#在列表末尾一次性追加另一个序列中的多个值
        return results

    def fit_on_batch(self,train_data,valid_data=None,n_earlystop=30,filename='best.model',
                     cnt_in_epoch=100,n_epoch=500,best_type='best_acc'):
        best_loss=1000
        best_epoch=0
        best_acc=0
        early_stop=0
        n_stop=n_earlystop
        
        import datetime
        
            
        for i in range(n_epoch):
            early_stop+=1
            val=self.ytrain(train_data[0],train_data[1],cnt_in_epoch)
            print('\r',i+1,'train','loss',val[0],'acc',val[1],'    ')
            if valid_data:
                print('testing...',end='')
                val=self.ytest(valid_data[0],valid_data[1])
                if (val[0]<best_loss and best_type=='best_loss') or (val[1]>best_acc and best_type=='best_acc'):
                    print(best_type)
                    best_loss=val[0]
                    best_epoch=i
                    best_acc=val[1]
                    self.save_weights(filename)
                    early_stop=0
                t=datetime.datetime.now().strftime('%H:%M:%S')
                print('\r',i+1,'test',t,'loss:%f, acc:%f'%val)
                print('-----')
                if early_stop>n_stop:
                    print('early stop')
                    break
        if valid_data:
            print('best:',best_epoch,best_loss,best_acc)
            self.load_weights(filename)
        else:
            best_epoch=n_epoch
            best_loss=val[0]
            best_acc=val[1]
            self.save_weights(filename)
        return best_epoch,best_loss,best_acc