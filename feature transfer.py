# -*- coding: utf-8 -*-


"""
Created on Sun Oct 18 20:58:16 2020

@author: pjx
"""
# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
import keras
from keras import Input
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json, Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from MMD_tensorflow import mmd_loss
import itertools
import tensorflow as tf
import keras.backend as K
from keras.objectives import categorical_crossentropy



batch_size=20
leng=2


sess = tf.Session()
K.set_session(sess)
'''



tag1=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.007.csv",header=None))
tag2=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.014.csv",header=None))
tag3=np.array(pd.read_csv(r"D:\(497,247)西储大学 0.021.csv",header=None))
tag4=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.028.csv",header=None))


type1=np.array(pd.read_csv(r"D:\(496,247)西储大学 0.007 opposite.csv",header=None))
type2=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.014 center.csv",header=None))
type3=np.array(pd.read_csv(r"D:\(494,247)西储大学 0.021 opposite.csv",header=None))
type4=np.array(pd.read_csv(r"D:\(491,247)西储大学 0.028 ball3.csv",header=None))

#type1=np.array(pd.read_csv(r"D:\(497,247)西储大学 0.007mm 1797rpm opposite.csv",header=None))
#type2=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.014mm 1797rpm center.csv",header=None))
#type3=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.021mm 1797rpm opposite.csv",header=None))
#type4=np.array(pd.read_csv(r"D:\(491,247)西储大学 0.028mm 1797rpm ball3.csv",header=None))


type1=np.array(pd.read_csv(r"D:\(496,247)西储大学 0.007 opposite.csv",header=None))
type2=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.014 center.csv",header=None))
type3=np.array(pd.read_csv(r"D:\(494,247)西储大学 0.021 opposite.csv",header=None))
type4=np.array(pd.read_csv(r"D:\(491,247)西储大学 0.028 ball3.csv",header=None))

'''
#type1=np.array(pd.read_csv(r"D:\(497,247)西储大学 0.007mm 1797rpm opposite.csv",header=None))
#type2=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.014mm 1797rpm center.csv",header=None))
#type3=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.021mm 1797rpm opposite.csv",header=None))
#type4=np.array(pd.read_csv(r"D:\(491,247)西储大学 0.028mm 1797rpm ball3.csv",header=None))

type1=np.array(pd.read_csv(r"D:\(3496,247)燃油泵 叶片损伤1片.csv",header=None))
type2=np.array(pd.read_csv(r"D:\(3496,247)燃油泵 叶片损伤2片.csv",header=None))
type3=np.array(pd.read_csv(r"D:\(1748,247)燃油泵 0.02.csv",header=None))
type4=np.array(pd.read_csv(r"D:\(3496,247)燃油泵 叶片损伤10片.csv",header=None))


tag1=np.array(pd.read_csv(r"D:\(496,247)西储大学 0.007 opposite.csv",header=None))
tag2=np.array(pd.read_csv(r"D:\(495,247)西储大学 0.014 center.csv",header=None))
tag3=np.array(pd.read_csv(r"D:\(494,247)西储大学 0.021 opposite.csv",header=None))
tag4=np.array(pd.read_csv(r"D:\(491,247)西储大学 0.028 ball3.csv",header=None))



source=np.vstack((type1,type2,type3,type4))
tag=np.vstack((tag1,tag2,tag3,tag4))


# 载入数据

X_source1 = np.expand_dims(type1[:, 0:246].astype(float), axis=2)
X_source2 = np.expand_dims(type2[:, 0:246].astype(float), axis=2)
X_source3 = np.expand_dims(type3[:, 0:246].astype(float), axis=2)
X_source4 = np.expand_dims(type4[:, 0:246].astype(float), axis=2)

Y_source = source[:, 246]


X_tag1 = np.expand_dims(tag1[:, 0:246].astype(float), axis=2)
X_tag2 = np.expand_dims(tag2[:, 0:246].astype(float), axis=2)
X_tag3 = np.expand_dims(tag3[:, 0:246].astype(float), axis=2)
X_tag4 = np.expand_dims(tag4[:, 0:246].astype(float), axis=2)


Y_tag = tag[:, 246]

#type，source代表目标域，tag代表源域

 
# 湿度分类编码为数字
#comb=np.hstack((Y_source,Y_tag))
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y_source)
Y_tag_encoded = encoder.fit_transform(Y_tag)
#Y_onehot1 = np_utils.to_categorical(Y_encoded)
Y_source_onehot=np_utils.to_categorical(Y_encoded)
Y_tag_onehot=np_utils.to_categorical(Y_tag_encoded)


X_source1_train=X_source1[:leng]
Y_source1_train=Y_source_onehot[:leng]

X_source2_train=X_source2[:leng]
Y_source2_train=Y_source_onehot[len(X_source1):len(X_source1)+leng]

X_source3_train=X_source3[:leng]
Y_source3_train=Y_source_onehot[len(X_source1)+len(X_source2):len(X_source1)+len(X_source2)+leng]

X_source4_train=X_source4[:leng]
Y_source4_train=Y_source_onehot[len(X_source1)+len(X_source2)+len(X_source3):len(X_source1)+len(X_source2)+len(X_source3)+leng]




X_tag1_train=X_tag1
Y_tag1_train=Y_tag_onehot[:len(X_tag1_train)]

X_tag2_train=X_tag2
Y_tag2_train=Y_tag_onehot[len(X_tag1_train):len(X_tag1_train)+len(X_tag2_train)]

X_tag3_train=X_tag3
Y_tag3_train=Y_tag_onehot[len(X_tag1_train)+len(X_tag2_train):len(X_tag1_train)+len(X_tag2_train)+len(X_tag3_train)]

X_tag4_train=X_tag4
Y_tag4_train=Y_tag_onehot[len(X_tag1_train)+len(X_tag2_train)+len(X_tag3_train):]

X2=np.vstack((X_source1,X_source2,X_source3,X_source4))

#a=np.random.permutation(X_tag1_train)
#print(a)


'''
def X_train():
    a=np.random.permutation(X_tag1_train)
    b=np.random.permutation(X_tag2_train)
    c=np.random.permutation(X_tag3_train)
    d=np.random.permutation(X_tag4_train)
    return (np.vstack((a[0])))
'''







 #= np_utils.to_categorical()

#Y_tag_encoded = encoder.fit_transform(Y_tag)
#Y_tag_onehot "= np_utils.to_categorical(Y_tag"_encoded)


# 划分训练集，测试集
'''
X_source_train,_,Y_source_train,_=train_test_split(X_source, Y_source_onehot, test_size=0.001, random_state=0)

X_tag_train,_,Y_tag_train,_=train_test_split(X_tag, Y_tag_onehot, test_size=0.001, random_state=0)



X_train=X_source_train[:1500]
Y_train=Y_source_train[:1500]
X_test1=X_tag_train
Y_test1=Y_tag_train

#X_test=np.repeat(X_test1[:150],10,axis=0)
#Y_test=np.repeat(Y_test1[:150],10,axis=0)

X_test=np.vstack((X_test1[:150] for _ in range(10)))
Y_test=np.vstack((Y_test1[:150] for _ in range(10)))

#X_test=np.vstack((X_test1[:150],X_test1[:150],X_test1[:150],X_test1[:150],X_test1[:150],X_test1[:150],X_test1[:150],X_test1[:150],X_test1[:150],X_test1[:150],X_test1[:150]))
#Y_test=np.vstack((Y_test1[:150],Y_test1[:150],Y_test1[:150],Y_test1[:150],Y_test1[:150],Y_test1[:150],Y_test1[:150],Y_test1[:150],Y_test1[:150],Y_test1[:150],Y_test1[:150]))
X2=X_test1
Y_onehot2=Y_test1

#X_test=X_test1[:1500]
#Y_test=Y_test1[:1500]
#X2=X_test1[1500:3000]
#Y_onehot2=Y_test1[1500:3000]
    
'''
# 定义神经网络
def baseline_model():
    origin_input=Input(shape=(246, 1))
    cross_input=Input(shape=(246, 1))
    
    model = Sequential()
    model.add(Conv1D(16, 3,input_shape=(246, 1)))#
    model.add(Conv1D(16, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    
    #origin_seq,cross_seq=model([origin_input,cross_input])
    origin_seq=model(origin_input)
    cross_seq=model(cross_input)
    sess = tf.Session()
    K.set_session(sess)
    
    distance=mmd_loss(origin_seq,cross_seq,1)

        

    den=Dense(9, activation='softmax')
    #origin_output,cross_output=Dense(9, activation='softmax')([origin_seq,cross_seq])
    origin_output=den(origin_seq)
    cross_output=den(cross_seq)
    model=Model(inputs=[origin_input,cross_input],outputs=[origin_output,cross_output,tf.convert_to_tensor(distance)])
    '''
    def cross_error(c,d):  #(y_actual_a,y_actual_b,y_predicted_a,y_predicted_b):
        print('-----')
        print(keras.backend.shape(c))
        print(keras.backend.shape(d))
        print(c[0])
        print(c[1])
        print(d[0])
        print(d[1])
        #a=keras.backend.mean(keras.backend.square(c[0,:]-c[1,:]),axis=-1)
        #b=keras.backend.mean(keras.backend.square(d[0,:]-d[1,:]),axis=-1)
        a=keras.backend.mean(keras.backend.square(c[1]-d[1]))
        #b=keras.backend.mean(keras.backend.square(c[1]-d[1]),axis=-1)
        #distance=mmd_rbf(c,d)
        #a=keras.backend.mean(keras.backend.sum(keras.backend.square(y_actual_a-y_predicted_a)))
        #b=keras.backend.mean(keras.backend.sum(keras.backend.square(y_actual_b-y_predicted_b)))
        return a#*0.5+b*0.5#+distance
    '''
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss=['mean_squared_error','mean_squared_error','mean_squared_error'
                        ],
                  loss_weights=[0.8,1,1],
                  optimizer='adam', metrics=['accuracy'])
    return model
'''
def baseline_model():
    origin_input=Input(shape=(246, 1))
    cross_input=Input(shape=(246, 1))
    x=Conv1D(16, 3)(origin_input)#([origin_input,cross_input])#input_shape=(246, 1)
    x=Conv1D(16, 3, activation='tanh')(x)
    x=MaxPooling1D(3)(x)
    x=Conv1D(64, 3, activation='tanh')(x)
    x=Conv1D(64, 3, activation='tanh')(x)
    x=MaxPooling1D(3)(x)
    x=Conv1D(64, 3, activation='tanh')(x)
    x=Conv1D(64, 3, activation='tanh')(x)
    x=MaxPooling1D(3)(x)
    origin_seq,cross_seq=Flatten(x)
    #origin_seq,cross_seq=model([origin_input,cross_input])
    #model.add(Dense(9, activation='softmax'))
    origin_output,cross_output=Dense(9, activation='softmax')([origin_seq,cross_seq])
    print(tf.shape([origin_output,cross_output]))
    model=Model([origin_input,cross_input],[origin_output,cross_output])
    def cross_error(y_actual_a,y_actual_b,y_predicted_a,y_predicted_b):
        distance=mmd_rbf(origin_seq,cross_seq)
        a=keras.backend.mean(keras.backend.sum(keras.backend.square(y_actual_a-y_predicted_a)))
        b=keras.backend.mean(keras.backend.sum(keras.backend.square(y_actual_b-y_predicted_b)))
        return a+b+distance
    plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='cross_error',optimizer='adam', metrics=['accuracy'])
    return model

'''

origin_input=tf.placeholder(tf.float32,shape=(None,246, 1))#目标域输入
cross_input=tf.placeholder(tf.float32,shape=(None,246, 1))#源域输入
    
model = Sequential()
model.add(Conv1D(16, 3,input_shape=(246, 1)))#
model.add(Conv1D(16, 3, activation='tanh'))
model.add(MaxPooling1D(3))
model.add(Conv1D(64, 3, activation='tanh'))
model.add(Conv1D(64, 3, activation='tanh'))
model.add(MaxPooling1D(3))
model.add(Conv1D(64, 3, activation='tanh'))
model.add(Conv1D(64, 3, activation='tanh'))
model.add(MaxPooling1D(3))
model.add(Flatten())

model2 = Sequential()
model2.add(Conv1D(16, 3,input_shape=(246, 1)))#
model2.add(Conv1D(16, 3, activation='tanh'))
model2.add(MaxPooling1D(3))
model2.add(Conv1D(64, 3, activation='tanh'))
model2.add(Conv1D(64, 3, activation='tanh'))
model2.add(MaxPooling1D(3))
model2.add(Conv1D(64, 3, activation='tanh'))
model2.add(Conv1D(64, 3, activation='tanh'))
model2.add(MaxPooling1D(3))
model2.add(Flatten())
    
    #origin_seq,cross_seq=model([origin_input,cross_input])
origin_seq=model(origin_input)
cross_seq=model2(cross_input)




distance=mmd_loss(origin_seq,cross_seq,1)



#通过一个类别与其他类别的MMD值定系数

a=K.reshape(origin_seq[0],[1,-1])
aa=K.reshape(cross_seq[0],[1,-1])

b=K.reshape(origin_seq[1],[1,-1])
bb=K.reshape(cross_seq[1],[1,-1])

c=K.reshape(origin_seq[2],[1,-1])
cc=K.reshape(cross_seq[2],[1,-1])

d=K.reshape(origin_seq[3],[1,-1])
dd=K.reshape(cross_seq[3],[1,-1])



#wight_sort1=mmd_loss(a,aa,1)/(mmd_loss(a,bb,1)*mmd_loss(a,cc,1)*mmd_loss(a,dd,1))
#wight_sort2=mmd_loss(b,bb,1)/(mmd_loss(b,aa,1)*mmd_loss(b,cc,1)*mmd_loss(b,dd,1))
#wight_sort3=mmd_loss(c,cc,1)/(mmd_loss(c,aa,1)*mmd_loss(c,bb,1)*mmd_loss(c,dd,1))
#wight_sort4=mmd_loss(d,dd,1)/(mmd_loss(d,aa,1)*mmd_loss(d,bb,1)*mmd_loss(d,cc,1))

#wight_sort1=mmd_loss(a,aa,1)/(mmd_loss(aa,bb,1)*mmd_loss(aa,cc,1)*mmd_loss(aa,dd,1))
#wight_sort2=mmd_loss(b,bb,1)/(mmd_loss(bb,aa,1)*mmd_loss(bb,cc,1)*mmd_loss(bb,dd,1))
#wight_sort3=mmd_loss(c,cc,1)/(mmd_loss(cc,aa,1)*mmd_loss(cc,bb,1)*mmd_loss(cc,dd,1))
#wight_sort4=mmd_loss(d,dd,1)/(mmd_loss(dd,aa,1)*mmd_loss(dd,bb,1)*mmd_loss(dd,cc,1))


wight_sort1=mmd_loss(a,aa,1)#(mmd_loss(a,bb,1)*mmd_loss(a,cc,1)*mmd_loss(a,dd,1))/
wight_sort2=mmd_loss(b,bb,1)#(mmd_loss(b,aa,1)*mmd_loss(b,cc,1)*mmd_loss(b,dd,1))/
wight_sort3=mmd_loss(c,cc,1)#(mmd_loss(c,aa,1)*mmd_loss(c,bb,1)*mmd_loss(c,dd,1))/
wight_sort4=mmd_loss(d,dd,1)#(mmd_loss(d,aa,1)*mmd_loss(d,bb,1)*mmd_loss(d,cc,1))/
wight=distance+0.01*(0.2*wight_sort1+0.2*wight_sort2+0.2*wight_sort3+0.2*wight_sort4)#+0.6*mmd_loss(aa,bb,1)



den=Dense(4, activation='softmax')
model.add(den)

den2=Dense(4, activation='softmax')
model2.add(den2)
    #origin_output,cross_output=Dense(9, activation='softmax')([origin_seq,cross_seq])
origin_output=den(origin_seq)
cross_output=den2(cross_seq)

labels1 = tf.placeholder(tf.float32, shape=(None, 4))
labels2 = tf.placeholder(tf.float32, shape=(None, 4))

dis_loss=tf.constant(0.0)


#origin_output目标域输出，labels1目标域标签


#loss = tf.reduce_mean(1000000*tf.square(origin_output-labels1)+wight/100000*tf.square(cross_output-labels2)+100*tf.square(distance-dis_loss))
loss = tf.reduce_mean(tf.square(origin_output-labels1)+tf.square(cross_output-labels2)+0*0.1*tf.square(wight-dis_loss))#origin_output 表示目标域
loss_origin=tf.reduce_mean(tf.square(origin_output-labels1))
#loss=tf.reduce_mean(tf.square(origin_output-labels1))
loss_cross=tf.reduce_mean(tf.square(cross_output-labels2))
loss_deffrence=tf.reduce_mean(tf.square(distance-dis_loss))



train_step = tf.train.AdamOptimizer().minimize(loss)

init_op = tf.global_variables_initializer()

sess.run(init_op)


with sess.as_default():
    for i in range(100):
        step=0
            
        start=0

        end=start+batch_size
        while(end<1000):
        #train_step.run(feed_dict={origin_input:X_train, cross_input:X_test, labels1:Y_train, labels2:Y_test})
            a=np.random.randint(0,len(X_source1_train))
            b=np.random.randint(0,len(X_source2_train))
            c=np.random.randint(0,len(X_source3_train))
            d=np.random.randint(0,len(X_source4_train))
            
            
            a2=np.random.randint(0,len(X_tag1_train)-leng)
            b2=np.random.randint(0,len(X_tag2_train)-leng)
            c2=np.random.randint(0,len(X_tag3_train)-leng)
            d2=np.random.randint(0,len(X_tag4_train)-leng)
            
            #p=np.hstack((X_source1_train[a],X_source2_train[b],X_source3_train[c],X_source4_train[d]))
            #pp=p.T#np.reshape(p,(4,246),order='F')
            
            #ppp=np.expand_dims(pp, axis=2)
            #print(pp)
            '''
            _,loss1,loss2,loss3,loss4,ttt=sess.run([train_step,loss,loss_origin,loss_cross,loss_deffrence,wight],feed_dict={
                origin_input:np.vstack((np.vstack((X_source1_train[a:],X_source1_train[:a])),np.vstack((X_source2_train[b:],X_source2_train[:b])),np.vstack((X_source3_train[c:],X_source3_train[:c])),np.vstack((X_source4_train[d:],X_source4_train[:d])))), 
                cross_input:np.vstack((X_tag1_train[a2:a2+leng],X_tag2_train[b2:b2+leng],X_tag3_train[c2:c2+leng],X_tag4_train[d2:d2+leng])), 
                labels1:np.vstack((np.vstack((Y_source1_train[a:],Y_source1_train[:a])),np.vstack((Y_source2_train[b:],Y_source2_train[:b])),np.vstack((Y_source3_train[c:],Y_source3_train[:c])),np.vstack((Y_source4_train[d:],Y_source4_train[:d])))), 
                labels2:np.vstack((Y_tag1_train[a2:a2+leng],Y_tag2_train[b2:b2+leng],Y_tag3_train[c2:c2+leng],Y_tag4_train[d2:d2+leng]))})
            '''
            _,loss1,loss2,loss3,loss4,ttt=sess.run([train_step,loss,loss_origin,loss_cross,loss_deffrence,wight],feed_dict={
                origin_input:np.vstack((X_source1_train,X_source2_train,X_source3_train,X_source4_train)), 
                cross_input:np.vstack((X_tag1_train[a2:a2+leng],X_tag2_train[b2:b2+leng],X_tag3_train[c2:c2+leng],X_tag4_train[d2:d2+leng])), 
                labels1:np.vstack((Y_source1_train,Y_source2_train,Y_source3_train,Y_source4_train)), 
                labels2:np.vstack((Y_tag1_train[a2:a2+leng],Y_tag2_train[b2:b2+leng],Y_tag3_train[c2:c2+leng],Y_tag4_train[d2:d2+leng]))})
            #_,loss1,loss2,loss3,loss4,ttt=sess.run([train_step,loss,loss_origin,loss_cross,loss_deffrence,wight],feed_dict={origin_input:np.expand_dims(np.hstack((X_source1_train,X_source2_train,X_source3_train,X_source4_train)).T, axis=2), cross_input:np.expand_dims(np.hstack((X_tag1_train[a2],X_tag2_train[b2],X_tag3_train[c2],X_tag4_train[d2])).T, axis=2), labels1:np.vstack((Y_source1_train[a],Y_source2_train[b],Y_source3_train[c],Y_source4_train[d])).T, 

            #_,loss1=sess.run([train_step,loss_origin],feed_dict={origin_input:X_train[start:end],labels1:Y_train[start:end]})
            start+=batch_size

            end=start+batch_size
            if step%100==0:
                print(loss1,loss2,loss3,loss4)
                print('wight=')
                print(ttt)
            step+=1
            
            
    val_res=sess.run(origin_output,feed_dict={origin_input:X2})
    print(val_res)
        
        
        



#model=Model(inputs=[origin_input,cross_input],outputs=[origin_output,cross_output,tf.convert_to_tensor(distance)])


#a=baseline_model()
#a.fit([X_train,X_test],[Y_train,Y_test,0],epochs=1, batch_size=1)
# 训练分类器
#estimator = KerasClassifier(build_fn=baseline_model, epochs=1, batch_size=1, verbose=1)
#estimator.fit([X_train,X_test],[Y_train,Y_test])
 
# 卷积网络可视化
# def visual(model, data, num_layer=1):
#     # data:图像array数据
#     # layer:第n层的输出
#     layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
#     f1 = layer([data])[0]
#     print(f1.shape)
#     num = f1.shape[-1]
#     print(num)
#     plt.figure(figsize=(8, 8))
#     for i in range(num):
#         plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i+1)
#         plt.imshow(f1[:, :, i] * 255, cmap='gray')
#         plt.axis('off')
#     plt.show()

# 混淆矩阵定义
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks,('0%','3%','5%','8%','10%','12%','15%','18%','20%','25%'))
    #plt.yticks(tick_marks,('0%','3%','5%','8%','10%','12%','15%','18%','20%','25%'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.savefig('test_xx.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()


Y_onehot2=Y_source_onehot.argmax(axis=-1)
val_res=val_res.argmax(axis=-1)
conf_mat = confusion_matrix(y_true=Y_onehot2, y_pred=val_res)
plt.figure()
plot_confusion_matrix(conf_mat, range(np.max(Y_onehot2)+1))

# seed = 42
# np.random.seed(seed)
# kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
# result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
# print("Accuracy of cross validation, mean %.2f, std %.2f\n" % (result.mean(), result.std()))
'''  
# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))
 
# 将其模型转换为json
model_json = model.to_json()
with open(r"D:\model.json",'w')as json_file:
    json_file.write(model_json)# 权重不在json中,只保存网络结构
model.save_weights('model.h5')

# 加载模型用做预测
json_file = open(r"D:\model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
# 输出预测类别
predicted = loaded_model.predict(X)
predicted_label = loaded_model.predict_classes(X)
print("predicted label:\n " + str(predicted_label))
#显示混淆矩阵
plot_confuse(model, X_test, Y_test)
 
# 可视化卷积层
# visual(estimator.model, X_train, 1)
'''

