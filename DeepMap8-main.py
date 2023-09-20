import numpy as np
import time, math
import scipy.io as sci
import os
import graph_canonicalization as gc
from scipy.sparse import load_npz

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Multiply, Add,Input, Flatten, Dense, Dropout, Conv1D,Lambda, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Activation, Concatenate
from keras import regularizers
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import  RMSprop
from sklearn.model_selection import  GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#from keras.utils import multi_gpu_model
from multiprocessing import Pool
#from keras.utils import Sequence
import keras

# config = ConfigProto()
# sess = tf.Session(config=config)

#config = tf.ConfigProto(device_count={'CPU': 20})




#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#keras.backend.tensorflow_backend.set_session(sess)



class MyLayer1(keras.engine.base_layer.Layer):#特征变换

    def __init__(self,output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer1, self).__init__(**kwargs)

    def build(self, input_shape):

        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        #super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return tf.matmul(x, self.kernel)
    #
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[1], input_shape[2])
    def get_config(self):
        config = super(MyLayer1, self).get_config()

        return config


def channel_attention2(input_feature,ratio=0.25):#通道注意力拼接迭代
    channel = int(input_feature.shape[-1])

    shared_layer_one = Dense(int(channel // ratio),
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam = Add()([avg_pool, max_pool])
    cbam = Activation('sigmoid')(cbam)
    return cbam
    #return Multiply()([input_feature, cbam])


def get_model(filter_size, num_instance, feature_size, num_class,qianru_size,max_h):
    #最新 不ronghe
    input1 = Input((num_instance * filter_size, feature_size))

    x = channel_attention2(input1,5)


    flow=Multiply()([input1,x])
    #flow = MyLayer4()(y)
    #flow=ronghe(flow,qianru_size,max_h)
    #flow=Add()([flow[:,:,0:40], flow[:,:,40:80], flow[:,:,80:120], flow[:,:,120:160], flow[:,:,160:200]])
    #flow=channel_attention2(Input(filter_size*num_instance, feature_size),5)(y)
    flow=Conv1D(filters=32, kernel_size=filter_size, strides=filter_size, padding='same', use_bias=True,activation="relu")(flow)
    flow=Conv1D(filters=16, kernel_size=8, strides=1, padding='same', use_bias=True,activation="relu")(flow)
    #model1.add(Conv1D(filters=8, kernel_size=1, strides=1, padding='same', use_bias=True,activation="relu"))
    flow=Lambda(lambda x: K.sum(x, axis=1))(flow)
    output1=Dense(64, activation='relu', use_bias=True,kernel_regularizer=regularizers.l1_l2(0.001,0.001))(flow)
    #model1.build(input_shape=(None, num_instance * filter_size, feature_size))

    input2 = Input((num_instance , feature_size))
    flow = Multiply()([input2, x])
    #flow=Add()([flow[:,:,0:40], flow[:,:,40:80], flow[:,:,80:120], flow[:,:,120:160], flow[:,:,160:200]])
    #flow = ronghe(flow, qianru_size, max_h)
    flow=MyLayer1(64)(flow)
    output2=Lambda(lambda x: K.sum(x, axis=1))(flow)
    #model2.build(input_shape=(None, num_instance, feature_size))



    #flow = channel_attention(output1,output2,ratio=5)
    flow=keras.layers.concatenate([output1, output2], axis=-1)
    #flow=Dense(128, activation='relu', use_bias=True)(flow)
    flow = Dropout(0.5)(flow)
    predictions = Dense(num_class, activation='softmax')(flow)

    model = Model(inputs=[input1,input2], outputs=predictions)
    #model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    rmsprop=RMSprop(learning_rate=0.001)
    model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
    return model


def get_callbacks(patience_lr):
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=patience_lr, verbose=1, min_lr=0,
                                       mode='auto')

    return [reduce_lr_loss]

#def train_model(X, feature, y,  num_sample, feature_size, num_class, batch_size, EPOCHS,filter_size):
def train_model(X,feature , y, train_idx, test_idx,qianru_size,max_h, num_sample, feature_size, num_class, batch_size, EPOCHS, filter_size):
    # print(type(X))
    # print(X)
    # print(type(feature))
    # print(feature)
    X_val = np.array([X[id] for id in test_idx])
    feature_val = np.array([feature[id] for id in test_idx])
    y_val = y[test_idx, :]

    callbacks = get_callbacks(patience_lr=5)


    model = get_model(filter_size, num_sample, feature_size, num_class,qianru_size,max_h,)
    model.summary()
    

    x=np.array([X[id] for id in train_idx])
    features=np.array([feature[id] for id in train_idx])
    y=y[train_idx, :]

    result = model.fit([x, features],y, batch_size=batch_size,
                        epochs=EPOCHS, verbose=1, callbacks=callbacks,
                        validation_data=([X_val,feature_val], y_val), shuffle=True)

    val_acc = result.history['val_accuracy']
    acc = result.history['accuracy']


    return val_acc, acc

fname="jieguo_3080ti.txt"

if __name__ == "__main__":
    # location to save the results
    OUTPUT_DIR = "results/"
    # location of the datasets
    DATA_DIR = "datasets/"


    dataset = [ "ENZYMES",'NCI109',"PROTEINS",'IMDB-BINARY',  'IMDB-MULTI']
    hasnodelabel = [ 1,1,1,0,0]
    hasnodeattribute = [0, 0, 0, 0, 0]

    filter_size = 3
    kfolds = 10
    batch_size = 32
    EPOCHS = 100

    graphlet_size = 5
    max_h = 5

    feature_type = 3  # 1 (graphlet), 2 (SP), 3 (WL)

    for i in range(len(dataset)):
    #for i in [3]:
        
        

        ds_name = dataset[i]  # dataset name
        filename = DATA_DIR + ds_name + '.mat'
        data = sci.loadmat(filename)
        graph_data = data['graph']
        graph_labels = data['label'].T[0]
        num_graphs = len(graph_data[0])
        zybs=0
        for q in graph_labels:
            if q==1:
                zybs=zybs+1
        print("正样本数"+str(zybs))

        num_class = len(np.unique(graph_labels))

        print("Dataset: %s\n" % (ds_name))
        print('num_graphs:',num_graphs)

        hasnl = hasnodelabel[i]
        hasatt = hasnodeattribute[i]
        val_acc = np.zeros((kfolds, EPOCHS))
        acc = np.zeros((kfolds, EPOCHS))
        
        f = open(fname, 'a+')
        f.write( "\n"+ds_name+":\n")
        f.write( "feature_size：5*")
        f.close()


        #for max_h in [5]:
        #for z in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        #for qianru_size in [20,40,60,80,100]:
        # for filter_size in [3]:
        #
        #     qianru_size = 40


        max_h = 5
        qianru_size = 40
        filter_size = 3
        if ds_name == 'ENZYMES':
            qianru_size = 100
        if ds_name == 'IMDB-BINARY':
            max_h = 3

        z=0.5
        f = open(fname, 'a+')
        f.write(str(qianru_size) + "：")
        f.close()
        start = time.time()
        X, feature_size, num_sample, feature = gc.canonicalization(z, ds_name, graph_data, hasnl, filter_size,
                                                                   feature_type, graphlet_size, max_h, qianru_size)


        folds = list(
            StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=7).split(np.zeros(num_graphs), graph_labels))
        print('len(folds)', len(folds))
        # print('feature_size',feature_size)

        encoder = LabelEncoder()
        encoder.fit(graph_labels)
        encoded_Y = encoder.transform(graph_labels)
        y = np_utils.to_categorical(encoded_Y)

        # y:one-hot encoding

        # grid_result=train_model(X, feature, y,  num_sample, feature_size, num_class, batch_size, EPOCHS, filter_size)
        # print("best:"+str(grid_result.best_score_)+"     "+str(grid_result.best_params_))
        # f = open(fname, 'a+')
        # f.write("best:"+str(grid_result.best_score_)+"     "+str(grid_result.best_params_))
        # f.close()
        for j, (train_idx, test_idx) in enumerate(folds):
            print('\nFold ', j)
            print('train:', len(train_idx), ' test:', len(test_idx))
            print('最大节点数:', num_sample)

            scores_val_acc, scores_acc = train_model(X, feature, y, train_idx, test_idx, qianru_size, max_h, num_sample,
                                                     feature_size, num_class, batch_size, EPOCHS, filter_size)

            val_acc[j, :] = scores_val_acc
            acc[j, :] = scores_acc

        val_acc_mean = np.mean(val_acc, axis=0) * 100
        val_acc_std = np.std(val_acc, axis=0) * 100
        best_epoch = np.argmax(val_acc_mean)
        Average_Accuracy = "%.2f%% (+/- %.2f%%)" % (val_acc_mean[best_epoch], val_acc_std[best_epoch])

        end = time.time()
        print("Average Accuracy: ")
        print(Average_Accuracy)
        f = open(fname, 'a+')
        f.write(Average_Accuracy)
        f.write("  ")
        f.close()
        f = open(fname, 'a+')
        f.write("eclipsed time: %g" % (end - start))
        f.write("  ")
        f.close()
        print("eclipsed time: %g" % (end - start))
