from tflearn import *
import deep_learning
import configparser
import tflearn
import os.path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import datetime
def gettime():
    now = datetime.datetime.now().time()
    return str(now.hour) + '_' + str(now.minute)


model_name = 'inception_ercis.model'
def main():
    model_name = 'inception_ercis.model'
    counter = gettime()
    '''
    print '******** counter: ', counter
    model_name = train('AWS', 5, learning_rate = 1.0)
    test('AWS', model_name)
 
    print '******** counter: ', counter
    model_name = train('AWS', 25, learning_rate = 0.01, model_name = model_name)
    '''
    test('AWS', model_name)
    print '******** counter: ', counter
    '''
    train('AWS', 25, learning_rate = 0.1, model_name = model_name)
    test('AWS', model_name)
    print '******** counter: ', counter
    model_name = train('AWS', 25, learning_rate = 0.001, model_name = model_name)
    test('AWS', model_name)
    print '******** counter: ', counter
    model_name = train('AWS', 20, learning_rate = 0.01, model_name = model_name)
    test('AWS', model_name)
    print '******** counter: ', counter
    model_name = train('AWS', 25, learning_rate = 0.001, model_name = model_name)
    test('AWS', model_name)
    print '******** counter: ', counter
    model_name = train('AWS', 25, learning_rate = 0.001, model_name = model_name)
    test('AWS', model_name)
    #print '******** counter: ', counter
    #model_name = train('AWS', 10, learning_rate = 0.0001, model_name = model_name)
    #test('AWS', model_name)
    #print '******** counter: ', counter
    #model_name = train('AWS', 10, learning_rate = 0.001, model_name = model_name)
    #test('AWS', model_name)
    '''

def train(env, epoch_num, learning_rate = 0.01, clean_start = False, model_name = None):
    '''
    a function that create input, create network, train network and report results
    :param env: local/AWS; environment that our system work on
    :param epoch_num: number of epochs to run
    :return:
    '''

    #----------------------reading constants from config file---------------------------------
    config = configparser.ConfigParser()
    config.read('config.ini')
    conf = config[env]
    image_size = conf['window_size']
    mean_colors = [float(conf['mean_r']), float(conf['mean_g']), float(conf['mean_b'])]
    #-----------------------------------------------------------------------------------------


    #----------------------------------------input layer------------------------------------------
    # read input  http://tflearn.org/data_utils/#build-hdf5-image-dataset
    X_train, Y_train, X_test, Y_test = deep_learning.data_prep(conf, clean_start = clean_start)
    #prepare input layer  http://tflearn.org/data_preprocessing/
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=mean_colors,per_channel=True)
    # http://tflearn.org/layers/core/#input-data
    input_layer = input_data(shape=[None, image_size, image_size, 3], name = 'input_layer')
                             #,data_preprocessing=img_prep)
    #---------------------------------------------------------------------------------------------




    #-------------------------------create model--------------------------------------------------------
    # network
    softmax = deep_learning.inception(input_layer, 2)
    #softmax = deep_learning.VGGNet(input_layer, 2)
    # estimator layer
    f_score = tflearn.metrics.F2Score()
    network = tflearn.regression(softmax, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate, metric=f_score)  #if want to finetune give 'restore=False'
    # model  http://tflearn.org/models/dnn/
    model = tflearn.DNN(network, checkpoint_path='model_inception',
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs")
    if model_name != None:
        if os.path.isfile(model_name) and not clean_start:
            print 'load model learning_rate: ' + str(learning_rate) 
            model.load(model_name,weights_only=True)

    model.fit(X_train, Y_train, validation_set = (X_test, Y_test),n_epoch=epoch_num,  shuffle=True,
              show_metric=True, batch_size=128, snapshot_step=500, snapshot_epoch=False, run_id='inception_ercis_v031')
    #---------------------------------------------------------------------------------------------

    model.save('inception_ercis.model')
    return 'inception_ercis.model'

def test(env,  model_file, clean_start = False):
    '''
    a function that create input, create network, train network and report results
    :param env: local/AWS; environment that our system work on
    :param epoch_num: number of epochs to run
    :return:
    '''
    #----------------------reading constants from config file---------------------------------
    config = configparser.ConfigParser()
    config.read('config.ini')
    conf = config[env]
    image_size = conf['window_size']
    mean_colors = [float(conf['mean_r']), float(conf['mean_g']), float(conf['mean_b'])]
    #-----------------------------------------------------------------------------------------

    #----------------------------------------input layer------------------------------------------
    # read input  http://tflearn.org/data_utils/#build-hdf5-image-dataset
    X_train, Y_train, X_test, Y_test = deep_learning.data_prep_(conf)
    #prepare input layer  http://tflearn.org/data_preprocessing/
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=mean_colors,per_channel=True)
    # http://tflearn.org/layers/core/#input-data
    input_layer = input_data(shape=[None, image_size, image_size, 3], name = 'input_layer')
                             #,data_preprocessing=img_prep)
    #---------------------------------------------------------------------------------------------


    #-------------------------------create model--------------------------------------------------------
    # network
    softmax = deep_learning.inception(input_layer, 2)
    #softmax = deep_learning.VGGNet(input_layer, 2)
    # estimator layer
    f_score = tflearn.metrics.F2Score()
    network = tflearn.regression(softmax, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001, metric=f_score)  #if want to finetune give 'restore=False'
    # model  http://tflearn.org/models/dnn/
    model = tflearn.DNN(network, checkpoint_path='model_inception',
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs_test")
    if os.path.isfile(model_file) and not clean_start:
        model.load(model_file)

    #model.fit(X_train, Y_train, validation_set = (X_test, Y_test),n_epoch=epoch_num,  shuffle=True,
    #          show_metric=True, batch_size=100, snapshot_step=100, snapshot_epoch=False, run_id='inception_ercis')
    #---------------------------------------------------------------------------------------------
    #model.save('inception.model')

    predict = model.predict(X_test)
    with open('results/test_pred_{0}.pik'.format(gettime()), 'w') as f:
        pickle.dump(predict, f)
    
    target = np.argmax(Y_test[()], axis=1)
    pred = np.argmax(predict, axis=1)
    #print Y_test.shape
    #for x,y in zip(target, pred):
    #    print x, y
    #print type(target)
    #print target.shape
    #print type(pred)
    #print pred.shape
    #for y in Y_test:
    #    print y
    #try reporting
    #try:
    print 'accuracy_score: ', accuracy_score(target, pred)
    print 'recall_score: ', recall_score(target, pred)
    print 'precision_score: ', precision_score(target, pred)
    print 'f2_score: ', fbeta_score(target, pred, 2)
    print 'confusion_matrix: '
    print confusion_matrix(target, pred)
    with open('results/results.txt', 'a+') as f:
        f.write('# ' + gettime() + '\n')
        f.write('accuracy_score: {0}\n'.format(accuracy_score(target, pred)))
        f.write('recall_score: {0}\n'.format(recall_score(target, pred)))
        f.write('precision_score: {0}\n'.format(precision_score(target, pred)))
        f.write('f2_score: {0}\n'.format(fbeta_score(target, pred, 2)))
        f.write('confusion_matrix: \n')
        f.write(str(confusion_matrix(target, pred)))
        f.write('\n\n\n')
    #except:
        #print 'there was some error'
        #pass

if __name__ == '__main__':
    main()


