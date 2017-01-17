from tflearn import *
import deep_learning
import configparser
import tflearn
import os
import pickle

def main():
    test('AWS','inception_ercis.model')

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
    model = tflearn.DNN(network, checkpoint_path='model_googlenet',
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs")
    if os.path.isfile(model_file) and not clean_start:
        model.load(model_file)

    #model.fit(X_train, Y_train, validation_set = (X_test, Y_test),n_epoch=epoch_num,  shuffle=True,
    #          show_metric=True, batch_size=100, snapshot_step=100, snapshot_epoch=False, run_id='inception_ercis')
    #---------------------------------------------------------------------------------------------
    #model.save('inception.model')

    predict = model.predict(X_train)
    #print type(predict)
    #print len(predict)
    with open('train_pred.pik', 'w') as f:
        pickle.dump(predict, f)
    predict = model.predict(X_test)
    #print type(predict)
    #print len(predict)
    with open('test_pred.pik', 'w') as f:
        pickle.dump(predict, f)

if __name__ == '__main__':
    main()



