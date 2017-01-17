from tflearn import *
import deep_learning
import configparser
import tflearn
import os.path

def main():
    #train('AWS', 10, learning_rate = 1.0)
    #train('AWS', 5, learning_rate = 0.1)
    #train('AWS', 25, learning_rate = 0.01)
    train('AWS', 25, learning_rate = 0.001)
    #train('AWS', 100, learning_rate = 0.0001)

def train(env, epoch_num, learning_rate = 0.01, clean_start = False):
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
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs2")
    if os.path.isfile('inception_ercis.model') and not clean_start:
        print 'load model learning_rate: ' + str(learning_rate) 
        model.load('inception_ercis.model',weights_only=True)

    model.fit(X_train, Y_train, validation_set = (X_test, Y_test),n_epoch=epoch_num,  shuffle=True,
              show_metric=True, batch_size=128, snapshot_step=200, snapshot_epoch=False, run_id='inception_ercis')
    #---------------------------------------------------------------------------------------------

    model.save('inception_ercis.model')


if __name__ == '__main__':
    main()


