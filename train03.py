from tflearn import *
import deep_learning
import configparser
import tflearn

def main():
    train('AWS', 500)

def train(env, epoch_num):
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
    print 'data: ', conf['data_folder']
    #-----------------------------------------------------------------------------------------

    #----------------------------------------input layer------------------------------------------
    # read input  http://tflearn.org/data_utils/#build-hdf5-image-dataset
    X_train, Y_train, X_test, Y_test = deep_learning.data_prep(conf)
    print('data prep done.')
    #prepare input layer  http://tflearn.org/data_preprocessing/
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(mean=mean_colors,per_channel=True)
    # http://tflearn.org/layers/core/#input-data
    input_layer = input_data(shape=[None, image_size, image_size, 3], name = 'input_layer',
                             data_preprocessing=img_prep)
    #---------------------------------------------------------------------------------------------




    #-------------------------------train--------------------------------------------------------
    # network
    softmax = deep_learning.inception(input_layer, 2)
    # estimator layer
    network = tflearn.regression(softmax, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)  #if want to finetune give 'restore=False'
    # model  http://tflearn.org/models/dnn/
    model = tflearn.DNN(network, checkpoint_path='./models/model_googlenet',
                        max_checkpoints=1, tensorboard_verbose=0, tensorboard_dir="./logs")
    print('network is created, fit started.')
    model.fit(X_train, Y_train, n_epoch=epoch_num,  shuffle=True,
              show_metric=True, batch_size=128, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenet_ercis')
    print('fit done.')
    #---------------------------------------------------------------------------------------------


    model.save('./model/inception_ercis_last.model')




    '''
    model_file = os.path.join(model_path, "vgg16.tflearn")
    model.load(model_file, weights_only=True)

    # Start finetuning
    model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_epoch=False,
              snapshot_step=200, run_id='vgg-finetuning')

    model.save('your-task-model-retrained-by-vgg')





    model.fit(X, Y, n_epoch=500, shuffle=True,
              show_metric=True, batch_size=32, snapshot_step=500,
    snapshot_epoch=False, run_id='vgg_oxflowers17')
    '''

if __name__ == '__main__':
    main()


