
'''Train a simple convnet on the MNIST dataset.
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py
Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D
from keras.utils import np_utils
import getData
import theano
from keras.optimizers import SGD


def CV_3D_2():
    theano.config.openmp = True
    batch_size = 128
    nb_classes = 12
    nb_epoch = 32

    # input image dimensions
    dimx = 51
    dimy = 61
    dimz = 23
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    testscore = 0.0
    testaccuracy = 0.0

    for i in range(0,9):
        print('CV round ' + str(i))
        model = Sequential()

        model.add(Convolution3D(nb_filters/2, nb_conv, nb_conv, nb_conv,
                                border_mode='same',
                                input_shape=(1, dimx, dimy, dimz), dim_ordering='tf'))

        # model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
        #                         border_mode='same',
        #                         input_shape=(dimx, dimy, dimz), dim_ordering='tf'))
        model.add(Activation('relu'))

        # model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))
        # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        # model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
        model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        model.add(Activation('relu'))

        # model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        # model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))
        #
        # # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.25))

        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # the data, shuffled and split between train and test sets
        (X_train, Y_train), (X_test, Y_test)= getData.load_dataCV_subject(i)

        X_train = X_train.reshape(X_train.shape[0],1,dimx,dimy,dimz)
        X_test = X_test.reshape(X_test.shape[0],1,dimx,dimy,dimz)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)

        model.compile(loss='categorical_crossentropy', optimizer='adamax')

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1)
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score in CV ' + str(i) + ':', score[0])
        print('Test accuracy in CV ' + str(i)  + ':', score[1])

        testscore = testscore + score[0]
        testaccuracy = testaccuracy + score[1]

    print('Average Test score:', testscore/9)
    print('Average Test accuracy:', testaccuracy/9)



def CV_3D_1():
    theano.config.openmp = True
    batch_size = 128
    nb_classes = 12
    nb_epoch = 32

    # input image dimensions
    dimx = 51
    dimy = 61
    dimz = 23
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    testscore = 0.0
    testaccuracy = 0.0

    for i in range(0,9):
        print('CV round ' + str(i))
        model = Sequential()

        model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv,
                                border_mode='same',
                                input_shape=(1, dimx, dimy, dimz), dim_ordering='tf'))

        # model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
        #                         border_mode='same',
        #                         input_shape=(dimx, dimy, dimz), dim_ordering='tf'))
        model.add(Activation('relu'))

        # model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))
        # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        # model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
        model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        model.add(Activation('relu'))

        # model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        # model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))
        #
        # # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.25))

        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # the data, shuffled and split between train and test sets
        (X_train, Y_train), (X_test, Y_test)= getData.load_dataCV_subject(i)

        X_train = X_train.reshape(X_train.shape[0],1,dimx,dimy,dimz)
        X_test = X_test.reshape(X_test.shape[0],1,dimx,dimy,dimz)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)

        model.compile(loss='categorical_crossentropy', optimizer='adamax')

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1)
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score in CV ' + str(i) + ':', score[0])
        print('Test accuracy in CV ' + str(i)  + ':', score[1])

        testscore = testscore + score[0]
        testaccuracy = testaccuracy + score[1]

    print('Average Test score:', testscore/9)
    print('Average Test accuracy:', testaccuracy/9)


def CV_2D_1():
    theano.config.openmp = True
    batch_size = 128
    nb_classes = 12
    nb_epoch = 48

    # input image dimensions
    dimx = 51
    dimy = 61
    dimz = 23
    # number of convolutional filters to use
    nb_filters = 16
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3
    testscore = 0.0
    testaccuracy = 0.0
        
    for i in range(0,9):
        print('CV round ' + str(i))
        model = Sequential()

        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv,
        #                         border_mode='valid',
        #                         input_shape=(1, dimx, dimy, dimz)))

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='same',
                                input_shape=(dimx, dimy, dimz), dim_ordering='tf'))
        model.add(Activation('relu'))

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        # model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))
        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        model.add(Activation('relu'))

        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # the data, shuffled and split between train and test sets
        (X_train, Y_train), (X_test, Y_test)= getData.load_dataCV_subject(i)

        # X_train = X_train.reshape(X_train.shape[0],1,dimx,dimy,dimz)
        # X_test = X_test.reshape(X_test.shape[0],1,dimx,dimy,dimz)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)

        model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1)
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score in CV ' + str(i) + ':', score[0])
        print('Test accuracy in CV ' + str(i)  + ':', score[1])

        testscore = testscore + score[0]
        testaccuracy = testaccuracy + score[1]

    print('Average Test score:', testscore/9)
    print('Average Test accuracy:', testaccuracy/9)

def CV_2D_onsub():
    theano.config.openmp = True
    batch_size = 128
    nb_classes = 12
    nb_epoch = 32

    # input image dimensions
    dimx = 51
    dimy = 61
    dimz = 23
    # number of convolutional filters to use
    nb_filters = 65
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    testscore = 0.0
    testaccuracy = 0.0


    for i in range(0,9):
        print('CV round ' + str(i))
        # the data, shuffled and split between train and test sets
        (X_train, Y_train), (X_test, Y_test)= getData.load_dataCV_subject(i)

        # X_train = X_train.reshape(X_train.shape[0],1,dimx,dimy,dimz)
        # X_test = X_test.reshape(X_test.shape[0],1,dimx,dimy,dimz)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)


        model = Sequential()

        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv,
        #                         border_mode='valid',
        #                         input_shape=(1, dimx, dimy, dimz)))

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=(dimx, dimy, dimz), dim_ordering='tf'))

        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))

        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv))

        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        # model.add(MaxPooling3D(pool_size=(nb_pool, nb_pool, nb_pool)))

        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score in CV ' + str(i) + ':', score[0])
        print('Test accuracy in CV ' + str(i)  + ':', score[1])

        testscore = testscore + score[0]
        testaccuracy = testaccuracy + score[1]

    print('Average Test score:', testscore/9)
    print('Average Test accuracy:', testaccuracy/9)


def CV_onsample(path="data-P2.mat", test_split=0.1, nb_test = 12):
    theano.config.openmp = True
    print('CV on samples...')
    batch_size = 32
    nb_classes = 12
    nb_epoch = 50

    # input image dimensions
    dimx = 51
    dimy = 61
    dimz = 23
    # number of convolutional filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 4

    X, Y = getData.load_dataAll(path)

    nb_data = X.shape[0]

    nb_CV = nb_data / nb_test

    X = X.astype('float32')
    Y = np_utils.to_categorical(Y, nb_classes)

    testscore = 0.0
    testaccuracy = 0.0


    for i in range(0, nb_CV):
        model = Sequential()

        # model.add(Convolution3D(nb_filters, nb_conv, nb_conv, nb_conv,
        #                         border_mode='valid',
        #                         input_shape=(1, dimx, dimy, dimz)))

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=(dimx, dimy, dimz), dim_ordering='tf'))

        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy', optimizer=sgd)

        if i == 0:
            # continue
            X_test = X[: nb_test]
            Y_test = Y[: nb_test]
            X_train = X[nb_test: ]
            Y_train = Y[nb_test: ]
        elif i == nb_CV - 1:
            X_test = X[i*nb_test: ]
            Y_test = Y[i*nb_test: ]
            X_train = X[: i*nb_test]
            Y_train = Y[: i*nb_test]
        else:
            X_test = X[i*nb_test: (i+1)*nb_test]
            Y_test = Y[i*nb_test: (i+1)*nb_test]
            X_train = np.concatenate((X[: i*nb_test], X[(i+1)*nb_test: ]))
            Y_train = np.concatenate((Y[: i*nb_test], Y[(i+1)*nb_test: ]))
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score in CV round ' + str(i) + ':', score[0])
        print('Test accuracy in CV round ' + str(i)  + ':', score[1])
        testscore = testscore + score[0]
        testaccuracy = testaccuracy + score[1]

    print('Average Test score:', testscore/nb_CV)
    print('Average Test accuracy:', testaccuracy/nb_CV)


CV_onsample("data-P3.mat")
# CV_onsub()

CV_2D_1()
