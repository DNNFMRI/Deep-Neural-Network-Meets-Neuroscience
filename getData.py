from scipy.io import loadmat
import numpy as np

def load_data(path="data-P1.mat", test_split=0.1, seed= 42, nb_test = 3):

    data = loadmat(path)

    Xorigin = data['XSeq']

    Yorigin = data['YSeq']

    nb_data = 360
    dimx = 51
    dimy = 61
    dimz = 23

    X = np.empty([nb_data,dimx,dimy,dimz])

    labels = np.empty([nb_data],dtype = int)

    for idx in range(0,nb_data):
        X[idx] = Xorigin[idx][0].copy()
        labels[idx] = Yorigin[idx][0][0][0] - 2

    Xorigin = None
    Yorigin = None

    # X = X.reshape(nb_data,1,dimx,dimy,dimz)

    # np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(labels)

    # X_train = X[:int(len(X) * (1 - test_split))]
    # y_train = labels[:int(len(X) * (1 - test_split))]
    #
    # X_test = X[int(len(X) * (1 - test_split)):]
    # y_test = labels[int(len(X) * (1 - test_split)):]

    X_train = X[: nb_data-nb_test]
    y_train = labels[:nb_data-nb_test]

    X_test = X[nb_data-nb_test:]
    y_test = labels[nb_data-nb_test:]

    return (X_train, y_train), (X_test, y_test)

def load_dataCV_subject(idx_CV=0):
    nb_data = 360
    dimx = 51
    dimy = 61
    dimz = 23

    X_train = np.empty([nb_data*8,dimx,dimy,dimz])
    y_train = np.empty([nb_data*8],dtype = int)

    lastIdx = 0
    for i in range(0,9):
        if i == idx_CV:
            continue
        path="data-P" + str(i+1) + ".mat"
        data = loadmat(path)

        Xorigin = data['XSeq']

        Yorigin = data['YSeq']


        for idx in range(0, nb_data):
            X_train[idx + lastIdx] = Xorigin[idx][0].copy()
            y_train[idx + lastIdx] = Yorigin[idx][0][0][0] - 2

        lastIdx = lastIdx + idx + 1


    path="data-P" + str(idx_CV+1) + ".mat"
    data = loadmat(path)

    Xorigin = data['XSeq']

    Yorigin = data['YSeq']

    X_test = np.empty([nb_data,dimx,dimy,dimz])
    Y_test = np.empty([nb_data],dtype = int)


    for idx in range(0, nb_data):
        X_test[idx] = Xorigin[idx][0].copy()
        Y_test[idx] = Yorigin[idx][0][0][0] - 2




    return (X_train, y_train), (X_test, Y_test)

load_dataCV_subject(1)