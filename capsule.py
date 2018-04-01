""" Capsule class """

import pandas as pd
import numpy as np
import tensorflow as tf
import math
import pdb
import os
import gc
import random
from tensorflow.python import keras
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, LSTM, Dropout, BatchNormalization
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error

class Capsule:
    def __init__(self, fix_seed = True):
        print('Tensorflow: {0}'.format(tf.__version__))

        self.create_new_session(fix_seed = fix_seed)

        # インスタンス変数の初期化
        self.model = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.sequence_length = None
        self.in_out_neurons = None
        self.test_ratio = None
        self.y_index = None
    
    def create_new_session(self, fix_seed = True):
        # 初期設定

        allow_growth = True
        if fix_seed:
            # 毎回結果が同じになるように seed を固定する
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(42)
            random.seed(12345)

            # 毎回結果が同じになるようにスレッド数を 1 に制限する
            config = tf.ConfigProto(
                intra_op_parallelism_threads = 1,
                inter_op_parallelism_threads = 1,
                gpu_options = tf.GPUOptions(allow_growth = allow_growth)
            )
        else:
            config = tf.ConfigProto(
                gpu_options = tf.GPUOptions(allow_growth = allow_growth)
            )

        keras.backend.set_session(tf.Session(config = config))

        if fix_seed:
            # 毎回結果が同じになるように seed を固定する
            tf.set_random_seed(1234)

    def reset_resource(self):
        del self.model
        keras.backend.clear_session()
        keras.backend.get_session().close()
        gc.collect()
        self.create_new_session()

    def create_model(self, hidden_neurons = 200, dropout = 0.2, lstm_layers = 2, kernel_regularizer = None):
        self.model = Sequential()

        for i in range(lstm_layers):
            if i == 0:
                self.model.add(BatchNormalization(input_shape=(self.sequence_length, self.in_out_neurons)))
            else:
                self.model.add(BatchNormalization())

            self.model.add(LSTM(hidden_neurons, dropout = dropout,
                kernel_regularizer = kernel_regularizer,
                return_sequences = (True if i != lstm_layers - 1 else False)))

        #self.model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l1_l2(0.01, 0.01), return_sequences=True))

        self.model.add(Dense(1))
        self.model.add(Activation('linear'))
        self.model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())
        print(self.model.summary())

    def fit(self, patience = 20, batch_size = 50, y_index = 0):
        print('X_train Shape: {0}, X_test Shape: {1}'.format(self.X_train.shape, self.X_test.shape))

        self.y_index = y_index
        self.model.fit(self.X_train, self.Y_train[:,y_index], batch_size=batch_size, epochs=1000, validation_split=0.01, shuffle=True,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=patience),
                TensorBoard(
                    log_dir="/tmp/keras-lstm", histogram_freq=1,
                    write_graph=True, write_images=True
                )
            ])

        return self.__calc_score_last()

    def __calc_score(self, x_train, x_test):
        trainBasis = math.sqrt(mean_squared_error(self.Y_train[:,self.y_index], x_train))
        trainScore = math.sqrt(mean_squared_error(self.Y_train[:,self.y_index], self.model.predict(self.X_train)))

        testBasis = math.sqrt(mean_squared_error(self.Y_test[:,self.y_index], x_test))
        testScore = math.sqrt(mean_squared_error(self.Y_test[:,self.y_index], self.model.predict(self.X_test)))

        evaluate = self.model.evaluate(self.X_test, self.Y_test[:,self.y_index])
        return (trainBasis, trainScore, testBasis, testScore, evaluate)

    def __print_score(self, score):
        print('Train Basis : %.5f MSE' % score[0])
        print('Train Score : %.5f MSE' % score[1])
        print('Test Basis  : %.5f MSE' % score[2])
        print('Test Score  : %.5f MSE' % score[3])
        print('Evaluate    : %.5f MSE' % score[4])

    def __calc_score_last(self):
        # all test sequence, last day, code 0
        score = self.__calc_score(self.X_train[:,-1,self.y_index], self.X_test[:,-1,self.y_index])
        self.__print_score(score)
        return score

    def __calc_score_zero(self):
        score = self.__calc_score(np.zeros(len(self.Y_train)), np.zeros(len(self.Y_test)))
        self.__print_score(score)
        return score

    def draw_graph(self, file_name_index):
        pd.DataFrame({
            'actual': self.Y_train[:,self.y_index].flatten(),
            'basis': self.X_train[:,-1,self.y_index].flatten(),
            'predict': self.model.predict(self.X_train).flatten()
            }).plot(figsize=(15, 5)).get_figure().savefig("./graphs/%04d_graph_train.png" % file_name_index)

        (actual, basis, predict) = self.predict()

        pd.DataFrame({
            'predict': predict.flatten(),
            'basis': basis.flatten(),
            'actual': actual.flatten()
            }).plot(figsize=(15, 5)).get_figure().savefig("./graphs/%04d_graph_predict.png" % file_name_index)


    def predict(self):
        return (self.Y_test[:,0], self.X_test[:,-1,0], self.model.predict(self.X_test))

    def set_sequence_data(self, ndarray, sequence_length = 100, num_after = 1, test_ratio = 0.1):
        # day x code -> sample x code
        (x, y) = self.__generate_xy_data_from_x(ndarray, sequence_length, num_after)
        self.set_xy_data(x, y, test_ratio)

    def set_xy_data(self, xndarray, yndarray, test_ratio = 0.1):
        assert(len(xndarray) == len(yndarray))

        self.__set_internal(xndarray, test_ratio)

        (self.X_train, self.X_test) = self.__train_test_split(xndarray)
        (self.Y_train, self.Y_test) = self.__train_test_split(yndarray)

    def __set_internal(self, ndarray, test_ratio):
        # 1次元の場合は2次元に変換する
        if len(ndarray.shape) == 1:
            ndarray = ndarray.reshape(ndarray.shape[0], 1)

        self.sequence_length = ndarray.shape[1]
        self.in_out_neurons = ndarray.shape[2]
        self.test_ratio = test_ratio

        print('Shape: {0}'.format(ndarray.shape))
        print('Seq. : {0}'.format(self.sequence_length))
        print('Dim. : {0}'.format(self.in_out_neurons))
        #print('Test Ratio: {0}'.format(self.test_ratio))

    def __train_test_split(self, ndarray):
        # 学習データ数(デフォルトは 90% のデータ)
        num_trains = int(round(len(ndarray) * (1 - self.test_ratio)))

        # 最初の 90% のデータで学習して残りの 10% をテストデータとする
        return (ndarray[0:num_trains], ndarray[num_trains:])

    # sequence_length の長さと次の値を X, Y に設定する
    def __generate_xy_data_from_x(self, ndarray, sequence_length, num_after = 1):
        X, Y = [], []
        for i in range(len(ndarray) - sequence_length - (num_after - 1)):
            X.append(ndarray[i:i + sequence_length])
            Y.append(ndarray[i + sequence_length + (num_after - 1)])
        np_X = np.array(X)
        np_Y = np.array(Y)

        return np_X, np_Y
