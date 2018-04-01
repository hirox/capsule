""" stock price predictor """

import os

import pdb
import capsule
import loader
import debug
import numpy as np
import pandas as pd
import plotly
import cufflinks as cf
import GPy, GPyOpt
from tensorflow.python.keras.regularizers import l1_l2
import multiprocessing
from multiprocessing import Process, Value, Array

# 株価データから学習用データを作る
def make_train_dataset():
    global vals

    __loader = loader.Loader()
    __loader.load()

    vals = __loader.to_percentage_from_start()
    #vals = __loader.to_percentage_from_last()

    # 正規化(n_samples, n_features)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #vals = scaler.fit_transform(vals)

    __loader.save_data()

    df = pd.DataFrame(vals)
    df.plot(figsize=(15, 5)).get_figure().savefig("graph_orig.png")

# 学習用データをロードする
def load_train_dataset():
    global vals
    print('loading csv data...  ', end='', flush=True)
    vals = pd.read_csv("data.csv", index_col = 0).as_matrix()
    print('done')

def run_capsule(index, param, result):
    load_train_dataset()

    # copy param
    p = [0] * len(param)
    for i in range(len(param)):
        p[i] = param[i]

    c = capsule.Capsule()
    c.set_sequence_data(vals,
        test_ratio = 0.05,
        num_after = 5,
        sequence_length = int(p.pop(0))
        )
    c.create_model(
        hidden_neurons = int(p.pop(0)),
        dropout = p.pop(0),
        lstm_layers = int(p.pop(0))
        #kernel_regularizer = l1_l2(0.0001, 0.0001)
        )
    score = c.fit(
        patience = int(p.pop(0)),
        batch_size = int(p.pop(0))
        )
    c.draw_graph(index.value)

    for i in range(len(score)):
        result[i] = score[i]

if __name__ == '__main__':
    bounds = [
        {'name': 'sequence_length', 'type': 'continuous',  'domain': (50, 200)},
        {'name': 'hidden_neurons',  'type': 'continuous',  'domain': (50, 500)},
        {'name': 'dropout',         'type': 'continuous',  'domain': (0.0, 0.5)},
        {'name': 'lstm_layers',     'type': 'discrete',    'domain': (1, 2)},
        {'name': 'patience',        'type': 'discrete',    'domain': (60, 80, 100)},
        {'name': 'batch_size',      'type': 'discrete',    'domain': (25, 50, 75)}]
        
        #{'name': 'l1_drop',         'type': 'continuous',  'domain': (0.0, 0.1)},
        #{'name': 'l2_drop',         'type': 'continuous',  'domain': (0.0, 0.1)},

    index = Value('i', 0)

    make_train_dataset()

    def f(x):
        global index
        for i in range(len(bounds)):
            print("%15s : %.5f" % (bounds[i]['name'], x[0,i]))

        param = Array('d', x[0])
        result = Array('d', [100] * 5)
        p = Process(target = run_capsule, args = (index, param, result))
        p.start()
        p.join()

        index.value = index.value + 1

        print(result[4])
        return result[4]

    # ベイズ最適化
    opt = GPyOpt.methods.BayesianOptimization(f = f, domain = bounds)
    opt.run_optimization(max_iter = 50)
    print("optimized parameters: {0}".format(opt.x_opt))
    print("optimized loss: {0}".format(opt.fx_opt))

    pdb.set_trace()

