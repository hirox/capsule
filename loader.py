import os
import datetime
import pdb
import math
import re
import pandas as pd
import numpy as np

class Loader:
    def __init__(self):
        self.code_exists = {}
        self.date_to_index = {}
        self.index_to_code = {}
        self.vals = None
        self.days = 0

    def load(self):
        base_path = '../capsule_data/'

        date = datetime.date(2014, 1, 1)
        load_days = 366 * 4

        #date = datetime.date(2017, 9, 21)
        #load_days = 15

        one_day = datetime.timedelta(1)
        self.vals = np.zeros((9000, load_days))

        for _ in range(load_days):
            self.__push_if_exists(base_path + 'y', date)
            self.__push_if_exists(base_path + 'd', date)

            date += one_day
        
        print('')
        self.vals = self.vals[:,:self.days]
        self.__load_split_data(base_path + 'split.csv')
        self.__load_merge_data(base_path + 'merge.csv')
        self.__remove_not_exists()
        self.__fix_zero()
        self.__fix_irregular_data()
        #self.vals = self.vals[:1]
        #self.vals = self.vals[:10]
        self.vals = self.vals.transpose()   # code x day => day x code
        self.vals_orig = self.vals.copy()
    
    def __to_date_string(self, date):
        return '%02d%02d%02d' % (date.year - 2000, date.month, date.day)

    def __push_if_exists(self, base, date):
        date_string = self.__to_date_string(date)
        path = base + date_string + '.txt'
        if not os.path.exists(path):
            return
        
        print('.', end='', flush=True)
        df = pd.read_csv(path, header = None, delim_whitespace = True, encoding = 'sjis', skiprows = 1)
        #pdb.set_trace()
        #for val in df.iloc[:,0]:
        #    codes[val] = True
        #    assert(val >= 1000)
        codes, names, starts, highs, lows, ends, volumes = \
            list(df[0]), list(df[1]), list(df[2]), list(df[3]), \
            list(df[4]), list(df[5]), list(df[6])
        while len(codes) > 0:
            code, name, start, high, low, end, volume = \
                codes.pop(), names.pop(), starts.pop(), highs.pop(), \
                lows.pop(), ends.pop(), volumes.pop()
            # [TODO] 高値・安値・始値の情報も入れる
            val = (float(start) + float(high) + float(low) + float(end)) / 4
            self.vals[code - 1000, self.days] = val
            if val != 0:
                self.code_exists[code - 1000] = True

        self.date_to_index[date_string] = self.days
        self.days += 1
    
    def __load_split_data(self, path):
        # 株式分割データのロード
        # https://mxp1.monex.co.jp/mst/servlet/ITS/info/StockSplit
        print('load split data')
        df = pd.read_csv(path, header = None, encoding = 'sjis', skiprows = 1)

        codes, split_dates, rates = \
            list(df[2]), list(df[1]), list(df[5])
        while len(codes) > 0:
            code, split_date, rate = \
                codes.pop(), split_dates.pop(), rates.pop()

            date = datetime.datetime.strptime(split_date, '%Y/%m/%d')
            date_string = self.__to_date_string(date)
            if date_string in self.date_to_index:
                end_index = self.date_to_index[date_string]
                (before, after) = rate.split(':')
                rate_float = float(before) / float(after)

                for i in range(end_index):
                    self.vals[int(code) - 1000, i] *= rate_float

    def __load_merge_data(self, path):
        # 株式併合データのロード
        # https://kabu.com/investment/meigara/gensi.html
        print('load merge data')
        df = pd.read_csv(path, header = None, encoding = 'sjis', skiprows = 1)

        before_merge_dates, codes, rates = \
            list(df[4]), list(df[1]), list(df[3])
        while len(codes) > 0:
            before_merge_date, code, rate = \
                before_merge_dates.pop(), codes.pop(), rates.pop()

            date = datetime.datetime.strptime(before_merge_date, '%Y/%m/%d')
            date_string = self.__to_date_string(date)
            if date_string in self.date_to_index:
                #pdb.set_trace()

                m = re.search(r"([0-9\.]+)株→([0-9\.]+)株", rate)
                if m:
                    end_index = self.date_to_index[date_string] + 1
                    (before, after) = (m.group(1), m.group(2))
                    rate_float = float(before) / float(after)

                    for i in range(end_index):
                        self.vals[int(code) - 1000, i] *= rate_float

                #pdb.set_trace()

    def __remove_not_exists(self):
        print('remove not exist code')
        keys = sorted(self.code_exists.keys())
        _tmp = np.zeros((len(keys), self.days))
        for i in range(len(keys)):
            _tmp[i] = self.vals[keys[i]]
            self.index_to_code[i] = keys[i] + 1000
        self.vals = _tmp

    def __fix_zero(self):
        print('fix zero')
        for i in range(len(self.vals)): # code
            last = 0
            row = self.vals[i]
            for j in range(len(self.vals[i])): # day
                if row[j] == 0:
                    row[j] = last
                else:
                    if last == 0:
                        # index 0-j までの値を j の値に設定する
                        for k in range(j):
                            row[k] = row[j]
                    last = row[j]

    def __fix_irregular_data(self):
        print('fix irregular')
        for i in range(len(self.vals)): # code
            last = 0
            row = self.vals[i]
            for j in range(len(self.vals[i])): # day
                if last != 0 and abs(row[j] / last - 1.0) > 0.5: # 前日比50%以上の解離があるとき
                    print('%d %d %f %f' % (i, j, row[j], last))
                    row[j] = last
                else:
                    last = row[j]
        
    def to_percentage_from_last(self):
        print('abs to percent(last)')
        _last = self.vals[0]
        _tmp = np.zeros((len(self.vals) - 1, len(self.vals[0])))

        for j in range(len(self.vals) - 1):   # day
            _tmp[j] = (self.vals[j + 1] / _last - 1.0)
            _last = self.vals[j + 1]

        self.vals = _tmp
        return self.vals

    def to_percentage_from_start(self):
        print('abs to percent(start)')
        _start = self.vals[0]

        _tmp = np.zeros((len(self.vals) - 1, len(self.vals[0])))

        for j in range(len(self.vals) - 1):   # day
            _tmp[j] = (self.vals[j + 1] / _start - 1.0)

            #for x in range(len(_tmp[j])):
            #    if _tmp[j, x] > 100:
            #        print("day: %d code: %d val: %.2f" % (j, self.index_to_code[x], _tmp[j, x]))

        self.vals = _tmp
        return self.vals

    def save_data(self):
        df = pd.DataFrame(self.vals)
        df.to_csv("data.csv")

        df = pd.DataFrame(self.date_to_index, index = [""])
        df.to_csv("date_to_index.csv")

        df = pd.DataFrame(self.index_to_code, index = [""])
        df.to_csv("index_to_code.csv")
