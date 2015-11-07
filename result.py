import pickle
from dataset_loader import DataSetLoader
import numpy as np

def print_result():
    nb = pickle.load(open('result/decsiontree.obj', 'rb'))
    for d_name in DataSetLoader.dataset_name:
        lst_nb = nb[d_name]
        data_lst = np.array(lst_nb)
        print 'data set name ', d_name
        print 'acc 75 : ', np.mean(data_lst[:, 0]), 'fsc 75 : ', np.mean(data_lst[:, 1])
        print 'acc 50 : ', np.mean(data_lst[:, 3]), 'fsc 50 : ', np.mean(data_lst[:, 4])
        print 'acc 25 : ', np.mean(data_lst[:, 6]), 'fsc 25 : ', np.mean(data_lst[:, 7])

if __name__ == '__main__':
    print_result()
