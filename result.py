import pickle
from dataset_loader import DataSetLoader
import numpy as np
from numpy import dtype

def print_nb_2(ml_name):
    print '*********** {} ***************'.format(ml_name)
    nb = pickle.load(open('result2/{}.obj'.format(ml_name), 'rb'))
    for d_name in DataSetLoader.dataset_name:
        lst_nb = nb[d_name]
        data_lst = np.array(lst_nb)
        print '*************** ', d_name
        print 'acc 75 : ', np.mean(data_lst[:, 0]), 'fsc 75 : ', np.mean(data_lst[:, 1])
        print 'acc 50 : ', np.mean(data_lst[:, 4]), 'fsc 50 : ', np.mean(data_lst[:, 5])
        print 'acc 25 : ', np.mean(data_lst[:, 10]), 'fsc 25 : ', np.mean(data_lst[:, 11])
        
def print_result(ml_name):
#     nb = pickle.load(open('result/decsiontree.obj', 'rb'))
#     svm_lst = pickle.load(open('result/svm_austra.obj', 'rb'))
#     print svm_lst['austra'][0]
#     print '*********** {} ***************'.format(ml_name)
    for d_name in DataSetLoader.dataset_name:
        nb = pickle.load(open('result2/{}_{}.obj'.format(ml_name, d_name), 'rb'))
        lst_nb = nb[d_name]
        data_lst = np.array(lst_nb)
#         print '*************** ',d_name
        if ml_name == 'svm':
            print np.mean(np.array(data_lst[:, 0], dtype='float')), ',', np.mean(np.array(data_lst[:, 1], dtype='float'))
            print np.mean(np.array(data_lst[:, 5], dtype='float')), ',', np.mean(np.array(data_lst[:, 6], dtype='float'))
            print np.mean(np.array(data_lst[:, 10], dtype='float')), ',', np.mean(np.array(data_lst[:, 11], dtype='float'))      
        else:
            print np.mean(data_lst[:, 0]), ',', np.mean(data_lst[:, 1])
            print np.mean(data_lst[:, 5]), ',', np.mean(data_lst[:, 6])
            print np.mean(data_lst[:, 10]), ',', np.mean(data_lst[:, 11])
            
def print_nb(ml_name):
#     print '*********** {} ***************'.format(ml_name)
    nb = pickle.load(open('result4/{}.obj'.format(ml_name), 'rb'))
    for d_name in DataSetLoader.dataset_name:
        lst_nb = nb[d_name]
        data_lst = np.array(lst_nb)
#         print '0 ', data_lst[:, 0]
#         print '1 ', data_lst[:, 1]
#         print '2 ', data_lst[:, 2]
#         print '3 ', data_lst[:, 3]
#         print '4 ', data_lst[:, 4]
#         print '5 ', data_lst[:, 5]
#         print '6 ', data_lst[:, 6]
#         print '7 ', data_lst[:, 7]
#         print '8 ', data_lst[:, 8]
#         print '9 ', data_lst[:, 9]
#         print '10 ', data_lst[:, 10]
#         print '11 ', data_lst[:, 11]     
#         break
        print np.mean(data_lst[:, 0]), ',', np.mean(data_lst[:, 1])
        print np.mean(data_lst[:, 5]), ',', np.mean(data_lst[:, 6])
        print np.mean(data_lst[:, 10]), ',', np.mean(data_lst[:, 11])

def print_missing(ml):
    import cmp
    data_set_name = ['heart', 'letter', 'austra', 'german', 'sat', 'segment', 'vehicle']
    for d_name in data_set_name:
        file_name = 'all_result/missing/{}_{}.obj'.format(ml, d_name)
        obj = pickle.load(open(file_name, 'rb'))
        data_lst = np.array(obj[d_name])
#         print np.mean(data_lst[:, 0]), ',', np.mean(data_lst[:, 1]), ',', np.mean(data_lst[:, 5]), ',', np.mean(data_lst[:, 6]), ',', np.mean(data_lst[:, 10]), ',', np.mean(data_lst[:, 11])
        print np.mean(np.array(data_lst[:, 0], dtype='float')), ',', np.mean(np.array(data_lst[:, 1], dtype='float')),',',np.mean(np.array(data_lst[:, 5], dtype='float')), ',', np.mean(np.array(data_lst[:, 6], dtype='float')),',', np.mean(np.array(data_lst[:, 10], dtype='float')), ',', np.mean(np.array(data_lst[:, 11], dtype='float'))          

        
if __name__ == '__main__':
#     file_name = 'all_result/missing/{}_{}.obj'.format('bagging', 'heart')
#     obj = pickle.load(open(file_name, 'rb'))
#     obj_arr = np.array(obj['heart'])
#     print np.mean(obj_arr[:,0])
    
    print_missing('svm')
    
#     print_result('bagging')
#     print_result('boosted')
#     print_result('randomforest')
#     print_result('svm')
#     print_result('knn')
#     print_nb('nb')
#     print_nb('decsiontree')
