#coding = utf-8
import codecs
import re
import nltk
from nltk.corpus import stopwords
import cPickle
import numpy as np
import time
from sklearn import mixture
import math
import random
import matplotlib
import matplotlib.pyplot as plt


vecdim = 50
max = 21
Iteration = 500
K = 50
learn_rate = 0.05
lowe_learn_rate = 0.0001
NEG_number = 10
tol=1e-3

def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item]

def init():
    file_r = codecs.open('05position_string.txt', 'r', 'utf-8')
    lines = file_r.readlines()
    file_r.close()
    wordcount = 1
    sindex = 0
    senindex = {}
    wordindex = {}
    wordindex['UTK'] = 0
    sen_wordlist = {}
    position_list = []
    for line in lines:
        line = line.strip()
        line = line.split('::')
        if line[1] in senindex:
            continue
        senindex[line[1]] = sindex
        position_list.append(len(line[0]))
        line[1] = re.sub('[^A-Za-z ]', '', line[1]).lower()
        line[1] = re.sub(' +', ' ', line[1])
        #print line
        s = nltk.stem.SnowballStemmer('english')
        words = line[1].split(' ')
        stopset = set(stopwords.words('english'))
        wordlist = [word for word in words if word not in stopset]
        # print wordlist
        line1 = [s.stem(word) for word in wordlist]
        line[1] = [word for word in line1 if word.isalpha()]
        #if len(line) > max:
            #max = len(line)
            #print line
        for word in line[1]:
            if word not in wordindex:
                wordindex[word] = wordcount
                wordcount = wordcount + 1
        word_index_input = []
        for word in line[1]:
            if len(word_index_input) < (max - 1):
                word_index_input.append(wordindex[word])
            else:
                break
        while len(word_index_input) < max:
            word_index_input.append(0)
        sen_wordlist[sindex] = word_index_input
        sindex = sindex + 1
    file_w = open( 'data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/position_list', 'wb')
    cPickle.dump(position_list, file_w)
    file_w.close()
    file_w = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/wordindex', 'wb')
    cPickle.dump(wordindex, file_w)
    file_w.close()
    file_w = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/senindex', 'wb')
    cPickle.dump(senindex, file_w)
    file_w.close()
    file_w = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/sen_wordlist', 'wb')
    cPickle.dump(sen_wordlist, file_w)
    file_w.close()
    print senindex
    sencount = len(senindex)
    print len(position_list)
    print sencount
    return wordcount, sencount
def sigmoid(x):
    result =1.0 / (1 + math.exp(-x * 1.0))
    return result
def iter():
    wordcount, sencount = init()
    print sencount
    allword_vector = np.random.random(size=(wordcount, vecdim)).astype(np.float32)*2-1
    print allword_vector
#Random_a.dtype = 'float32'

    allsen_vector = np.random.random(size=(sencount, vecdim)).astype(np.float32)*2-1
    print len(allsen_vector)
    file_w = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/sen_vector', 'wb')
    cPickle.dump(allsen_vector, file_w)
    file_w.close()
#Random_b.dtype = 'float32'
    allsen_parameter = np.random.random(size=(sencount, max*vecdim + 2*vecdim)).astype(np.float32)
    print 'start-------------'
    last_lost = float("-inf")
    loss = [[], []]
    for t in range(Iteration):
        if t > 20:
            current_learn_rate = current_learn_rate - (learn_rate - lowe_learn_rate)/(Iteration - 10)
        else:
            current_learn_rate = learn_rate
        ISOTIMEFORMAT = '%Y-%m-%d %X'
        print time.strftime(ISOTIMEFORMAT, time.localtime())
        print 'The Iteration of   ' + str(t) + ':'
        file_r = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/senindex', 'rb')
        senindex = cPickle.load(file_r)
        file_r.close
        file_r = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/wordindex', 'rb')
        wordindex = cPickle.load(file_r)
        file_r.close
        file_r = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/sen_wordlist', 'rb')
        sen_wordlist = cPickle.load(file_r)
        file_r.close
        alldata = allsen_vector
        if t == 0:
            g = mixture.GMM(n_components=K, n_iter= 2, covariance_type='full')
        else:
            g = mixture.GMM(n_components=K, n_iter= 2,covariance_type='full', init_params='')
            g.weights_ = weight
            g.means_ = mean
            g.covars_ = covars
        g.fit(alldata)
        weight = g.weights_
        mean = g.means_
        covars = g.covars_
        # iter sentence
        keys = senindex.keys()
        lost = 0
        j = -1
        alldata_topic = g.predict(alldata)
        for key in keys:
            j = j + 1
            if j%10000 == 0:
                print j
            T = g.predict_proba(alldata[senindex[key]].reshape(1, -1))[0]
            T = np.ndarray.tolist(T)
            sen_vector = allsen_vector[senindex[key]]
            sen_vector = np.ndarray.tolist(sen_vector)
            word_vec_list = sen_wordlist[senindex[key]]
            word_vector = []
            #print len(word_vec_list)
            for i in range(max):
                word_vector.append(allword_vector[word_vec_list[i]])
            word_vector = np.array(word_vector)
            word_vector = word_vector.reshape(1,-1)[0]
            word_vector = np.ndarray.tolist(word_vector)
            feature  = T + sen_vector + word_vector
            feature = np.array(feature)
            #print feature.shape
            sen_parameter = allsen_parameter[senindex[key]]
            sen_parameter = sen_parameter + current_learn_rate*(1 - sigmoid(np.dot(sen_parameter, feature)))*feature
            allsen_parameter[senindex[key]] = sen_parameter
            # update sample
            random_list_topic = [i for i in range(K)]
            current_topic = g.predict(alldata[senindex[key]].reshape(1, -1))
            random_list_topic.remove(current_topic)
            random_list_topic = random.sample(random_list_topic, NEG_number)
            random_list = []
            for i in range(len(random_list_topic)):
                random_list_index = find_all_index(alldata_topic, random_list_topic[i])
                random_list_index = random.sample(random_list_index, 1)
                random_list.append(random_list_index[0])
            #print random_list
            for i in range(len(random_list)):
                sen_parameter = allsen_parameter[random_list[i]]
                sen_parameter = sen_parameter + current_learn_rate * (0 - sigmoid(np.dot(sen_parameter, feature))) * feature
                allsen_parameter[random_list[i]] = sen_parameter
            # update sentence vector and word vector
            update_feature = [0 for i in range(max*vecdim + 2*vecdim)]
            sen_parameter = allsen_parameter[senindex[key]]
            update_feature = update_feature + (1 - sigmoid(np.dot(sen_parameter, feature))) * sen_parameter
            for i in range(len(random_list)):
                sen_parameter = allsen_parameter[random_list[i]]
                update_feature = update_feature +  (0 - sigmoid(np.dot(sen_parameter, feature))) * sen_parameter
            feature = feature + current_learn_rate * update_feature
            allsen_vector[senindex[key]] = feature[vecdim : 2*vecdim]
            for i in range(max):
                allword_vector[word_vec_list[i]] = feature[(2+i)*vecdim : (3+i)*vecdim]
            lost = lost + math.log(sigmoid(np.dot(allsen_parameter[senindex[key]], feature)))
            for i in range(len(random_list)):
                lost = lost + math.log(1 - sigmoid(np.dot(allsen_parameter[random_list[i]], feature)))
        print lost
        file_w = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/sen_vector', 'wb')
        cPickle.dump(allsen_vector, file_w)
        file_w.close()
        file_w = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/sen_vector'+ str(t) + '.txt', 'w')
        for key in keys:
            file_w.write(key + '/t' + str(allsen_vector[senindex[key]]) + '\n')
        file_w.close()
        file_w = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/word_vector'+ str(t) + '.txt', 'w')
        words = wordindex.keys()
        for word in words:
            file_w.write(word + '/t' + str(allword_vector[wordindex[word]]) + '\n')
        file_w.close()
        file_w = open('data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/word_vector', 'wb')
        cPickle.dump(allword_vector, file_w)
        file_w.close()
        file_w = open( 'data_GMM_sample' + '_' + str(vecdim) + '_' + str(learn_rate) + '_' + str(lowe_learn_rate) + '/sen_topic', 'wb')
        cPickle.dump(alldata_topic, file_w)
        file_w.close()
        if abs(lost - last_lost) < tol:
            break
        last_lost = lost
        loss[0].append([t])
        loss[1].append([lost])
    fig = plt.figure()
    loss = np.array(loss)
    ax = fig.add_subplot(111)
    ax.scatter(loss[0, :], loss[1, :])
    plt.show()


iter()