# coding = utf-8

import os
import codecs
import math
import cPickle
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

vecdim = 64
pa = 'data_w2v_64_0.05_0.0001_15'
name = 'newDUC05'
data_name = pa
sum_name = 'summ_' + pa
summ_len = 251
Threshold_cos = 0.5


def computecos(x,y):
    if (len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return
    x = np.matrix(x)
    y = np.matrix(y)
    result1 = x * y.T
    result2 = x * x.T
    result3 = y * y.T
    result = result1 / (math.pow(result2 * result3 , 0.5))
    return result


def idf(currentword, file):
    cantion_file = 0
    file_list = os.listdir(name)
    total_file = len(file_list)
    for eachfile in file_list:
        if eachfile == file:
            cantion_file = cantion_file + 1
            continue
        fileopen = codecs.open(name + '/' + eachfile, 'r', 'utf-8')
        file_r = fileopen.readlines()
        fileopen.close
        for line in file_r:
            line = line.strip()
            line = line.split('::')
            if line[0].find('d') > -1:
                continue
            else:
                line[1] = re.sub('[^A-Za-z ]', '', line[1]).lower()
                line[1] = re.sub(' +', ' ', line[1])
                # print line
                s = nltk.stem.SnowballStemmer('english')
                words = line[1].split(' ')
                stopset = set(stopwords.words('english'))
                wordlist = [word for word in words if word not in stopset]
                # print wordlist
                line1 = [s.stem(word) for word in wordlist]
                line[1] = [word for word in line1 if word.isalpha()]
                if currentword in line[1]:
                    cantion_file = cantion_file + 1
                    break
    return cantion_file, total_file



def tf(file):
    wordcount = {}
    fileopen = codecs.open(name + '/' + file, 'r', 'utf-8')
    file_r = fileopen.readlines()
    fileopen.close
    count = 0
    for line in file_r:
        line = line.strip()
        line = line.split('::')
        if line[0].find('d') > -1:
            if line[0].find('q') > -1:
                query = line[1]
            elif line[0].find('s') > -1:
                supplment = line[1]
        else:
            line[1] = re.sub('[^A-Za-z ]', '', line[1]).lower()
            line[1] = re.sub(' +', ' ', line[1])
            # print line
            s = nltk.stem.SnowballStemmer('english')
            words = line[1].split(' ')
            stopset = set(stopwords.words('english'))
            wordlist = [word for word in words if word not in stopset]
            # print wordlist
            line1 = [s.stem(word) for word in wordlist]
            line[1] = [word for word in line1 if word.isalpha()]
            for word in line[1]:
                if word == ' ':
                    continue
                elif wordcount.has_key(word):
                    count = count + 1
                    wordcount[word] = wordcount[word] + 1
                else:
                    count = count + 1
                    wordcount[word] = 1
    file_w = open(sum_name + '/' + file + 'wordcount', 'wb')
    cPickle.dump(wordcount, file_w)
    file_w.close()
    file_w = open(sum_name + '/' + file + 'wordcount.txt', 'wb')
    wkeys = wordcount.keys()
    for key in wkeys:
        file_w.write(key + ' ' + str(wordcount[key]).replace('\n', '') + '\n')
    file_w.close()
    return count


def position():
    file_r = open(data_name + '/position_list', 'rb')
    position_list = cPickle.load(file_r)
    file_r.close
    return position_list


def tf_idf():
    file_list = os.listdir(name)
    for file in file_list:
        fileopen = codecs.open(name + '/' + file, 'r', 'utf-8')
        file_r = fileopen.readlines()
        fileopen.close
        text = []
        for line in file_r:
            line = line.strip()
            line = line.split('::')
            if line[0].find('d') == 0:
                line[1] = re.sub('[^A-Za-z ]', '', line[1]).lower()
                line[1] = re.sub(' +', ' ', line[1])
                # print line
                s = nltk.stem.SnowballStemmer('english')
                words = line[1].split(' ')
                stopset = set(stopwords.words('english'))
                wordlist = [word for word in words if word not in stopset]
                # print wordlist
                line1 = [s.stem(word) for word in wordlist]
                line[1] = [word.encode("utf-8") for word in line1 if word.isalpha()]
                line1 = ''
                for i in range(len(line[1])):
                    line1 = line1 + line[1][i] + ' '
                line1 = line1.strip()
                text.append(line1)
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(text))
        word = vectorizer.get_feature_names()
        weight = tfidf.toarray()
        #print type(word)
        #print type(weight)
        file_w = open(sum_name + '/' + file + 'word', 'wb')
        cPickle.dump(word, file_w)
        file_w.close()
        file_w = open(sum_name + '/' + file + 'weight', 'wb')
        cPickle.dump(weight, file_w)
        file_w.close()
        """for i in range(len(weight)):
            print u"-------", i, u"tf-idf------"
            for j in range(len(word)):
                print word[j], weight[i][j]"""


def topic():
    file_r = open(data_name + '/sen_topic49', 'rb')
    topic = cPickle.load(file_r)
    file_r.close
    return topic


def sen_tf(file,line1,j):
    #print line1
    #count = tf(file)
    file_w = open(sum_name + '/' + file + 'weight', 'rb')
    weight = cPickle.load(file_w)
    file_w.close()
    file_w = open(sum_name + '/' + file + 'word', 'rb')
    words = cPickle.load(file_w)
    file_w.close()

    line1 = re.sub('[^A-Za-z ]', '', line1).lower()
    line1 = re.sub(' +', ' ', line1)
    # print line
    s = nltk.stem.SnowballStemmer('english')
    words = line1.split(' ')
    stopset = set(stopwords.words('english'))
    wordlist = [word for word in words if word not in stopset]
    # print wordlist
    line1 = [s.stem(word) for word in wordlist]
    line11 = [word.encode("utf-8") for word in line1 if word.isalpha()]
    line1 = ''
    for i in range(len(line11)):
        line1 = line1 + line11[i] + ' '
    line1 = line1.strip()
    line1 = line1.split(' ')
    tf_score = 0.0

    for word in line1:
        if word not in words:
            print word
            continue
        index = words.index(word)
        print j
        tf_score = tf_score + weight[j][index]
    return tf_score

"""file_w = open(sum_name + '/' + file + 'wordcount', 'rb')
    wordcount = cPickle.load(file_w)
    file_w.close()

    fileopen = codecs.open(name + '/' + file, 'r', 'utf-8')
    file_r = fileopen.readlines()
    fileopen.close
    tf_score = {}
    for line in file_r:
        sen_tf_zhi = 0.0
        line = line.strip()
        line = line.split('::')
        if line[0].find('d') > -1:
            if line[0].find('q') > -1:
                query = line[1]
            elif line[0].find('s') > -1:
                supplment = line[1]
        else:
            line1 = re.sub('[^A-Za-z ]', '', line[1]).lower()
            line1 = re.sub(' +', ' ', line1)
            # print line
            s = nltk.stem.SnowballStemmer('english')
            words = line1.split(' ')
            stopset = set(stopwords.words('english'))
            wordlist = [word for word in words if word not in stopset]
            # print wordlist
            line1 = [s.stem(word) for word in wordlist]
            line11 = [word for word in line1 if word.isalpha()]
            for word in line11:
                idf_count,total= idf(word, file)
                idf_score = np.log(float(total)/idf_count)
                sen_tf_zhi = sen_tf_zhi + wordcount[word]*idf_score
                sen_tf_zhi = sen_tf_zhi/count
            tf_score[line[1]] = sen_tf_zhi"""



def summarization_mert(weight,file):
    # 0.22 0.14 0.45 0.19
    # 05-32 2.7 9.7 1.0 1.0
    # 05-64 3.1 7.3 3.9 1.0
    # 05-16 4.6 6.3 2.4 1.0
    # 06-32 3.8 8.1
    # 06-64 6.4 5.5 1.0 1.0
    #06-64-0.5 1.0 4.1 1.0 1.0
    #05-64-0.5 2.5 2.0 1.0 1.0
    #06-64-10 1.0 4.1 1.0 1.0

    a_tf = 1.0
    a_sensim = 4.1
    #a_position = 0.81
    a_topic = 1.0

    file_r = open(sum_name + '/' + file + 'score', 'rb')
    score = cPickle.load(file_r)
    file_r.close()
    file_r = open(sum_name + '/' + file + 'sen_score', 'rb')
    senscore = cPickle.load(file_r)
    file_r.close()
    file_r = open(data_name + '/sen_vector49', 'rb')
    sen_vector = cPickle.load(file_r)
    file_r.close
    file_r = open(data_name + '/senindex', 'rb')
    senindex = cPickle.load(file_r)
    file_r.close
    keys = senscore.keys()
    for key in keys:
        scorelist = senscore[key]
        #score = a_tf * scorelist[0]  + a_sensim * scorelist[1] + a_position * scorelist[2] + a_topic * scorelist[3]
        score = scorelist[2] *(a_tf * scorelist[0] + a_sensim * scorelist[1] + a_topic * scorelist[3])
        senscore[key] = score
    senscore = sorted(senscore.iteritems(), key=lambda e: e[1], reverse=True)
    # for i in range(len(score)):
    # print score[i][1]
    length_summarization = 0
    summarization = []
    # stop_summarization = []
    i = -1
    # print score[0][0]
    length = 0
    while length < 10:
        i = i + 1
        length = senscore[i][0].split(' ')
        length = len(length)

    #print i
    summarization.append(senscore[i][0])
    i = i + 1

    while length_summarization < summ_len:
        # for i in range(1,len(score)):
        if i > len(senscore) - 1:
            length = senscore[i-1][0].split(' ')
            length = len(length)
            length_summarization = length_summarization + length
            summarization.append(senscore[i-1][0])
            continue
        length = senscore[i][0].split(' ')
        length = len(length)
        if length < 10:
            i += 1
            continue
        j = 0
        for j in range(len(summarization)):
            sim = computecos(sen_vector[senindex[senscore[i][0]]], sen_vector[senindex[summarization[j]]])
            # print sim
            if sim > Threshold_cos:
                break
        if j == len(summarization) - 1:
            summarization.append(senscore[i][0])
            # line = stop(score[i][0])
            # stop_summarization.append(line)
            length = senscore[i][0].split(' ')
            length = len(length)
            length_summarization = length_summarization + length
            # print  length_summarization
        i += 1
    # print summarization
    file_write = codecs.open(sum_name + '/' + 'task' + file[0:-4] + '_' + str(weight ) + '.txt', 'w', 'utf-8')
    for t in range(len(summarization)):
        file_write.write(summarization[t] + '\n')
    file_write.close()


def stop(summarization):
    print summarization
    s = nltk.stem.SnowballStemmer('english')
    words = summarization.split(' ')
    print words
    stopset = set(stopwords.words('english'))
    wordlist = [word.encode("utf-8") for word in words if word not in stopset]
    line = ''
    for i in range(len(wordlist)):
        line = line + wordlist[i] + ' '
    line = line.strip()
    return line


def NERandWF(summarization):
    line = str(summarization)
    print line
    tokens = nltk.word_tokenize(line)
    print tokens
    tags = nltk.pos_tag(tokens)
    print tags
    ners = nltk.ne_chunk(tags)
    print '%s --- %s' % (str(ners), str(ners.node))
    return str(ners)


def main(a_position):
    if not os.path.exists(sum_name):
        os.makedirs(sum_name)

    tf_idf()

    file_r = open(data_name + '/sen_vector49', 'rb')
    sen_vector = cPickle.load(file_r)
    file_r.close
    print len(sen_vector)
    file_r = open(data_name + '/sen_topic49', 'rb')
    sen_topic = cPickle.load(file_r)
    file_r.close
    file_r = open(data_name + '/senindex', 'rb')
    senindex = cPickle.load(file_r)
    file_r.close
    print len(senindex)
    """fig = plt.figure()
    data = sen_vector[500:1500]
    sen_topic1 = sen_topic[0:500]
    data_1 = [[np.random.random() for i in range(1000)]]
    data_1 = np.array(data_1)
    data_1 = data_1.T
    X_tsne = TSNE(learning_rate=100).fit_transform(data)
    print type(X_tsne)
    print len(X_tsne[0])
    print len(X_tsne[1])
    # np.insert(X_tsne, 2, values=data_1, axis=0)
    X_tsne = np.concatenate([X_tsne, data_1], axis=1)
    print X_tsne
    ax = fig.add_subplot(111)
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:,2])
    plt.show()"""

    position_list = position()
    print len(position_list)
    file_list = os.listdir(name)
    for file in file_list:
        file_open = codecs.open(name + '/' + file, 'r', 'utf-8')
        lines = file_open.readlines()
        file_open.close()
        score = {}
        sen_score = {}
        print file
        text_index = -1
        for line in lines:
            line = line.strip()
            line = line.split('::')
            if line[0].find('d') == 0:
                text_index = text_index + 1
            elif '--------' in line[1]:
                continue
            elif line[0].find('s') > -1:
                query = line[1]
            elif line[0].find('q') >-1:
                line[0]
            else:
                if line[1] not in senindex:
                    continue
                x = sen_vector[senindex[query]]
                #print x
                topic = sen_topic[senindex[query]]
                x_topic = [0 for i in range(vecdim)]
                x_topic[topic] = 1
                #print x_topic
                #x = np.insert(x, vecdim, values=x_topic)
                #print x
                y = sen_vector[senindex[line[1]]]
                topic = sen_topic[senindex[line[1]]]
                y_topic = [0 for i in range(vecdim)]
                y_topic[topic] = 1
                #y = np.insert(y, vecdim, values=y_topic)
                #print position_list[senindex[line[1]]]
                #print senindex[line[1]]
                #print line[1]
                #position_score = 1.0/position_list[senindex[line[1]]]
                position_score = a_position ** position_list[senindex[line[1]]]
                topic_score = computecos(x_topic, y_topic)
                tf1 = sen_tf(file, line[1],text_index)
                result = computecos(x, y)
                zscore = [tf1, result,position_score,topic_score]
                sen_score[line[1]] = zscore
                score[line[1]] = result
        file_w = open(sum_name + '/' + file + 'score', 'wb')
        cPickle.dump(score, file_w)
        file_w.close()
        file_w = open(sum_name + '/' + file + 'sen_score', 'wb')
        cPickle.dump(sen_score, file_w)
        file_w.close()
        file_w = open(sum_name + '/' + file + 'sen_score.txt', 'wb')
        wkeys = sen_score.keys()
        for key in wkeys:
            file_w.write(key + ' ' + str(sen_score[key]).replace('\n', '') + '\n')
        file_w.close()

def final_summ(a_position):
    file_list = os.listdir(name)
    for file in file_list:
        print file
        for i in np.arange(1.0, 1.1, 0.1):
            #print i
            weight = 1
            #weight = a_position
            summarization_mert(weight, file)


"""for i in np.arange(0.57, 1.0, 0.01):
    # print i
    a_position = i
    main(a_position)
    final_summ(a_position)"""

main(0.99)
final_summ(0.99)
    #summarization = stop(summarization)
    #NERandWF(summarization)








