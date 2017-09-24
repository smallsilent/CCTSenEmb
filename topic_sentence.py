# coding = utf-8
import cPickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

data_name = '06data_w2v_64_0.05_0.0001'
K = 64
topic_1 = 11
topic_2 = 18
topic_3 = 27
topic_4 = 3
zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

def read_data():
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
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2])
    plt.show()"""

    strint_topic = {}
    keys = senindex.keys()
    for key in keys:
        strint_topic[key] = sen_topic[senindex[key]]

    strint_topic = sorted(strint_topic.iteritems(), key=lambda e: e[1], reverse=True)

    file_w = open('topic/06-'+ str(K) + 'sen_topic.txt', 'w')

    topic_1_data = []
    topic_2_data = []
    topic_3_data = []
    topic_4_data = []

    for i in range(0, len(strint_topic)):
        file_w.write(str(strint_topic[i][1]) + '::' + str(strint_topic[i][0]) + '\n')

        if strint_topic[i][1] == topic_1:
            topic_1_data.append(sen_vector[senindex[str(strint_topic[i][0])]])
        elif strint_topic[i][1] == topic_2:
            topic_2_data.append(sen_vector[senindex[str(strint_topic[i][0])]])
        elif strint_topic[i][1] == topic_3:
            topic_3_data.append(sen_vector[senindex[str(strint_topic[i][0])]])
        elif strint_topic[i][1] == topic_4:
            topic_4_data.append(sen_vector[senindex[str(strint_topic[i][0])]])

    X_tsne_1 = TSNE(learning_rate=100).fit_transform(topic_1_data)
    X_tsne_1 = X_tsne_1 + 0.4
    X_tsne_2 = TSNE(learning_rate=100).fit_transform(topic_2_data)
    X_tsne_3 = TSNE(learning_rate=100).fit_transform(topic_3_data)
    X_tsne_4 = TSNE(learning_rate=100).fit_transform(topic_4_data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Topic1 = ax.scatter(X_tsne_1[:, 0], X_tsne_1[:, 1],marker = 'x',color='m',s = 5)
    Topic2 = ax.scatter(X_tsne_2[:, 0], X_tsne_2[:, 1],marker = '+',color='c')
    ax.legend((Topic1, Topic2), (u'Topic11', u'Topic18'), loc=2, fontsize=20)
    #legend.get_title().set_fontsize(fontsize=32)
    ax.scatter(X_tsne_3[:, 0], X_tsne_3[:, 1],color='b')
    ax.scatter(X_tsne_4[:, 0], X_tsne_4[:, 1],color='m')


    #ax.set_xlabel('temp', fontsize=18, labelpad=12.5)
    #ax.set_ylabel('temp', fontsize=18, labelpad=12.5)
    plt.xlim(-0.5, 0.8)
    plt.ylim(-0.5, 0.8)
    xmajorLocator = MultipleLocator(0.4)
    ax.xaxis.set_major_locator(xmajorLocator)
    ymajorLocator = MultipleLocator(0.4)
    ax.yaxis.set_major_locator(ymajorLocator)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.show()





if __name__ == "__main__":
    read_data()

