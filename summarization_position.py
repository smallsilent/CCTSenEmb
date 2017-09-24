#coding = utf-8
import cPickle
import random
import os
import codecs

summ_length = 251
name = 'newDUC05'
data_name = 'data_w2v_64_0.05_0.0001'
sum_name = '05position_summarization'

def position():
    file_r = open(data_name + '/position_list', 'rb')
    position_list = cPickle.load(file_r)
    file_r.close
    return position_list

def main():
    if not os.path.exists(sum_name):
        os.makedirs(sum_name)
    file_r = open(data_name + '/senindex', 'rb')
    senindex = cPickle.load(file_r)
    file_r.close
    positon_list = position()

    file_list = os.listdir(name)
    for file in file_list:
        file_open = codecs.open(name + '/' + file, 'r', 'utf-8')
        lines = file_open.readlines()
        file_open.close()

        candidate_sentence = []

        for line in lines:
            line = line.strip()
            line = line.split('::')
            if line[0].find('d') == 0:
                line[0]
            elif line[0].find('s') > -1:
                query = line[1]
            elif line[0].find('q') > -1:
                line[0]
            else:
                #print positon_list[senindex[line[1]]]
                if positon_list[senindex[line[1]]] == 2:
                    candidate_sentence.append(line[1])

        print candidate_sentence

        random_list = [i for i in range(len(candidate_sentence))]

        length = 0
        summarization = ''
        while length < summ_length:
            random_key = random.sample(random_list, 1)
            print random_key[0]
            random_list.remove(random_key[0])
            summarization = summarization + candidate_sentence[random_key[0]] + '\n'
            #print candidate_sentence[random_key[0]]
            sen_length = candidate_sentence[random_key[0]].split(' ')
            sen_length = len(sen_length)
            length = length + sen_length

        file_write = codecs.open(sum_name + '/' + 'task' + file[0:-4] + '_1.txt', 'w', 'utf-8')
        file_write.write(summarization)
        file_write.close()


main()
