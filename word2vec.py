#coding = utf-8

import codecs
import re
import nltk
from nltk.corpus import stopwords

def combine():
    filewrite = codecs.open('wors2vec_corpus.txt', 'w', 'utf-8')
    fileopen = codecs.open('giga_eng.true.en', 'r', 'utf-8')
    line1ss = fileopen.readlines()
    fileopen.close()
    for line1s in line1ss:
        line1s = re.sub('[^A-Za-z ]', '', line1s).lower()
        line1s = re.sub(' +', ' ', line1s)
        #print line
        s = nltk.stem.SnowballStemmer('english')
        words = line1s.split(' ')
        stopset = set(stopwords.words('english'))
        wordlist = [word for word in words if word not in stopset]
        print wordlist
        line1 = [s.stem(word) for word in wordlist]
        line1s = [word for word in line1 if word.isalpha()]
       # print line1s
        line = ''
        for i in range(len(line1s)):
            line = line + line1s[i]+ ' '
        line = line.strip()
        filewrite.write(str(line))
        filewrite.write('\r\n')

    fileopen = codecs.open('05.txt', 'r', 'utf-8')
    line2ss = fileopen.readlines()
    fileopen.close()
    for line2s in line2ss:
        line2s = re.sub('[^A-Za-z ]', '', line2s).lower()
        line2s = re.sub(' +', ' ', line2s)
        # print line
        s = nltk.stem.SnowballStemmer('english')
        words = line2s.split(' ')
        stopset = set(stopwords.words('english'))
        wordlist = [word for word in words if word not in stopset]
        print wordlist
        line2 = [s.stem(word) for word in wordlist]
        line2s = [word.encode("utf-8") for word in line2 if word.isalpha()]
        line = ''
        for i in range(len(line2s)):
            line = line + line2s[i] + ' '
        line = line.strip()
        if line == '':
            continue
        filewrite.write(str(line))
        filewrite.write('\r\n')
        #filewrite.write(str(line2s))
        #filewrite.write('\r\n')

    fileopen = codecs.open('06.txt', 'r', 'utf-8')
    line3ss = fileopen.readlines()
    fileopen.close()
    for line3s in line3ss:
        line3s = re.sub('[^A-Za-z ]', '', line3s).lower()
        line3s = re.sub(' +', ' ', line3s)
        s = nltk.stem.SnowballStemmer('english')
        words = line3s.split(' ')
        stopset = set(stopwords.words('english'))
        wordlist = [word for word in words if word not in stopset]
        print wordlist
        line3 = [s.stem(word) for word in wordlist]
        line3s = [word.encode("utf-8") for word in line3 if word.isalpha()]
        line = ''
        for i in range(len(line3s)):
            line = line + line3s[i] + ' '
        line = line.strip()
        if line == '':
            continue
        filewrite.write(str(line))
        filewrite.write('\r\n')
        #filewrite.write(str(line3s))
        #filewrite.write('\r\n')


    filewrite.close()



def read():
    fileopen = codecs.open('wors2vec_corpus.txt', 'r', 'utf-8')
    lines = fileopen.readlines()
    fileopen.close()
    for line in lines:
        if not line.split():
            print 'konghang'

combine()
#read()

