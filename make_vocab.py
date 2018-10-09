import argparse
import collections
from nltk import word_tokenize,pos_tag
parser = argparse.ArgumentParser()
parser.add_argument('-file_merge',default="./data/yelp/train.merge",type=str,
                    help="""File to create vocab""")
parser.add_argument('-file_vocab',default="./data/yelp/vocab",type=str,
                    help="""File used to save vocab""")


opt = parser.parse_args()

file_merge = open(opt.file_merge,'r')
file_vocab = open(opt.file_vocab,'a+')

file_set = file_merge.readlines()

file_vocab.write('<PAD>'+'\n')
file_vocab.write('<BOS>'+'\n')
file_vocab.write('<EOS>'+'\n')
file_vocab.write('<UNK>'+'\n')

vocab_set = []
vocab_set.append('<PAD>')
vocab_set.append('<BOS>')
vocab_set.append('<EOS>')
vocab_set.append('<UNK>')
word_box = []


def fre(s):
    return s[1]

for i in range(len(file_set)):
    temp = word_tokenize(file_set[i])
    word_box.extend(temp)
    if i % 1000 == 0:
        print(i)
cnt = collections.Counter(word_box)
list1 = cnt.most_common(11000)
print(list1)

for tube in list1:
    file_vocab.write(tube[0]+'\n')
file_merge.close()
file_vocab.close()

