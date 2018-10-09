from nltk import pos_tag, word_tokenize
import torchtext.vocab as torch_vocab
import argparse
import torch
parser = argparse.ArgumentParser()

parser.add_argument('-filename', default="./samples/output.txt", type=str,
                    help="""File name""")
opt = parser.parse_args()

file1 = open(opt.filename,'r')
fileset = file1.readlines()
dic_glove = torch_vocab.GloVe(name='twitter.27B',dim=100)
loss_nn  = 0
count = 0
i = 0
while i < len(fileset):
    sentence_ori = fileset[i]
    sentence_gen = fileset[i+1]
    temp_res_ori = pos_tag(word_tokenize(sentence_ori))
    temp_res_gen = pos_tag(word_tokenize(sentence_gen))
    temp_nn_ori = []
    temp_nn_gen = []
    temp_nn_vector_ori = []
    temp_nn_vector_gen = []
    for tube in temp_res_ori:
        if tube[1] == 'NN':
            temp_nn_ori.append(tube[0])
    for tube in temp_res_gen:
        if tube[1] == 'NN':
            temp_nn_gen.append(tube[0])    
    for word in temp_nn_ori:
        try:
            temp_nn_vector_ori.append(dic_glove.vectors[dic_glove.stoi[word]])
        except KeyError:
            a = 1 
    for word in temp_nn_gen:
        try:
            temp_nn_vector_gen.append(dic_glove.vectors[dic_glove.stoi[word]])
        except KeyError:
            a = 1         
    if temp_nn_vector_ori != [] and temp_nn_vector_gen != []:
        loss_list = []
        for vector_target in temp_nn_vector_ori:
            for vector_gen in temp_nn_vector_gen:
                tensor_gen = torch.FloatTensor(vector_gen)
                tensor_target = torch.FloatTensor(vector_target)
                temp_loss = torch.dist(tensor_gen,tensor_target)
                loss_list.append(temp_loss)
        loss_list_new = sorted(loss_list)
        loss_list_new1 = loss_list_new[:min(len(temp_nn_vector_ori),len(temp_nn_vector_gen))]
        loss_nn += (sum(loss_list_new1)/len(loss_list_new1))*(1+abs(len(temp_nn_vector_ori)-len(temp_nn_vector_gen))/len(temp_nn_vector_ori))
        count+=1
    i+=2
    print(i)
print(loss_nn/count)