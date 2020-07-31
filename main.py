import os
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import sys
from config import Config
from model import CtrlGenModel
import torchtext.vocab as torch_vocab
from nltk import pos_tag, word_tokenize
import argparse
import pickle

parser = argparse.ArgumentParser(description='main.py')


parser.add_argument('-if_eval', default=False, type=bool,
                    help="""If the result should be evaluated.""")
parser.add_argument('-if_classifier',default=False,type=bool,
                    help="""If we can test the result of the generated text.""")
parser.add_argument('-if_saveData',default=True,type=bool,
                    help="""If we need to save the data used.""")
parser.add_argument('-file_save', default='./reproduce.txt', type=str,
                    help="""The name of the file which saved the result of the generated text.""")
parser.add_argument('-checkpoint',default='',type=str,
                    help="""The checkpoint of the model we trained before.""")
parser.add_argument('-checkpoint_path',default='./Checkpoint_3/',type=str,
                    help="""The path of the checkpoint.""")
parser.add_argument('-batch_size',default=128,type=int,
                    help="""Batch size.""")
parser.add_argument('-pretrain',default=1,type=int,
                    help="""pretrained epoch before add class loss""")
parser.add_argument('-datapath',default="./data/yelp/",type=str,
                    help="""Path to the dataset.""")
parser.add_argument('-gpu',default=2,type=int,
                    help="""Device to run on""")
parser.add_argument('-max_epoch',default=5,type=int,
                    help="""Max number of epochs""")




opt = parser.parse_args()

def Cross_Entropy(ori,target,batch_size,sentence_length,if_input = False,if_target = False):    
    m = nn.Softmax()
    if if_input:
        ori = m(ori)
    if if_target:
        target = m(target)
    ori = torch.log(ori)
    loss = torch.sum(ori*target)
    return -loss/(batch_size*sentence_length)

def makeVocab(filename):
    
    word_to_id = {}
    id_to_word = {}
    idx = 0
    label_list = []
    for line in open(filename):
        fields = line.split()
        if len(fields) > 1:
            label = ' '.join(fields[:])
        else:
            label = fields[0]
        if label not in label_list:
            word_to_id[label] = idx
            id_to_word[idx] = label
            idx+=1
            label_list.append(label)
    return word_to_id,id_to_word
def make_dev_Data(srcFile0,srcFile1,srcFile_original0,srcFile_original1,filename):
    
    #max_length = 0
    srcF0 = open(srcFile0,'r')
    srcF1 = open(srcFile1,'r')
    srcF_original0 = open(srcFile_original0,'r')
    srcF_original1 = open(srcFile_original1,'r')
    
    file = open(filename,'a+')
    
    srcF0_total = srcF0.readlines()
    srcF1_total = srcF1.readlines()
    srcF_original0_total = srcF_original0.readlines()
    srcF_original1_total = srcF_original1.readlines()
    
    for i in range(len(srcF0_total)):
        flag = 0
        new_sentence = ""
        srcF0_sp = srcF0_total[i].strip().split()
        srcF0_original_sp = srcF_original0_total[i].strip().split()
        for j in range(len(srcF0_original_sp),len(srcF0_sp)):
            if srcF0_sp[j][-1] == '.':
                new_sentence = new_sentence+" "+srcF0_sp[j][:-1]+" "+'.'
                flag = 1
            elif srcF0_sp[j][-1] == '!':
                new_sentence = new_sentence+" "+srcF0_sp[j][:-1]+" "+'!'
                flag = 1
            elif srcF0_sp[j][-1] == '?':
                new_sentence = new_sentence+" "+srcF0_sp[j][:-1]+" "+'?'
                flag = 1
            else:new_sentence = new_sentence + " "+srcF0_sp[j]
        if srcF0_sp[-1] != srcF0_original_sp[-1] and flag == 0:
            new_sentence = new_sentence+" "+srcF0_original_sp[-1]
        file.write(" "+srcF_original0_total[i])
        file.write(new_sentence+'\n')
    for i in range(len(srcF1_total)):
        flag = 0
        new_sentence = ""
        srcF1_sp = srcF1_total[i].strip().split()
        srcF1_original_sp = srcF_original1_total[i].strip().split()
        for j in range(len(srcF1_original_sp),len(srcF1_sp)):
            if srcF1_sp[j][-1] == '.':
                new_sentence = new_sentence+" "+srcF1_sp[j][:-1]+" "+'.'
                flag = 1
            elif srcF1_sp[j][-1] == '!':
                flag = 1
                new_sentence = new_sentence+" "+srcF1_sp[j][:-1]+" "+'!'
            elif srcF1_sp[j][-1] == '?':
                new_sentence = new_sentence+" "+srcF1_sp[j][:-1]+" "+'?'
                flag = 1
            else:new_sentence = new_sentence + " "+srcF1_sp[j]
        if srcF1_sp[-1] != srcF1_original_sp[-1] and flag == 0:
            new_sentence = new_sentence+" "+srcF1_original_sp[-1]
        file.write(" "+srcF_original1_total[i])
        file.write(new_sentence+'\n')

def writefile(human,gen,save):
    human_file = open(human,'r')
    gen_file = open(gen,'r')
    save_file = open(save,'a+')
    
    human_set = human_file.readlines()
    gen_set = gen_file.readlines()
    i = 0
    while i < len(human_set):
        save_file.write(human_set[i])
        save_file.write(gen_set[i*2+1])
        i+=1
        
        
    
def makeData(srcFile,labelFile, word_to_id,glove,save_file,hidden_size,if_nn_save = False,if_glove = True,if_nn = True,if_shuff = True,if_gender = True):
    Dataset = []
    input = {}
    
    original_text = []
    original_id = []
    original_label = []
    original_length = [] 
    
    original_hidden = []
    original_hidden2 = []
    temp_tensor = torch.FloatTensor(hidden_size)
    if if_nn == True:
        if if_gender == True:
            original_nn = []
        else:
            original_nn = np.load(save_file)
    else:
        original_nn = []
    original_nn_vector = []
   
    text_line = []
    max_length = 0

    print('Processing %s & %s ...' % (srcFile, labelFile))
    srcF = open(srcFile, "r")
    labelF = open(labelFile, "r")

    srcSet = srcF.readlines()
    labelSet = labelF.readlines()

    for i in range(len(srcSet)):
        id_line = []
        temp_nn_vector = []
        temp_nn = []
        temp_hidden = []
        if if_nn == False and if_nn_save == True:
            original_nn.append([])
        srcSet[i] = srcSet[i].strip()
        if if_nn_save == False: 
            temp_res = pos_tag(word_tokenize(srcSet[i]))
            for tube in temp_res:
                if tube[1] == 'NN':
                    temp_nn.append(tube[0])
        if if_nn_save == False: 
            original_nn.append(temp_nn)
        if if_nn == True:
            for word in original_nn[i]:
                try:
                    temp_nn_vector.append(glove.vectors[glove.stoi[word]])
                except KeyError:
                    a = 1
                    print("Unknown words: ",word)
        original_label.append(int(labelSet[i].strip()))
        if temp_nn_vector == []:
            original_hidden.append( [int(labelSet[i].strip())]*300)
            original_hidden2.append([1-int(labelSet[i].strip())]*300)
        else:
            temp_tensor = torch.zeros(300)
            for vector in temp_nn_vector:
                temp_tensor+=vector
            temp_tensor = temp_tensor/len(temp_nn_vector)
            original_hidden.append([yi.item() for yi in temp_tensor])
            original_hidden2.append(original_hidden[i])
        if i % 5000 == 0:
            print("now ",i)
        text_line = ["<BOS>"]+srcSet[i].split()+["<EOS>"]
        original_text.append(text_line)
        original_length.append(len(text_line))
        original_nn_vector.append(temp_nn_vector)
        
        if len(text_line) > max_length:
            max_length = len(text_line)
        for word in text_line:
            try:
                id = word_to_id[word] 
            except KeyError:
                id = 3 
            id_line.append(id)
        original_id.append(id_line)
        if if_nn_save == False:
            if i % 1000 == 0:
                print("total: {0} now: {1}".format(len(srcSet),i))
                print("nn: ",original_nn[i])
    if if_nn_save == False:
        save_nn_vector = np.array(original_nn)
        np.save(save_file,save_nn_vector)
    if if_shuff:
        print('... shuffling sentences')
        perm = torch.randperm(len(original_text))
        original_id = [original_id[idx] for idx in perm]
        original_label = [original_label[idx] for idx in perm]
        original_length = [original_length[idx] for idx in perm]
        original_text = [original_text[idx] for idx in perm]
        original_hidden = [original_hidden[idx] for idx in perm]
        original_hidden2 = [original_hidden2[idx] for idx in perm]
    if if_nn == True:
        original_nn = [original_nn[idx] for idx in perm]
        original_nn_vector = [original_nn_vector[idx] for idx in perm]
    
    print('... pedding')
    for i in range(len(original_text)):
        if original_length[i] < max_length:
            for j in range(max_length - original_length[i]):
                original_text[i].append("<PAD>")
                original_id[i].append(0)
    Dataset = {"text":original_text,"length":original_length,"text_ids":original_id,"labels":original_label,"nn":original_nn,"nn_vector":original_nn_vector,"hidden":original_hidden,"hidden2":original_hidden2}
    return Dataset,max_length
def makeBatch(Dataset,batch_size):
    Dataset_total = []
    text = []
    length = []
    text_ids = []
    labels = []
    nn = []
    nn_vector = []
    hidden = []
    hidden2 = []
    unk = []
    temp = {"text":text,"length":length,"text_ids":text_ids,"labels":labels,"nn":nn,"nn_vector":nn_vector,"unk":unk,"hidden":hidden,"hidden2":hidden2}
    for i in range(len(Dataset['text'])):
        temp["text"].append(Dataset['text'][i])
        temp["length"].append(Dataset['length'][i])
        temp["text_ids"].append(Dataset['text_ids'][i])
        temp["labels"].append(Dataset['labels'][i])
        temp["nn"].append(Dataset["nn"][i])
        temp["nn_vector"].append(Dataset["nn_vector"][i])
        temp["hidden"].append(Dataset["hidden"][i])
        temp["hidden2"].append(Dataset["hidden2"][i])
        if ((i+1) % batch_size == 0) or (i == len(Dataset['text']) - 1):
            store = {"text":[row for row in temp['text']],"length":[row for row in temp['length']],"text_ids":[row for row in temp['text_ids']],"labels":[row for row in temp['labels']],"nn":[row for row in temp['nn']],"nn_vector":[row for row in temp['nn_vector']],"hidden":[row for row in temp['hidden']],"hidden2":[row for row in temp['hidden2']]}
            Dataset_total.append(store)
            temp['text'].clear()
            temp['length'].clear()
            temp['text_ids'].clear()
            temp['labels'].clear()
            temp['nn'].clear()
            temp['nn_vector'].clear()
            temp['hidden'].clear()
            temp['hidden2'].clear()
    for i in range(len(Dataset_total)):
        raw_inputs = Dataset_total[i]
        input_text_ids = torch.LongTensor(raw_inputs["text_ids"])
        input_labels = torch.LongTensor(raw_inputs["labels"])
        input_length = torch.IntTensor(raw_inputs["length"])
        input_hidden = torch.FloatTensor(raw_inputs['hidden'])
        input_hidden2 = torch.FloatTensor(raw_inputs['hidden2'])
        #input_nn_vector = torch.FloatTensor(raw_inputs["nn_vector"])
        inputs = {"text":raw_inputs["text"],"labels":input_labels.cuda(),"length":input_length.cuda(),"text_ids":input_text_ids.cuda(),"nn":raw_inputs["nn"],"nn_vector":raw_inputs["nn_vector"],"hidden":input_hidden.cuda(),"hidden2":input_hidden2.cuda()}
        Dataset_total[i] = inputs
    return Dataset_total
def eval_classifier(model,eval_data):
    total_count = 0
    acc_count = 0
    for i in range(len(eval_data)):
        total_count += len(eval_data[i]["labels"])
        probs,classes = model(eval_data[i],if_dis = True)
        acc_count+=torch.sum(classes ==1- eval_data[i]["labels"]) 
        print(i)
    acc_d = acc_count.cpu().float()/float(total_count)
    return acc_d
def trainModel(model,glove,id_to_word,train_data,dev_data,g_vars,config,max_epoch,checkpoint_path,batch_num,criterion1,criterion2,d_vars,lm_vars,eval_data,lambda_n,lambda_g,gamma,file_name,batch_size,sentence_length,sentence_length_dev,if_eval = False,pretrain = 1):
    sys.stdout.flush()
    batch_num_count = 0
    max_acc = 0.90
    lambda_gen = 0
    if if_eval == True:
   
        f1 = open(file_name,"a+")
        total_count = 0
        acc_count = 0
        for mn in range(len(eval_data)):
            eval_output,soft_outputs,probs,classes,_ = model(eval_data[mn],sentence_length)
            total_count+=len(eval_data[mn]["labels"])
            result2 = torch.argmax(soft_outputs.transpose(0,1),2)
            for i in range(batch_size):
                p = ""
                q = ""
                m = ""
                #p = [id_to_word[j.item()] for j in eval_data[79]["text_ids"][i][:]]
                for j in eval_data[mn]["text"][i][1:]:
                    if j == "<EOS>":
                        break
                    p = p+" "+j
                for j in result2[i][:]:
                    if j == 2:
                        break
                    m = m+ " "+ id_to_word[j.item()]
                f1.write(p+'\n')
                f1.write(m+'\n')

                #print("original: ",p)
                #print("generated: ",q)
                #print("soft",m)
            acc_count+=torch.sum(classes == (1-eval_data[mn]["labels"]))
        acc_d = acc_count.cpu().float()/float(total_count)
        print("acc_d: ",acc_d)
               
        return 
    for epoch in range(1,max_epoch):
        batch_num_count = 0
        #shuffule
        perm = torch.randperm(len(train_data))
        train_data = [train_data[idx] for idx in perm]
        
        
        if epoch > pretrain:
            lambda_g = 2
            lambda_gen = 0.5
            gamma = max(0.001, gamma * config.gamma_decay)
        loss_g = 0
        for batch_train_data in train_data:
            batch_num_count += 1
            inputs = train_data[batch_num_count-1] 
            loss_g = 0
            loss_g_ae = 0
            loss_g_clas = 0
            loss_nn = 0
            loss_lm = 0
            loss_lm_res = 0
            loss_lm_gen = 0
            lambda_n = 0.1
            
            #Train the classifier
            d_vars.zero_grad()
            probs,classes = model(inputs,sentence_length,if_dis = True)
            loss_d_clas = criterion2(probs,inputs["labels"])
            loss_d_clas.backward()
            d_vars.step()
            
            #Train the language model
            lm_vars.zero_grad()
            lm_outputs_ori = model(inputs,sentence_length,gamma=gamma,if_lm = True)
            for i in range(16):
                loss_lm += criterion2(lm_outputs_ori[:,i,:].cuda(),inputs["text_ids"][:,i+1])
            loss_lm = loss_lm/16
            loss_lm.backward()
            lm_vars.step()
            
            g_vars.zero_grad()  
            g_outputs,my_soft_outputs,probs,classes,lm_outputs = model(inputs,sentence_length,gamma=gamma)
 
            loss_lm_gen = Cross_Entropy(lm_outputs,my_soft_outputs,batch_size,sentence_length,if_input = True)    
            
            #add my loss here after epoch > pretrain
            if epoch > pretrain:
                g_res = torch.argmax(my_soft_outputs.transpose(0,1),2)
                for i in range(len(g_res)):
                    min_loss = 10000
                    sentence = ""
                    for j in g_res[i][:]:
                        sentence = sentence+" "+id_to_word[j.item()]
                    pos_list = pos_tag(word_tokenize(sentence))
                    soft_nn_vector = []
                    word = ""
                    for tube in pos_list:
                        if tube[1] == 'NN':
                            word = tube[0]
                            if tube[0] == '_num_':
                                word = "number"
                            try:
                                soft_nn_vector.append(glove.vectors[glove.stoi[word]])
                            except KeyError:
                                word = ""
                    loss_list = []
                    if soft_nn_vector != [] and inputs["nn_vector"][i] != []:
                        for vector_target in inputs["nn_vector"][i]:
                            for vector_gen in soft_nn_vector:
                                tensor_gen = torch.FloatTensor(vector_gen).cuda()
                                tensor_target = vector_target.cuda()
                                temp_loss = torch.dist(tensor_gen,tensor_target)
                                loss_list.append(temp_loss)
    #                             if temp_loss < min_loss:
    #                                 min_loss = temp_loss
                        loss_list_new = sorted(loss_list)
    #                    print('loss_list: ',loss_list_new)
                        loss_list_new1 = loss_list_new[:min(len(inputs["nn_vector"][i]),len(soft_nn_vector))]
                        loss_nn += (sum(loss_list_new1)/len(loss_list_new1))*(1+abs(len(inputs["nn_vector"][i])-len(soft_nn_vector))/len(inputs["nn_vector"][i]))
    #                    print("loss_nn: ",loss_nn)
                loss_nn = loss_nn/len(g_res)

            
            for i in range(sentence_length-1):
                loss_g_ae+= criterion1(g_outputs[i,:,:].cuda(),inputs["text_ids"][:,i+1])
            loss_g_clas = criterion2(probs,1-inputs["labels"])
            loss_g = loss_g_ae+lambda_g*loss_g_clas+lambda_n*loss_nn+lambda_gen*loss_lm_gen
            loss_g.backward()
            g_vars.step()
            
            
            if (batch_num_count % 10) == 0 and (batch_num_count != batch_num):
                print("Epoch:{0} Step:{1} Loss_g : {2} Loss_g_ae: {3} loss_d_clas: {4} loss_g_clas: {5} loss_nn: {6} loss_lm: {7} loss_lm_gen:{8} Ppl:{9} ".format(epoch,batch_num_count,loss_g,loss_g_ae,loss_d_clas,loss_g_clas,loss_nn,loss_lm,loss_lm_gen,np.exp(loss_lm.item())))
                if batch_num_count % 200 == 0 and epoch > pretrain:
                    total_count = 0
                    acc_count = 0
                    for mn in range(len(dev_data)):
                        eval_outputs,soft_outputs,probs,classes,_ = model(dev_data[mn],sentence_length_dev)
                        total_count+=len(dev_data[mn]["labels"])
                        acc_count+=torch.sum(classes == (1-dev_data[mn]["labels"]))
                    acc = acc_count.cpu().float()/float(total_count)
                    print("acc",acc)
                    if acc > max_acc:
                        torch.save(model.state_dict(),checkpoint_path+'checkpoint_%d_step_%d_acc_%.2f.pt' % (epoch,batch_num_count,acc))          
                     
            if batch_num_count == batch_num:
                print("End of Epoch:{0} step:{1} Loss_g:{2} ".format(epoch,batch_num_count,loss_g))
                # then evalaute the model
        
        torch.save(model.state_dict(),checkpoint_path+'checkpoint_%d_loss_%.2f.pt' % (epoch,loss_g))

def make_pretrain_embeddings(glove,id_to_word,emb_dim):
    weights_matrix = []
    for i in range(len(id_to_word)):
        try:
            weights_matrix.append(glove.vectors[glove.stoi[id_to_word[i]]])
        except KeyError:
            weights_matrix.append(np.random.normal(scale=0.6, size=(emb_dim,)))

    new_weight = torch.FloatTensor(weights_matrix).cuda()
    return new_weight

    
def main():
    #Config
    config = Config()    
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with -gpu 0")
    if opt.gpu != -1:
        torch.cuda.set_device(opt.gpu)
        print("Using GPU: ",opt.gpu)

    #Make dictionary
    dic_glove = torch_vocab.GloVe(name='twitter.27B',dim=100)
    word_to_id = {}
    id_to_word = {}
    word_to_id,id_to_word = makeVocab(opt.datapath+'vocab')
    vocab_size = len(word_to_id)
    print("Vocab size",len(word_to_id))
    
    #Make the weight matrix which will be used as pretrained embedding.
    weights_matrix = make_pretrain_embeddings(dic_glove,id_to_word,config.model['embedder']['dim'])  
    
    #Glove used to embed the noun contents.
    glove = torch_vocab.GloVe(name='840B', dim=300)
    
    #Data
    train_nn_file = opt.datapath+"train_nn.npy"
    dev_nn_file = opt.datapath+"dev_nn.npy"
    test_nn_file = ""
    Dataset_train = {}
    Dataset_dev = {}
    Dataset_test = {}
    Dataset_dev_new = {}
    
    #define the max length of the sentence including <POS> and <EOS>
    max_length = 17
    max_length_dev = 17
    
    if opt.if_eval == False and opt.if_saveData == True:
        Dataset_train,max_length = makeData(opt.datapath+'train.merge',opt.datapath+'train.labels',word_to_id,glove,train_nn_file,hidden_size = config.hidden_size)
        Dataset_dev,max_length_dev = makeData(opt.datapath+'dev.merge',opt.datapath+'dev.labels',word_to_id,glove,dev_nn_file,hidden_size = config.hidden_size)
    
    if opt.if_eval == True:   
        Dataset_test,_ = makeData(opt.datapath+'test.merge',opt.datapath+'test.labels',word_to_id,glove,test_nn_file,hidden_size = config.hidden_size,if_nn=False,if_nn_save=True,if_shuff = False)

    
    batch_size = opt.batch_size
    print("batch_size :",batch_size)
    Dataset_train_total = []
    Dataset_dev_total = []
    Dataset_test_total = []
    if opt.if_eval == False and opt.if_saveData == True:
        Dataset_train_total = makeBatch(Dataset_train,batch_size)
        Dataset_dev_total = makeBatch(Dataset_dev,batch_size)
                        
        train_batch_num = len(Dataset_train_total)
        dev_batch_num = len(Dataset_dev_total)
        print("train_batch_num: ",train_batch_num)
        print("dev_batch_num: ",dev_batch_num)       
    if opt.if_eval == True:        
        Dataset_test_total = makeBatch(Dataset_test,batch_size)  
        test_batch_num = len(Dataset_test_total)
        print("test_batch_num: ",test_batch_num)
        
    if opt.if_eval == False:
        if opt.if_saveData :
            print("saving train")
            torch.save(Dataset_train_total,opt.datapath+"train_data.pkl")
            print("saving dev")
            torch.save(Dataset_dev_total,opt.datapath+"dev_data.pkl")
            Dataset_train_total = Dataset_train_total[:-1]
            Dataset_dev_total = Dataset_dev_total[:-1]        
        else:
            if opt.gpu!=1:
                print("Loading train data")
                Dataset_train_total = torch.load(opt.datapath+"train_data.pkl",map_location={'cuda:1':'cuda:'+str(opt.gpu)})
                Dataset_train_total = Dataset_train_total[:-1]
                print("Train data: ",len(Dataset_train_total))
                print("Loading dev data")
                Dataset_dev_total = torch.load(opt.datapath+"dev_data.pkl",map_location={'cuda:1':'cuda:'+str(opt.gpu)})  
                Dataset_dev_total = Dataset_dev_total[:-1]
                print("Dev data: ",len(Dataset_dev_total))
            else:    
                print("Loading train data")
                Dataset_train_total = torch.load(opt.datapath+"train_data.pkl")
                Dataset_train_total = Dataset_train_total[:-1]
                print("Train data: ",len(Dataset_train_total))
                print("Loading dev data")
                Dataset_dev_total = torch.load(opt.datapath+"dev_data.pkl")  
                Dataset_dev_total = Dataset_dev_total[:-1]
                print("Dev data: ",len(Dataset_dev_total))
    
    
              
    model = CtrlGenModel(config,vocab_size,batch_size,weights_matrix)
    print(model)
    model = model.cuda()
    print("parameters()",model.parameters())
    #Parameters needed to be optimized in generator.
    g_vars = torch.optim.Adam(
        [   {'params':model.embedder.parameters()},
            {'params':model.encoder.parameters()},
            {'params':model.label_connector.parameters()},
            {'params':model.connector.parameters()},
            {'params':model.decoder.parameters()},
        ],lr = config.learning_rate)
    #Parameters needed to be optimized in classifier.
    d_vars = torch.optim.Adam(
        [
            {'params':model.classifier.parameters()},
            {'params':model.clas_embedder.parameters()},
            ],lr = config.learning_rate)
    #Parameters needed to be optimeized in language model.
    lm_vars = torch.optim.Adam(
        [
            {'params':model.lm.parameters()},
            {'params':model.lm_output.parameters()},
            {'params':model.lm_embedder.parameters()}
        ],lr = config.learning_rate)
    #criterions to be used to calculate the loss.
    criterion1 = nn.NLLLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion1 = criterion1.cuda()
    criterion2 = criterion2.cuda()
   
    print("Begining training.")
    # if we need to restore the model.
    if opt.checkpoint !='':
        model.load_state_dict(torch.load(opt.checkpoint))
    file_name = opt.file_save
    lambda_n = 0
    lambda_g = 0
    gamma = 1.0
    # if we need to use classifier to classify the spectific samples. 
    if opt.if_classifier:
        print(eval_classifier(model,Dataset_new_dev_total))
    trainModel(model,glove,id_to_word,Dataset_train_total,Dataset_dev_total,g_vars,config,opt.max_epoch,opt.checkpoint_path,len(Dataset_train_total),criterion1,criterion2,d_vars,lm_vars,Dataset_test_total,lambda_n,lambda_g,gamma,file_name,batch_size,max_length,max_length_dev,if_eval = opt.if_eval,pretrain=opt.pretrain)
    print("End of training!")
if __name__ == "__main__":
    main()
