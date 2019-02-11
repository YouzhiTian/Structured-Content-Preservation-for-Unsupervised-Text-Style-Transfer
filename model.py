import torch
from torch import nn
import torch.nn.functional as F
from BahdanauAttnDecoderRNN import BahdanauAttnDecoderRNN
from CNNModels import CnnTextClassifier
class CtrlGenModel(nn.Module):
    def __init__(self,config,vocab_size,batch_size,weights_matrix):
        super(CtrlGenModel,self).__init__()
        #64*16(17)*100
        embed_size = config.model["embedder"]["dim"]
        hidden_size = config.model["encoder"]["rnn_cell"]["kwargs"]["num_units"]
        self.hidden_size = hidden_size
        num_layers = 2
        self.softmax = F.log_softmax
        self.embedder = nn.Embedding(vocab_size, embed_size).cuda()
        self.embedder.load_state_dict({'weight': weights_matrix})
        
        #Classifier pretrained Embedding
        self.clas_embedder = nn.Embedding(vocab_size,embed_size).cuda()
        self.clas_embedder.load_state_dict({'weight': weights_matrix})
        
        self.vocab_size = vocab_size
        self.vocab_tensor = torch.LongTensor([i for i in range(vocab_size)]).cuda()
        self.batch_size = batch_size
            
        #The number layer can be two
        self.encoder = nn.GRU(input_size = embed_size,hidden_size = hidden_size,dropout = 0.5,batch_first = True).cuda()
        self.dropout = nn.Dropout(0.5).cuda()
        self.dim_c = config.model["dim_c"]
        self.label_connector = nn.Sequential(nn.Linear(1,hidden_size),nn.Linear(hidden_size,self.dim_c)).cuda()
        self.connector = nn.Linear(700,hidden_size).cuda()
        self.decoder = BahdanauAttnDecoderRNN(hidden_size,embed_size,vocab_size,dropout_p=0.5).cuda()
        
        #Classifier
        self.classifier = CnnTextClassifier(num_filters = 128,vocab_size = vocab_size,emb_size = embed_size,num_classes = 2).cuda()
        
        self.lm = nn.GRU(input_size = embed_size,hidden_size = hidden_size,dropout = 0.5,batch_first = True).cuda()
        self.lm_output = nn.Linear(hidden_size,vocab_size).cuda()
        self.lm_embedder = nn.Embedding(vocab_size,embed_size).cuda()
        self.lm_embedder.load_state_dict({'weight': weights_matrix})
    def forward(self, inputs,sentence_length,if_dis = False,if_eval = False,if_lm = False,gamma = 1):
       
        if if_dis:
            probs,classes = self.classifier(self.clas_embedder(inputs["text_ids"].cuda()))
            return probs,classes

        
        #Train the language model
        #Initial hidden state should be (num_layers * num_directions, batch, hidden_size)
        
        if if_lm:
            hidden_state_1 = inputs["hidden"].cuda()
            hidden_state_2 = inputs["labels"].view(-1,1).float().expand(self.batch_size,(self.hidden_size-300)).cuda()
            hidden_state_lm = torch.cat((hidden_state_2,hidden_state_1),1).unsqueeze(0).cuda()
            text_embedding = self.lm_embedder(inputs["text_ids"].cuda())
            lm_outputs,_ = self.lm(text_embedding,hidden_state_lm)
            lm_outputs = self.lm_output(lm_outputs)
            return lm_outputs
        
        
        input_length = len(inputs["text_ids"].cuda())
        # Change the vocab_tensor
        vocab_tensor = self.vocab_tensor.expand(input_length,self.vocab_size).cuda()
        enc_text_ids = inputs["text_ids"][:,1:].cuda()
        #enc_inputs shape(64,16,100)
        #enc_outputs shape(64,16,700)
        #final_state shape(1,64,700)
        text_embedding = self.embedder(enc_text_ids)
        enc_outputs,final_state = self.encoder(text_embedding)
      
        #Get the final_state
        z = final_state[0,:,self.dim_c:].cuda()
        labels = inputs["labels"].view(-1,1).float().cuda()
        c = self.label_connector(labels).cuda()
        c_ = self.label_connector(1-labels).cuda()
        h = torch.cat((c,z),1).cuda()
        h_ = torch.cat((c_,z),1).cuda()
        #h 64*700
                   
        #(self,embedding, word_input, initial_state, encoder_outputs):
        
        #get the regular decoder result each time using the target input as inupt to calculate the loss_ae
        decoder_hidden = self.connector(h).unsqueeze(0)
        decoder_outputs = torch.Tensor(sentence_length,input_length,self.vocab_size).cuda()
        for di in range(sentence_length):
            decoder_output,decoder_hidden = self.decoder(embedding = self.embedder,word_input = inputs["text_ids"][:,di].cuda(), initial_state = decoder_hidden ,encoder_outputs= enc_outputs)
            #print("decoder_output: ",decoder_output.shape)
            decoder_outputs[di] = decoder_output
        
        #soft_output.sample id called soft_outputs 64 16 9657
        if if_eval:
            decoder_gumbel_hidden = self.connector(h_).unsqueeze(0)
            soft_outputs_ = torch.Tensor(sentence_length,input_length,self.vocab_size).cuda()

            decoder_soft_outputs,decoder_gumbel_hidden = self.decoder(embedding = self.embedder,word_input = inputs["text_ids"][:,0].cuda(),initial_state = decoder_gumbel_hidden,encoder_outputs = enc_outputs,gumbel = True,gamma = gamma)
            soft_outputs_[0] = decoder_soft_outputs
            for di in range(1,sentence_length):
                decoder_soft_outputs,decoder_gumbel_hidden = self.decoder(embedding = self.embedder,word_input = torch.argmax(decoder_soft_outputs,1),initial_state = decoder_gumbel_hidden,encoder_outputs = enc_outputs,gumbel = True,gamma = gamma)
                soft_outputs_[di] = decoder_soft_outputs

            clas_input = torch.bmm(soft_outputs_.transpose(0,1),self.clas_embedder(vocab_tensor))
            probs,classes = self.classifier(clas_input)
        else:
            decoder_gumbel_hidden = self.connector(h_).unsqueeze(0)
            soft_outputs_ = torch.Tensor(sentence_length,input_length,self.vocab_size).cuda()

            decoder_soft_outputs,decoder_gumbel_hidden = self.decoder(embedding = self.embedder,word_input = inputs["text_ids"][:,0].cuda(),initial_state = decoder_gumbel_hidden,encoder_outputs = enc_outputs,gumbel = True,gamma = gamma)
            soft_outputs_[0] = decoder_soft_outputs
            for di in range(1,sentence_length):
                decoder_soft_outputs,decoder_gumbel_hidden = self.decoder(embedding = self.embedder,word_input = torch.argmax(decoder_soft_outputs,1),initial_state = decoder_gumbel_hidden,encoder_outputs = enc_outputs,gumbel = True,gamma = gamma)
                soft_outputs_[di] = decoder_soft_outputs
            soft_outputs_new = soft_outputs_.transpose(0,1)
            #soft_outputs_new is 64*17*9431
                       
            clas_input = torch.bmm(soft_outputs_new,self.clas_embedder(vocab_tensor))     
            probs,classes = self.classifier(clas_input)     
                        
            #language model for the input remove the last output which is generated by EOS and cat the first BOS
            hidden_state_1 = inputs["hidden2"].cuda()
            hidden_state_2 =(1-inputs["labels"]).view(-1,1).float().expand(self.batch_size,(self.hidden_size-300)).cuda()
            hidden_state_lm = torch.cat((hidden_state_2,hidden_state_1),1).unsqueeze(0).cuda()   
            lm_input_new = torch.bmm(soft_outputs_new,self.lm_embedder(vocab_tensor))
            lm_test_input = torch.cat((self.lm_embedder(inputs["text_ids"][:,0].cuda()).unsqueeze(1),lm_input_new[:,:-1,:]),1)
            lm_outputs,_ = self.lm(lm_test_input,hidden_state_lm)
            lm_outputs = self.lm_output(lm_outputs)
            lm_outputs = lm_outputs.transpose(0,1)
        return decoder_outputs,soft_outputs_,probs,classes,lm_outputs
        
            
