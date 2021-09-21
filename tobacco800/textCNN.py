import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Text CNN model
class textCNN(nn.Module):
    
    def __init__(self, vocab_built, dim_channel, kernel_wins, dropout_rate, num_class):
        super(textCNN, self).__init__()
        #load pretrained embedding in embedding layer.
        emb_dim = vocab_built.vectors.size()[1]
        self.embed = nn.Embedding(len(vocab_built), emb_dim)
        self.embed.weight.data.copy_(vocab_built.vectors)
    
        #Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        #Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        #FC layer
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
        
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)
        
        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit
    
    
    
class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                      embedding_dim=kwargs["embedding_dim"],
                                      padding_idx=kwargs["pad_idx"])
        self.embeddings.weight.requires_grad = True  # to not refine-tune

        if kwargs["model"] == "lstm":
            self.lstm = nn.LSTM(input_size=kwargs["embedding_dim"],  # input
                                hidden_size=kwargs["lstm_units"],  # output
                                num_layers=kwargs["lstm_layers"],
                                bidirectional=False,
                                batch_first=True)
        if kwargs["model"] == "BiLSTM":
            self.lstm = nn.LSTM(input_size=kwargs["embedding_dim"],  # input
                                hidden_size=kwargs["bilstm_units"],  # output
                                num_layers=kwargs["bilstm_layers"],
                                bidirectional=True,
                                batch_first=True)

        self.dropout = nn.Dropout(kwargs["dropout"])
        self.tanh = F.tanh
        self.dropout = nn.Dropout(kwargs["dropout"])

    def forward(self):
        pass


class LSTM_Model(Model):
    """
    a class to define multiple models
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, question, answer):
        question_embedding = self.embeddings(question)
        # print("question embedding shape:", question_embedding.shape)
        answer_embedding = self.embeddings(answer)
        # print("answer embedding shape:", answer_embedding.shape)
        q_output, (qhidden, qcell) = self.lstm(question_embedding)
        print("q_output shape:", q_output.shape)
        # print("qhidden shape:", qhidden.shape)
        # print("qcell shape:", qcell.shape)
        a_output, (ahidden, acell) = self.lstm(answer_embedding)
        print("a_output shape:", a_output.shape)
        # print("ahidden shape:", ahidden.shape)
        # print("acell shape:", acell.shape)
        # qa_similary = torch.mm(qhidden[-1], ahidden[-1])
        # qa_similary =torch.matmul((qhidden[-1]), torc.th(ahidden[-1]))
        q_output = q_output[-1]
        q_output = q_output.squeeze()
        a_output = a_output[-1]
        a_output = a_output.squeeze()
        mm = torch.mul((q_output), (a_output))
        mm -= mm.min(1, keepdim=True)[0]
        mm /= mm.max(1, keepdim=True)[0]
        qa_similary =torch.mean(mm, dim=1)
        # print("qa_similary shape:", qa_similary.shape)
        return qa_similary, qhidden

    print("**************************MODEL DEFINE & CREATED!****************************")