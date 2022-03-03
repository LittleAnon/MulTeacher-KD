import torch.nn as nn
import torch

class TN(nn.Module):
    def __init__(self, config, n_classes, output_mode, rank = 10):
        super().__init__()
        self.r = rank
        self.vocab_size = config.bert_config.vocab_size
        self.G = nn.Embedding(self.vocab_size,self.r * self.r)
        self.G1 = nn.Linear(1,self.r)   
        self.masked_token = self.G.weight.view(-1,self.vocab_size,self.r,self.r)
        self.n_class = n_classes
        self.linear = torch.nn.Linear(self.r,self.n_class) # 相当于G_n
        self.output_mode = output_mode
    def encoding(self,x):
        batch_size,seq_length = x.shape[0],x.shape[1]

        encoded = self.G(x)
        encoded = encoded.view(batch_size, seq_length,self.r,self.r)
        result = self.G1.weight.view(-1,self.r).unsqueeze(0).repeat([batch_size,1,1])

        for i in range(seq_length):
            # if x[:,i] == 100:
            #     result = torch.einsum("bij,bvjr->bvr",[result,self.masked_token])
            # else:
            result = torch.einsum("bij,bjk->bik",[result,encoded[:,i,:,:]])
            norm_result = torch.nn.functional.normalize(result,dim = 2)
        return norm_result.squeeze()

    def forward(self,x):
        input_ids, mask, segment_ids, seq_lengths = x
        encoded_x = self.encoding(input_ids)
        return self.linear(encoded_x).squeeze()

    def get_structure_penalty(self): # unitary or l2 loss
        embedding = self.encoder.weight.view(-1,self.d,self.d)
        embedding_transpose = embedding.transpose(-1,-2)
        product = torch.bmm(embedding_transpose,embedding)
        diff = torch.abs(product.mean(0).squeeze() -torch.eye(self.d,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        return torch.sum(diff)