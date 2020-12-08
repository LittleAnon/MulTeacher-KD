import torch
import torch.nn as nn
import genotypes as gt
import torch.nn.functional as F

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'stdconv_3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'stdconv_5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'stdconv_7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'skip_connect': lambda C, stride, affine: Identity(),
    'self_attn': lambda C, stride, affine: SelfAttention(C, 2),
    'dilconv_3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), # 5x5
    'dilconv_5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine), # 9x9
    'dilconv_7': lambda C, stride, affine: DilConv(C, C, 7, stride, 6, 2, affine=affine),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'rnn': lambda C, stride, affine: RNN(C, C, affine=affine)

}

class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )
    def forward(self, x):
        return self.net(x)

class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
    def set_active(self, k=-1):
        for i, op in enumerate(self._ops):
            if i==k:
                for param in op.parameters():
                    param.requires_grad = True
            else:
                for param in op.parameters():
                    param.requires_grad = False


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class Embedding(nn.Module):
    def __init__(self, Dword, Dchar, D):
        super().__init__()
        self.conv2d = nn.Conv2d(Dchar, D, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(Dword+D, D, bias=False)
        self.high = Highway(2, D)
        self.dropout_char = 0.1
        self.dropout = 0.1

    def forward(self, ch_emb, wd_emb):
        N = ch_emb.size()[0]
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        # ch_emb = ch_emb.squeeze()

        wd_emb = F.dropout(wd_emb, p=self.dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout=nn.Dropout(config.hidden_dropout_prob)
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings.permute(0,2,1)

class RobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        #self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        check_w_emb = words_embeddings.cpu()
        position_embeddings = self.position_embeddings(position_ids)
        check_p_emb = position_embeddings.cpu()
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)



        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        checkEmb = embeddings.cpu()
        return embeddings.permute(0,2,1)

class Gpt2Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(Gpt2Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.n_embd, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.n_ctx, config.n_embd)
        self.token_type_embeddings = nn.Embedding(
            1, config.n_embd)


        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.n_embd, eps=1e-12)
        self.dropout=nn.Dropout(config.embd_pdrop)
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        check_w_emb = words_embeddings.cpu()
        position_embeddings = self.position_embeddings(position_ids)
        check_p_emb = position_embeddings.cpu()
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)



        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        checkEmb = embeddings.cpu()
        return embeddings.permute(0,2,1)

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv1d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm1d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class Pooling(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = torch.nn.MaxPool1d(kernel_size, stride=None, padding=padding, return_indices=False, ceil_mode=False)

    def forward(self, x):
        return self.net(x)

class RNN(nn.Module):
    """ Standard conv
    rnn
    """
    def __init__(self, C_in, C_out, bidirectional=True, affine=True):
        super().__init__()
        if bidirectional:
            C_out = C_out // 2
        self.rnn = nn.LSTM(C_in, C_out, num_layers=1, bidirectional=bidirectional)
        # self.layer_norm = nn.LayerNorm((C_out,))

    def forward(self, x):
        self.rnn.flatten_parameters()
        # (N,C,L) -> (L,N,C)
        x = x.permute(2,0,1)
        x , _ = self.rnn(x)
        # (L,N,C) -> (N,C,L)
        x = x.permute(1,2,0)
        # print("before x size",x.size())
        x= F.layer_norm(x,list(x.size()[1:]))
        # print("after x size",x.size())
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride] * 0.

class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(C_in, C_out, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv1d(C_in, C_out, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x) 
        # print("x size", x.size())
        a = self.conv1(x)
        b = self.conv2(x[:, :, 1:])
        print("a size,",a.size())
        print("b size,",b.size())
        out = torch.cat([a,b], dim=-1)
        print("out size,",out.size())
        # concat each matrix
        # out = torch.cat([a,b], dim=-1)
        out = self.bn(out)
        return out

class Highway(nn.Module):
    def __init__(self, layer_num: int, size: int):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])
        self.dropout = 0.1

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=self.dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            #x = F.relu(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, D, Nh=1, use_bias=False):
        super().__init__()
        self.mem_conv = Initialized_Conv1d(D, D*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(D, D, kernel_size=1, relu=False, bias=False)
        self.bias = None
        if use_bias:
            bias = torch.empty(1)
            nn.init.constant_(bias, 0)
            self.bias = nn.Parameter(bias)
        self.Nh = Nh
        self.D = D
        self.dropout = 0.1

    def forward(self, queries, mask=None):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.Nh)
        K, V = [self.split_last_dim(tensor, self.Nh) for tensor in torch.split(memory, self.D, dim=2)]

        key_depth_per_head = self.D // self.Nh
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask = mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)

    def dot_product_attention(self, q, k , v, mask = None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q,k.permute(0,1,3,2))
        if self.bias is not None:
            logits += self.bias
        if mask is not None:
            shapes = [x  if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)
    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class Stem4OneInput(nn.Module):
    def __init__(self, word_mat, char_mat, Dword, Dchar, Dout, use_char=False):
        super().__init__()
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_mat), freeze=True)
        self.use_char = use_char
        if self.use_char:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_mat), freeze=False)
            self.emb = Embedding(Dword, Dchar, Dout)
        else:
            self.emb = nn.Linear(Dword, Dout)

    def forward(self, Wid, Cid=None):
        if self.use_char:
            Cw, Cc = self.word_emb(Wid), self.char_emb(Cid)
            x = self.emb(Cc, Cw)
        else:
            Cw = self.word_emb(Wid)
            x = self.emb(Cw).permute([0,2,1])
        return x


class Stem4Bert(nn.Module):
    def __init__(self, word_mat, char_mat, Dword, Dchar, Dout):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = BertModel.from_pretrained("uncased_L-12_H-768_A-12")
        self.bert_model.to(device)
        for param in self.bert_model.parameters():param.requires_grad = False
        print("Word Embeddings dim",Dword)
        self.conv1d = Initialized_Conv1d(Dword, Dout, bias=False)
        
    def forward(self, input_ids, input_mask):
        self.bert_model.eval()
        all_encoder_layers, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask)
        out = self.conv1d(all_encoder_layers[-1].permute(0,2,1))
        return out

# Using passage and question to obtain question-aware passage representation
# Co-attention
class PQMatcher(nn.Module):
    def __init__(self, in_size):
        super(PQMatcher, self).__init__()
        self.hidden_size = hidden_size * 2
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size*2, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wq = nn.Linear(self.in_size*2, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wg = nn.Linear(self.in_size*4, self.in_size*4, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, up, uq):
        (lp, _, _) = up.size()
        (lq, _, _) = uq.size()
        mixerp, mixerq = torch.arange(lp).long().to(device), torch.arange(lq).long().to(device)
        Up = torch.cat([up, up[mixerp]], dim=2)
        Uq = torch.cat([uq, uq[mixerq]], dim=2)
        vs = torch.zeros(lp, batch_size, self.out_size).to(device)
        v = torch.randn(batch_size, self.hidden_size).to(device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(device)
        
        Uq_ = Uq.permute([1, 0, 2])
        for i in range(lp):
            Wup = self.Wp(Up[i])
            Wuq = self.Wq(Uq)
            Wvv = self.Wv(v)
            x = F.tanh(Wup + Wuq + Wvv).permute([1, 0, 2])
            s = torch.bmm(x, V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, Uq_).squeeze()
            r = torch.cat([Up[i], c], dim=1)
            g = F.sigmoid(self.Wg(r))
            r_ = torch.mul(g, r)
            c_ = r_[:, self.in_size*2:]
            v = self.gru(c_, v)
            vs[i] = v
            del Wup, Wuq, Wvv, x, a, s, c, g, r, r_, c_
        del up, uq, Up, Uq, Uq_
        vs = self.dropout(vs)
        return vs

# Input is question-aware passage representation
# Output is self-attention question-aware passage representation
class SelfMatcher(nn.Module):
    def __init__(self, in_size):
        super(SelfMatcher, self).__init__()
        self.hidden_size = in_size
        self.in_size = in_size
        self.gru = nn.GRUCell(input_size=in_size, hidden_size=self.hidden_size)
        self.Wp = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.Wp_ = nn.Linear(self.in_size, self.hidden_size, bias=False)
        self.out_size = self.hidden_size
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, v):
        (l, _, _) = v.size()
        h = torch.randn(batch_size, self.hidden_size).to(device)
        V = torch.randn(batch_size, self.hidden_size, 1).to(device)
        hs = torch.zeros(l, batch_size, self.out_size).to(device)
        
        for i in range(l):
            Wpv = self.Wp(v[i])
            Wpv_ = self.Wp_(v)
            x = F.tanh(Wpv + Wpv_)
            x = x.permute([1, 0, 2])
            s = torch.bmm(x, V)
            s = torch.squeeze(s, 2)
            a = F.softmax(s, 1).unsqueeze(1)
            c = torch.bmm(a, v.permute([1, 0, 2])).squeeze()
            h = self.gru(c, h)
            hs[i] = h
            logger.gpu_mem_log("SelfMatcher {:002d}".format(i), ['x', 'Wpv', 'Wpv_', 's', 'c', 'hs'], [x.data, Wpv.data, Wpv_.data, s.data, c.data, hs.data])
            del Wpv, Wpv_, x, s, a, c
        hs = self.dropout(hs)
        del h, v
        return hs