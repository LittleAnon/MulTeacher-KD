import math

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions, \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import find_pruneable_heads_and_indices, prune_linear_layer, apply_chunking_to_forward, \
    ModuleUtilsMixin
from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaEmbeddings

from teacherkd.student_model import embeddings


class SelfAttention(nn.Module):
    def __init__(self,hidden_size, num_attention_heads=12,attention_probs_dropout_prob=0.1):
        super().__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class SelfOutput(nn.Module):
    def __init__(self,hidden_size,layer_norm_eps: float = 1e-5,hidden_dropout_prob=0.1 ):
        '''

        :param hidden_size:
        :param layer_norm_eps: a value added to the denominator for numerical stability
        :param hidden_dropout_prob:
        '''
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class Attention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads):
        super().__init__()
        self.self = SelfAttention(hidden_size,num_attention_heads)
        self.output = SelfOutput(hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Intermediate(nn.Module):
    def __init__(self,hidden_size,intermediate_size,hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self,hidden_size,intermediate_size,layer_norm_eps:float = 1e-5,hidden_dropout_prob:float = 0.1 ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class EncoderLayer(nn.Module):
    def __init__(self,hidden_size,num_attention_heads,intermediate_size,hidden_act, chunk_size_feed_forward=0):
        super().__init__()
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Attention(hidden_size,num_attention_heads)
        self.is_decoder = False
        self.add_cross_attention = False
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = Attention(hidden_size,num_attention_heads)
        self.intermediate = Intermediate(hidden_size,intermediate_size,hidden_act)
        self.output = Output(hidden_size,intermediate_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output




class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Encoder(nn.Module):
    def __init__(self,
                 num_hidden_layers,
                 hidden_size,
                 num_attention_heads,
                 intermediate_size,
                 hidden_act,
                 gradient_checkpointing=False,
                 add_cross_attention=False):
        '''

        :param num_hidden_layers:(:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        :param hidden_size:(:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        :param num_attention_heads:(:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        :param intermediate_size:(:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        :param hidden_act:(:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        :param gradient_checkpointing:(:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        :param add_cross_attention:(:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the `:class:~transformers.EncoderDecoderModel` class, which
            consists of all models in ``AUTO_MODELS_FOR_CAUSAL_LM``.
        '''
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.add_cross_attention = add_cross_attention
        self.layer = nn.ModuleList([EncoderLayer(hidden_size,num_attention_heads,intermediate_size,hidden_act) for _ in range(num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.add_cross_attention else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )





class AugmentTransform(nn.Module, ModuleUtilsMixin):
    def __init__(self, config,n_classes,output_mode,add_pooling_layer=True):
        super(AugmentTransform, self).__init__()

        if config.teacher_type == 'gpt2':
            self.stem = embeddings.Gpt2Embeddings(config.gpt_config)
        elif config.teacher_type == 'bert':
            self.stem = embeddings.BertEmbeddings(config.bert_config)
        elif config.teacher_type == 'roberta':
            self.stem = embeddings.RobertaEmbeddings(config.roberta_config)
        elif config.teacher_type == 'rAg':
            self.stem = embeddings.RobertaEmbeddings(config.roberta_config)

        self.num_hidden_layers = 4
        self.encoder = Encoder(self.num_hidden_layers,config.hidden_size,num_attention_heads=12,intermediate_size=3072,hidden_act='gelu')
        self.pooler = Pooler(hidden_size=config.hidden_size) if add_pooling_layer else None
        self.config = config
        self.config.is_decoder = False
        self.output_mode = output_mode
        self.n_classes = n_classes
        self.all_return = config.use_emd
        self.linear = nn.Linear(config.hidden_size, self.n_classes)
        # self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input=None,

        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        output_attentions=False,
        # output_hidden_states=False,
        return_dict=False,
    ):

        input_ids, attention_mask, token_type_ids, seq_lengths = input
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)


        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)
        embedding_output = self.stem(
            input_ids=input_ids, token_type_ids=token_type_ids,need_permute=False
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=self.all_return,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        output_seq = pooled_output if pooled_output is not None else sequence_output
        logits = self.linear(output_seq)

        if self.all_return:
            student_layer_out = encoder_outputs[1:][0]
            return logits, student_layer_out
        else:
            return logits


        # if not return_dict:
        #     return (sequence_output, pooled_output) + encoder_outputs[1:]
        #
        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )




