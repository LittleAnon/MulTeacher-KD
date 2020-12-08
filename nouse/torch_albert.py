from transformers import AlbertTokenizer, AutoModelWithLMHead, pipeline

model = AutoModelWithLMHead.from_pretrained('albert-base-v2')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True,)
# fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
# x = tokenizer.tokenize("I love eat apple")
# print(tokenizer.convert_tokens_to_ids(x))
# print(model.config)
import torch
d = {}
for k, v in model.named_parameters():
    if 'embeddings' in k:
        d[k] = v
        print(k)
torch.save(d, 'al_emb.bin')
# for k, v in model.named_parameters():
#     print(k, v.shape)
# from transformers.modeling_albert import AlbertModel, load_tf_weights_in_albert
# from transformers.configuration_albert import AlbertConfig
# config = AlbertConfig()
# model = AlbertModel(config)
# model = load_tf_weights_in_albert(model, config, "./model")