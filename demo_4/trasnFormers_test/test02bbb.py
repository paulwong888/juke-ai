import torch
from transformers import BertTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2 import GPT2Model
from transformers.models.llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch import Tensor

# model_name_or_path = "uer/gpt2-chinese-cluecorpussmall"
model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(model_name_or_path)

print(type(model))
model.eval()

text = "Hello World"
input = tokenizer.encode(text, return_tensors="pt")
print(input)
# {'input_ids': tensor([[    1, 15043,  2787]]), 'attention_mask': tensor([[1, 1, 1]])}

with torch.no_grad():
    response = model.generate(**input)

print(type(response))
if isinstance(response, Tensor):
    print(response.shape)
# print(response[0])
print(tokenizer.decode(response[0], skip_special_tokens=True))
