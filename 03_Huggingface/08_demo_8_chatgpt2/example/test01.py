#中文白话文文章生成
import torch
from transformers import GPT2LMHeadModel,BertTokenizer,TextGenerationPipeline

# 加载模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(model)

#使用Pipeline调用模型
text_generator = TextGenerationPipeline(model,tokenizer,device=device)

#使用text_generator生成文本
#do_sample是否进行随机采样。为True时，每次生成的结果都不一样；为False时，每次生成的结果都是相同的。
for i in range(3):
    print(text_generator("这是很久之前的事情了,", max_length=100, do_sample=True))
    print()