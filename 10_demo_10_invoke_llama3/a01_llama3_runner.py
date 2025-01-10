import torch
from dotenv import load_dotenv
from a00_constant import LLAMA3_REPO_ID, QWEN_REPO_ID
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

class Llama3Runner():
    def __init__(self):
        load_dotenv()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = QWEN_REPO_ID
        # model_name = LLAMA3_REPO_ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", 
            # load_in_4bit=True,
            load_in_8bit=True,
        )

    def predict(self, message: str):
        # {"role": "system", "content": "You are a helpful assistant system."},
        mesage = [
            {"role": "system", "content": "You are an assistant who provides precise and direct answers."},
            {"role": "user", "content": message},
        ]
        message = self.tokenizer.apply_chat_template(
            mesage, tokenize=False, add_generation_prompt=True
        )
        input = self.tokenizer([message], return_tensors="pt").to(self.device)

        # model_input = self.tokenizer(message, return_tensors="pt").to(self.device)

        output = self.model.generate(**input, max_new_tokens=512)
        # output = self.model.generate(**model_input)
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return response
    
if __name__ == "__main__":
    llama3_runner = Llama3Runner()
    # print(llama3_runner.predict("请介绍下香港演员周星驰的主要作品"))
    print(llama3_runner.predict("请介绍下周星驰的主要作品"))
    """
    Qwen/Qwen2.5-7B-Instruct
    print(llama3_runner.predict("请介绍下周星驰的主要作品"))
    ['system\nYou are an assistant who provides precise and direct answers.\n
      user\n请介绍下周星驰的主要作品\n
      assistant\n您提到的“星驰”可能是指电影《无间道》系列中的角色“星']
    ['system\nYou are an assistant who provides precise and direct answers.\n
      user\n请介绍下周星驰的主要作品\n
      assistant\n您提到的“星驰”可能是指周星驰，但目前没有具体信息显示周']
    
    # print(llama3_runner.predict("请介绍下香港演员周星驰的主要作品"))
    ['system\nYou are an assistant who provides precise and direct answers.\nuser\n请介绍下周星驰的主要作品\n
      assistant\n您好，但您提到的“星驰”可能指的是不同的对象。在娱乐界，“星驰”有时用来指香港著名导演周星驰。如果您是想了解周星驰导演的作品，以下是下周（2023年1月9日至15日）的相关信息：\n\n1. **电影《美人鱼2》**：这是周星驰执导的一部喜剧片，预计在2023年春节期间上映。虽然目前没有具体的消息表明该片会在下周进行大规模宣传或上映，但作为一部备受期待的大制作，可能会有预告片发布、海报更新等宣传活动。\n\n2. **其他活动**：除了上述电影之外，周星驰本人或其团队可能会参与一些宣传活动，如见面 会、媒体采访等，但具体安排需关注官方渠道发布的消息。\n\n请注意，以上信息基于当前可获得的信息进行推测，实际情况可能会有所变化。建议关注相关官方渠道获取最新、最准确的信息。如果您指的是其他“星驰”，请提供更多信息以便我能更准确地回答您的问题。']
    """