#使用openai的代码风格调用ollama
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1/",api_key="ollama")

responce  =client.chat.completions.create(
    messages= [{"role":"user","content":"你好，你叫什么名字？你是由谁创造的？"}],model="llama3"
)

print(responce.choices[0])