import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv, dotenv_values
from c_embedding_util import EmbeddingUtil

class OpenAiClient():
    def __init__(self):
        load_dotenv()
        print(dotenv_values())
        print(os.getenv("OPENAI_API_KEY"))
        self.openai_client = OpenAI()

    def build_prompt(self, **kwargs):
        prompt_template = """
        你是一个问答机器人。
        你的任务是根据下述给定的已知信息回答用户问题。

        已知信息:
        {context} # 检索出来的原始文档

        用户问：
        {query} # 用户的提问

        如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
        请不要输出已知信息中不包含的信息或答案。
        请用中文回答用户问题。
        """
        inputs = {}
        for k, v in kwargs.items():
            if isinstance(v, list) and all(isinstance(item, str)for item in v):
                v = "\n\n".join(v)
            inputs[k] = v
        
        return prompt_template.format(**inputs)

    def get_completion(self, prompt:str, model="gpt-4o"):
        messages = [{"role": "user", "content": prompt}]
        response = self.openai_client.chat.completions.create(
            messages=messages, model=model, temperature=0
        )
        return response.choices[0].message.content
    
    def get_embeddings(self, text, model="text-embedding-ada-002", dimensions=None):
        # if model == "text-embedding-3-small":
        if model == "text-embedding-ada-002":
            dimensions = None
        if dimensions:
            data = self.openai_client.embeddings.create(
                input=text, model=model, dimensions=dimensions
            ).data
        else:
            print(type(text))
            response = self.openai_client.embeddings.create(
                input=text, model=model
            )
            # print(response.data)
        return [item.embedding for item in response.data]

def test_completion():    
    openai_client = OpenAiClient()
    print(openai_client.get_completion("How do you do"))

def test_embeddings():
    openai_client = OpenAiClient()
    query = ["测试文本"]
    embeddings = openai_client.get_embeddings(query)[0]
    print(type(embeddings))
    print(len(embeddings))
    print(embeddings[:10])

def test_embeddings2():
    query = ["国际争端"]

    # 且能支持跨语言
    # query = "global conflicts"

    documents = [
        "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
        "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
        "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
        "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
        "我国首次在空间站开展舱外辐射生物学暴露实验",
    ]

    openai_client = OpenAiClient()
    query_vec = openai_client.get_embeddings(query)[0]
    doc_vecs = openai_client.get_embeddings(documents)

    embedding_util = EmbeddingUtil()
    print("Query与自己的余弦距离: {:.2f}".format(embedding_util.cos_sim(query_vec, query_vec)))
    print("Query与Documents的余弦距离:")
    for vec in doc_vecs:
        print(embedding_util.cos_sim(query_vec, vec))

    print()

    print("Query与自己的欧氏距离: {:.2f}".format(embedding_util.l2(query_vec, query_vec)))
    print("Query与Documents的欧氏距离:")
    for vec in doc_vecs:
        print(embedding_util.l2(query_vec, vec))

if __name__ == "__main__":
    # test_completion()
    test_embeddings()
    # test_embeddings2()