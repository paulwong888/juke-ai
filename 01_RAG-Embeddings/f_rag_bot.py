from e_my_vectordb_connector import MyVectorDBConnector
from b_openai_client import OpenAiClient

class RAGBot():
    def __init__(self, my_vectordb_connector: MyVectorDBConnector):
        self.my_vectordb_connector = my_vectordb_connector
        self.openai_client = OpenAiClient()
        
    def chat(self, user_query: str):
        # 1. 检索
        documents = self.my_vectordb_connector.search(user_query, 2)['documents'][0]

        # 2. 构建 Prompt
        prompt = self.openai_client.build_prompt(query = user_query, context = documents)

        # 3. 调用 LLM
        return self.openai_client.get_completion(prompt)

if __name__ == "__main__":
    my_vectordb_connector = MyVectorDBConnector("test")
    my_vectordb_connector.init_dummy_data()
    # user_query = "Llama 2有多少参数"
    user_query = "how safe is llama 2"

    rag_bot = RAGBot(my_vectordb_connector)
    print(rag_bot.chat(user_query))