import chromadb
from chromadb.config import Settings
from b_openai_client import OpenAiClient
from d_my_embedding_fun import MyEmbeddingFunction

class MyVectorDBConnector():
    def __init__(self, collection_name, my_embedding_function: MyEmbeddingFunction = MyEmbeddingFunction()):
        chromadb_client = chromadb.Client(Settings(allow_reset=True))
        # self.embedding_fn = embedding_fn
        self.my_embedding_function = my_embedding_function
        chromadb_client.reset()
        self.collection = chromadb_client.create_collection(name=collection_name, embedding_function=my_embedding_function)

    def add_documents(self, documents):
        self.collection.add(
            embeddings=self.my_embedding_function(documents),
            documents=documents,
            ids=[f"id{id}" for id in range(len(documents))]
        )

    def search(self, query:str, top_n):
        return self.collection.query(
            query_embeddings=self.my_embedding_function([query]),
            n_results=top_n
        )
    
    def init_dummy_data(self):
        from a_pdf_splitter import PdfSplitter
        from b_openai_client import OpenAiClient
        # 为了演示方便，我们只取两页（第一章）
        pdf_splitter = PdfSplitter()
        paragraphs = pdf_splitter.extract_text_from_pdf(
            "RAG-Embeddings/llama2.pdf",
            page_numbers=[2, 3],
            min_line_length=10
        )
        self.add_documents(paragraphs)

    
if __name__ == "__main__":
    my_embedding_function = MyEmbeddingFunction()
    my_vectordb_connector = MyVectorDBConnector("test", my_embedding_function)
    my_vectordb_connector.init_dummy_data()
    user_query = "Llama 2有多少参数"
    result = my_vectordb_connector.search(user_query, 2)
    for item in result["documents"][0]:
        print(item)