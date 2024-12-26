from c_pdf_splitter import PdfSpiltter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class VectorDb():
    def __init__(self, file_name: str, embedding_model: str = "text-embedding-ada-002"):
        pdf_splitter = PdfSpiltter()
        document_list = pdf_splitter.split(file_name, 300, 100)
        embeddings = OpenAIEmbeddings(model = embedding_model)
        faiss = FAISS.from_documents(document_list, embeddings)
        self.retriever = faiss.as_retriever(search_kwargs={"k": 3})

    def search(self, query: str):
        return self.retriever.invoke(query)
    
if __name__ == "__main__":
    vector_db = VectorDb("LangChain/llama2.pdf")
    document_list = vector_db.search("llama2有多少参数")

    for document in document_list:
        print(document.page_content)
        print('-------')