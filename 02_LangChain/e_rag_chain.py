from d_vector_db import VectorDb
from a_openai_client import OpenAiClient
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class RagChain():
    def __init__(self, file_name: str):
        vector_db = VectorDb(file_name)
        retriever = vector_db.retriever
        openai_client = OpenAiClient()
        template = """Answer the question only base on the following context:
        {context}
        Question:
        {question}
        """
        chat_prompt_template = ChatPromptTemplate.from_template(template)

        self.rag_chain = (
            {"question": RunnablePassthrough(), "context": retriever} |
            chat_prompt_template |
            openai_client.openai_client |
            StrOutputParser()
        )

if __name__ == "__main__":
    rag_chain = RagChain("LangChain/llama2.pdf")

    #直接输出
    print(rag_chain.rag_chain.invoke("llama2有多少参数"))
    print()
    
    #流式输出
    for s in rag_chain.rag_chain.stream("llama2有多少参数"):
        print(s, end="", flush=True)