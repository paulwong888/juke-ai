from chromadb import EmbeddingFunction
from b_openai_client import OpenAiClient

class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        super(MyEmbeddingFunction, self).__init__()
        self.openai_client = OpenAiClient()

    def __call__(self, input: list):
        return self.openai_client.get_embeddings(input)