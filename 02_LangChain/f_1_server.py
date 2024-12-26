# ```python
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from a_openai_client import OpenAiClient
from langchain_openai import OpenAI

fast_api = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
openai_client = OpenAiClient()
add_routes(
    app=fast_api,
    runnable=openai_client.build_chat_chain("讲一个关于{topic}的笑话"),
    # runnable=OpenAI(),
    path="/joke"
)

if __name__ == "__main__":
    uvicorn.run(app=fast_api, host="localhost", port=9999)
# ```