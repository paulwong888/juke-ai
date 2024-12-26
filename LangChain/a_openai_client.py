from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

class OpenAiClient():
    def __init__(self):
        load_dotenv()
        model = "gpt-4o-mini"
        self.openai_client = ChatOpenAI(model=model)
        self.openai_client_with_output_format = ChatOpenAI(model=model)
        self.openai_client_with_tool = ChatOpenAI(model=model).bind_tools([self.add, self.mutiple])
        self.messages = [
            SystemMessage("你是聚客AI研究院的课程助理。")
        ]
    
    @tool
    def mutiple(a: int, b: int) -> int:
        """Multiply two integers.
        Args:
            a: first arg
            b: scond arg
        """
        return a * b
    
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers.
        Args:
            a: first arg
            b: second arg
        """
        return a + b

    def build_chat_chain(self, template: str):
        chat_prompt_template = ChatPromptTemplate.from_template(template)
        chat_chain = (chat_prompt_template | self.openai_client | StrOutputParser())
        return chat_chain

    def build_chain_with_history_persit(self, template: str):
        return RunnableWithMessageHistory(
            runnable = self.build_chat_chain(template),
            get_session_history = lambda session_id: SQLChatMessageHistory(session_id=session_id, connection_string="sqlite:///memory.db")
        )

    def chat(self, message: str):
        response = self.openai_client.invoke(message)
        return response.content
    
    def chat_with_history(self, message: str):
        self.messages.append(HumanMessage(message))
        response = self.openai_client.invoke(self.messages).content
        self.messages.append(AIMessage(response))
        return response
    
    def chat_with_template(self, message: str):
        template = ChatPromptTemplate.from_template("给我讲个关于{subject}的笑话")
        response = self.openai_client.invoke(template.format(subject=message)).content
        # response = self.openai_client.invoke(template.format_prompt(subject=message)).content
        # response = self.openai_client.invoke(template.invoke({"subject": "message"})).content
        return response
    
    def chat_with_template_file(self, message: str):
        chat_template = PromptTemplate.from_file("LangChain/example_prompt_template.txt")
        response = self.openai_client.invoke(chat_template.format(topic=message)).content
        return response
    
    def chat_with_output_object(self, message: str, output_format):
        return self.openai_client_with_output_format \
                    .with_structured_output(output_format) \
                    .invoke(message)
    
    def chat_with_output_json(self, template: str, input_vars: list, output_class, **kwargs):
        json_parser = JsonOutputParser(pydantic_object=output_class)
        prompt_template = PromptTemplate(
            template=template, input_variables=input_vars,
            partial_variables={"format_instructions": json_parser.get_format_instructions()}
        )
        # print(kwargs)
        response = self.openai_client.invoke(prompt_template.invoke(kwargs))
        print(response.content)
        print(response)
        return response.content
    
    def chat_with_chain(self, template: str, message_dict: dict):
        # chat_prompt_template = ChatPromptTemplate.from_template(template)
        # chain = (chat_prompt_template | self.openai_client | StrOutputParser())
        chain = self.build_chat_chain(template)
        return chain.invoke(message_dict)
    
    def chat_with_tool(self, messag: str):
        messages = [HumanMessage(messag)]
        response = self.openai_client_with_tool.invoke(messag)
        print(response)

        messages.append(response)

        availabe_tools = {"add": self.add, "mutiple": self.mutiple}

        for tool_call in response.tool_calls:
            selected_tool = availabe_tools[tool_call["name"]]
            tool_result = selected_tool.invoke(tool_call)
            messages.append(tool_result)

        response = self.openai_client_with_tool.invoke(messages)
        print(response)

        return response.content
    
    def chat_with_history_persit(self, session_id: str, template: str, **message_kw):
        chain_with_history_persit = self.build_chain_with_history_persit(template)
        return chain_with_history_persit.invoke(
            message_kw,
            config={"configurable": {"session_id": session_id}}
        )

def test_chat():
    openai_client = OpenAiClient()
    print(openai_client.chat("你是谁"))
    print(openai_client.chat("我是学员，我叫大拿。"))
    print(openai_client.chat("我是谁"))

def test_chat_with_history():
    openai_client = OpenAiClient()
    print("=============")
    print(openai_client.chat_with_history("我是学员，我叫大拿。"))
    print(openai_client.chat_with_history("我是谁"))

def test_chat_with_template():
    openai_client = OpenAiClient()
    print(openai_client.chat_with_template("小明"))

def test_chat_with_template_file():
    openai_client = OpenAiClient()
    print(openai_client.chat_with_template_file("黑色幽默"))

def test_chat_with_output_object():
    from b_date import Date
    openai_client = OpenAiClient()
    template = """提取用户输入中的日期。
    用户输入:
    {query}"""
    message = ChatPromptTemplate.from_template(template).format(query="2024年十二月23日天气晴...")
    print(openai_client.chat_with_output_object(message, Date))

def test_chat_with_output_json():
    from b_date import Date
    from langchain_core.output_parsers import JsonOutputParser
    
    json_output_parser = JsonOutputParser(pydantic_object=Date)

    openai_client = OpenAiClient()
    template = """提取用户输入中的日期。
    用户输入:
    {query}
    {format_instructions}"""
    prompt_template = PromptTemplate(
        template=template, input_variables=["query"],
        partial_variables={"format_instructions": json_output_parser.get_format_instructions()}
    )
    message = prompt_template.format_prompt(query="2024年十二月23日天气晴...")
    # print(openai_client.chat(message))
    print(openai_client.chat_with_output_json(template, ["query"], Date, query="2024年十二月23日天气晴..."))

def test_chat_with_tool():
    openai_client = OpenAiClient()
    print(openai_client.chat_with_tool("3的4倍是多少?"))

def test_chat_with_chain():
    openai_client = OpenAiClient()
    template = "给我讲个关于{subject}的笑话"
    message_dict = {"subject": "小明"}
    print(openai_client.chat_with_chain(template, message_dict))

def test_chat_with_history_persit():
    openai_client = OpenAiClient()
    session_id = "paul"
    template = "你好，我叫{name}"
    # message_dict = {"name": "大拿"}
    # print(openai_client.chat_with_history_persit(session_id, template, name="大拿"))
    template = "{who}叫什么名字"
    print(openai_client.chat_with_history_persit(session_id, template, who="我"))


if __name__ =="__main__":
    # test_chat_with_template()
    # test_chat_with_template_file()
    # test_chat_with_output_object()
    # test_chat_with_output_json()
    # test_chat_with_tool()
    # test_chat_with_chain()
    test_chat_with_history_persit()