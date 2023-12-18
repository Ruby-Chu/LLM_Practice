from utils.config import get_openapi_key
import os

# import prompts module
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

# import models module
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# import parser module
from langchain.schema import StrOutputParser

# import Chain
from langchain.chains import LLMChain

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from langchain.schema.runnable import RunnableBranch
from typing import Literal

from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.pydantic_v1 import BaseModel
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

from operator import itemgetter

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
# 不用 Router 如下：
from langchain.schema import HumanMessage

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID


class TopicClassifier(BaseModel):
    "請分類問題為數學、物理、一般助手其中一種類型"

    title: Literal["Python"]


if __name__ == "__main__":
    # 數學
    python_template = """你是一位非常厲害的Python工程師，很擅長編寫程式。
    你能夠將難題分解成各個小問題部分，回答小問題後將它們組合起來回答更廣泛的問題。

    問題如下:
    {input}"""
    python_prompt = PromptTemplate.from_template(python_template)

    # 物理
    cshape_template = """你是一位非常厲害C#工程師，很擅長編寫程式。
    你擅長以簡潔易懂的方式回答有關物理的問題。當你不知道某個問題的答案時，你就承認你不知道。

    問題如下:
    {input}"""
    cshape_prompt = PromptTemplate.from_template(cshape_template)

    # 一般助手
    general_template = """你是一個多功能型的軟體工程師，請盡可能準確地回答問題。

    問題如下:
    {input}"""
    general_prompt = PromptTemplate.from_template(general_template)

    prompt_branch = RunnableBranch(
        (lambda x: x["title"] == "python", python_prompt),
        (lambda x: x["title"] == "cshape", cshape_prompt),
        general_prompt,
    )

    # 分類
    classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
    model = ChatOpenAI().bind(
        functions=[classifier_function], function_call={"name": "TopicClassifier"}
    )
    parser = PydanticAttrOutputFunctionsParser(
        pydantic_schema=TopicClassifier, attr_name="title"
    )
    classifier_chain = model | parser

    # 分類問題
    router_chain = (
            RunnablePassthrough.assign(title=itemgetter("input") | classifier_chain)
            | prompt_branch  # 可註解
            | ChatOpenAI()  # 可註解
            | StrOutputParser()  # 可註解
    )

    ans1 = router_chain.invoke(
        {
            "input": "Python爬蟲被cloudflare擋住，請問如何解決該問題?"
        }
    )
    print(ans1)

    model = ChatOpenAI()
    ans2 = model([HumanMessage(content="Python爬蟲被cloudflare擋住，請問如何解決該問題?")])
    print(ans2)
