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

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

# 不用 Router 如下：
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID


class TopicClassifier(BaseModel):
    "請分類問題為數學、物理、一般助手其中一種類型"

    title: Literal["Python"]


if __name__ == "__main__":
    synopsis_prompt = PromptTemplate.from_template(
        """你是一位劇本家，你可以為給定的一個標題寫一個故事概要。

    標題: {title}
    劇本概要: """
    )

    review_prompt = PromptTemplate.from_template(
        """您是知名的台灣劇本評論家，能為戲劇概要撰寫評論。

    劇本概要:
    {synopsis}

    評論: """
    )

    sequential_chain = (
            {"synopsis": synopsis_prompt | ChatOpenAI() | StrOutputParser()}
            | review_prompt
            | ChatOpenAI()
            | StrOutputParser()
    )
    print(sequential_chain.invoke({"title": "藍色瞳鈴眼"}))
    print('---------------\n')
    synopsis_chain = synopsis_prompt | ChatOpenAI() | StrOutputParser()
    review_chain = review_prompt | ChatOpenAI() | StrOutputParser()
    sequential_chain = {"synopsis": synopsis_chain} | RunnablePassthrough.assign(評論=review_chain)
    print(sequential_chain.invoke({"title": "犀利人妻"}))

    # prompt_template = PromptTemplate.from_template("""總結以下文章:
    #
    # {text}
    #
    # 總結:"""
    #                                                )
    #
    # with open("story.txt", encoding='utf-8') as f:
    #     story = f.read()
    #
    # transformation_chain = (
    #         {"text": lambda text: "\n\n".join(text.split("\n\n")[:3])}
    #         | prompt_template
    #         | ChatOpenAI()
    #         | StrOutputParser()
    # )
    # print(transformation_chain.invoke(story))
