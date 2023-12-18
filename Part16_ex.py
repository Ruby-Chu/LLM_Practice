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
        """你是一位Python專業工程師，你可以為給定的一個標題寫一個課程大綱。

    標題: {title}
    content: """
    )

    review_prompt = PromptTemplate.from_template(
        """您是專業的Python講師，能為課程大綱撰寫課程內容目錄。

    content:
    {synopsis}

    outline: """
    )

    sequential_chain = (
            {"synopsis": synopsis_prompt | ChatOpenAI() | StrOutputParser()}
            | review_prompt
            | ChatOpenAI()
            | StrOutputParser()
    )
    print(sequential_chain.invoke({"title": "Django Web 開發"}))

    synopsis_chain = synopsis_prompt | ChatOpenAI() | StrOutputParser()
    review_chain = review_prompt | ChatOpenAI() | StrOutputParser()
    sequential_chain = {"synopsis": synopsis_chain} | RunnablePassthrough.assign(outline=review_chain)
    print(sequential_chain.invoke({"title": "爬蟲"}))
    #
    # prompt_template = PromptTemplate.from_template("""課程大綱:
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
    # transformation_chain.invoke(story)
