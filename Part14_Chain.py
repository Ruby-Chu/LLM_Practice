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
from langchain.schema import StrOutputParser

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "您是一位專業的民調大師，對民調計算問題提供了準確且公正答案。",
            ),
            ("human", "{問題}"),
        ]
    )
    model = ChatOpenAI()
    parser = StrOutputParser()

    llm_chain = prompt_template | model | parser

    ans1 = llm_chain.invoke({"問題": "A和B兩人比民調，A說讓B民調3%是甚麼意思？"})
    print(ans1)

    prompt_template = PromptTemplate.from_template("請給我一個 {怎樣的} {產品}?")
    llm_chain = prompt_template | ChatOpenAI() | StrOutputParser()
    ans2 = llm_chain.invoke({"產品": "帽子", "怎樣的": "綠色的"})
    print(ans2)


