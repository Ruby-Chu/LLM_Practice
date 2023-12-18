from utils.config import get_openapi_key
import os

# from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
# from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

from langchain.schema import Document
from langchain.schema.prompt_template import format_document

from functools import partial
# import prompts module
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

# import models module
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# import parser module
from langchain.schema import StrOutputParser

# import Chain
# from langchain.chains import LLMChain

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()

    # stuff
    document_prompt = PromptTemplate.from_template("{page_content}")

    # Define LLM chain (Use LCEL)
    prompt_template = PromptTemplate.from_template("請總結以下文本:\n\n{文本內容}")
    stuff_chain = (
            {
                "文本內容": lambda docs: "\n\n".join(format_document(doc, document_prompt) for doc in docs)
            }
            | prompt_template
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
    )
    print(stuff_chain.invoke(docs))

    #
    # # refine
    # document_prompt = PromptTemplate.from_template("{page_content}")
    # partial_format_document = partial(format_document, prompt=document_prompt)
    #
    # first_prompt = PromptTemplate.from_template("請總結以下文本:\n\n{文本內容}")
    # summary_chain = {"文本內容": partial_format_document} | first_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    # summary = summary_chain.invoke(docs[0])
    #
    # refine_prompt = PromptTemplate.from_template("以下是之前的總結內容: {總結}.\n\n請與下面新的內容作整理: {新文本}")
    # refine_chain = (
    #         {
    #             "總結": lambda x: x["總結"],
    #             "新文本": lambda x: partial_format_document(x["文本"]),
    #         }
    #         | refine_prompt
    #         | ChatOpenAI(temperature=0)
    #         | StrOutputParser()
    # )
    # for i, doc in enumerate(docs[1:]):
    #     summary = refine_chain.invoke({"總結": summary, "文本": doc})
    #     print(f'第 {i} 次總結內容為：{summary}\n\n')
