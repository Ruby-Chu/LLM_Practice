from utils.config import get_openapi_key
import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser

from langchain.schema import Document
from langchain.schema.prompt_template import format_document

from functools import partial

from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID


def format_docs(docs):
    return "\n\n".join(partial_format_document(doc) for doc in docs)


if __name__ == "__main__":
    with open("story.txt", encoding='utf-8') as f:
        story = f.read()

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "source": "https://zh.wikipedia.org/zh-tw/%E7%88%B1%E4%B8%BD%E4%B8%9D%E6%A2%A6%E6%B8%B8%E4%BB%99%E5%A2%83"},
        )
        for chunk in story.split('\n\n')
    ]

    document_prompt = PromptTemplate.from_template("{page_content}")
    partial_format_document = partial(format_document, prompt=document_prompt)

    prompt_template = PromptTemplate.from_template("請在20字內總結以下文本:\n\n{文本}")
    summary_chain = (
            {"文本": partial_format_document}
            | prompt_template
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
    )

    # A wrapper chain to keep the original Document metadata
    map_chain = (
            RunnableParallel({"小結": summary_chain, "文件": RunnablePassthrough()})
            | (lambda x: Document(page_content=x["小結"], metadata=x["文件"].metadata))
    )

    reduce_prompt = PromptTemplate.from_template("合併以下總結:\n\n{文本}")
    reduce_chain = (
            {
                "文本": lambda docs: "\n\n".join(format_document(doc, document_prompt) for doc in docs)
            }
            | reduce_prompt
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
    )

    # The final full chain
    map_reduce_chain = (map_chain.map() | reduce_chain)

    print(map_reduce_chain.invoke(docs))
