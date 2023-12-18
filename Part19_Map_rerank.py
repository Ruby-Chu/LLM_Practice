from utils.config import get_openapi_key
import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser

from langchain.schema import Document
from langchain.schema.prompt_template import format_document

from langchain.pydantic_v1 import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.document_loaders import WebBaseLoader

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID


class AnswerAndScore(BaseModel):
    """Return the answer to the question and a relevance score."""

    answer: str = Field(
        description="問題的答案僅基於所提供的上下文。"
    )
    score: float = Field(
        description="0.0-1.0 的相關性得分，其中 1.0 表示所提供的上下文完全回答了問題，0.0 表示所提供的上下文根本沒有回答問題。"
    )


def top_answer(scored_answers):
    return max(scored_answers, key=lambda x: x.score).answer


if __name__ == "__main__":
    # with open("story.txt", encoding='utf-8') as f:
    #     story = f.read()
    #
    # docs = [
    #     Document(
    #         page_content=chunk,
    #         metadata={
    #             "source": "https://zh.wikipedia.org/zh-tw/%E7%88%B1%E4%B8%BD%E4%B8%9D%E6%A2%A6%E6%B8%B8%E4%BB%99%E5%A2%83"},
    #     )
    #     for chunk in story.split('\n\n')
    # ]
    loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B")
    docs = loader.load()
    print(docs)

    # docs = [
    #     Document(
    #         page_content=chunk,
    #         metadata={
    #             "source": "https://zh.wikipedia.org/wiki/%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B"},
    #     )
    #     for chunk in story.split('\n\n')
    # ]

    map_prompt = PromptTemplate.from_template(
        "參照以下文本回答問題。"
        "\n\文本:\n\n{文本}\n\n問題: {問題}"
    )

    function = convert_pydantic_to_openai_function(AnswerAndScore)
    map_chain = (
            map_prompt
            | ChatOpenAI().bind(
        temperature=0, functions=[function], function_call={"name": "AnswerAndScore"}
    )
            | PydanticOutputFunctionsParser(pydantic_schema=AnswerAndScore)
    )

    document_prompt = PromptTemplate.from_template("{page_content}")
    map_rerank_chain = (
            (
                lambda x: [
                    {
                        "文本": format_document(doc, document_prompt),
                        "問題": x["提問"],
                    }
                    for doc in x["文件"]
                ]
            )
            | map_chain.map()
            | top_answer
    )

    print(map_rerank_chain.invoke({"文件": docs, "提問": "「大型語言模型有哪些能力？」"}))
