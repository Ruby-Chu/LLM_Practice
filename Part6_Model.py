from typing import Optional, List, Any

from langchain_core.callbacks import CallbackManagerForLLMRun

from utils.config import get_openapi_key
import os

from langchain.prompts import PromptTemplate, ChatPromptTemplate

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.schema import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain.llms.base import LLM

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID


# class just_a_cat(LLM):
#
#     def _call(self, prompt: str, stop: Optional[List[str]] = None,
#               run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
#         return "Meow~"
#         # 說幾個字就有多少Meow
#         # return "Meow~" * int(len(prompt) / 3 + 1)
#
#     @property
#     def _llm_type(self) -> str:
#         return "我是一隻貓"

class just_a_cat(LLM):

    def _call(self, prompt, stop=None):
        # 說幾個字就有多少Meow
        return "Meow~" * int(len(prompt) / 3 + 1)

    def _llm_type(self):
        return "我是一隻貓"


if __name__ == "__main__":
    llm = OpenAI()
    chat_model = ChatOpenAI()

    print("-------------LLM-------------")
    result1 = llm("台北有哪些月老廟?")
    print(result1)
    print("-----------------------------")
    result2 = llm.invoke("台北哪間月老廟靈驗?")
    print(result2)
    print("-----------------------------")
    # 透過調整 temperature 參數為 0，可以使 llm 機率不發散，得到相同的結果。
    result3 = llm.invoke("台北有哪些文昌廟?", temperature=0)
    print(result3)
    print("-----------------------------")
    print("----------Chat Model---------")
    text = "請提供Python 爬蟲課程目錄"
    messages1 =[HumanMessage(content=text)]
    result4 = chat_model.invoke(messages1, temperature=0)
    print(result4.content)
    print("-----------------------------")
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system","你是一個{job}的專家"),
            ("human", "你是誰?"),
            ("ai", "您好! 這裡是Siri，是一個人類的好幫手?"),
            ("human", "{question}")
        ]
    )
    messages2 = chat_template.format_messages(job="Python", question="請提供Python 爬蟲課程目錄")
    print(messages2)
    result5 = chat_model.invoke(messages2, temperature=0)
    print(result5.content)

    # Batch Generate
    print("-----------------------------")
    print("--------Batch Generate-------")
    result6 = llm.invoke("請問今天禮拜幾")
    print(result6)
    print("-----------------------------")
    result7 = llm.generate(["請問今天星期幾?", "請問明年的農曆過年是國曆的幾月幾號?"])
    generations = result7.generations
    for i in range(len(generations)):
        print(generations[i][0].text)
    print("-----------------------------")

    # Define LLM's module
    print("-----Define LLM's module-----")
    llm_cat = just_a_cat()
    print(llm_cat("你好"))
    print(llm_cat("你好嗎?"))
    print(llm_cat("你好嗎~!?"))
    print(llm_cat._llm_type())
    print("-----------------------------")
    # Chain structure
    prompt_template = PromptTemplate.from_template("請告訴我{day}的日期")
    message = ChatPromptTemplate.from_template("請告訴我{day}的日期")

    chain1 = prompt_template | llm
    ans1 = chain1.invoke({"day": "明年中秋節"})
    print(ans1)

    chain2 = message | llm
    ans2 = chain2.invoke({"day": "明年中秋節"})
    print(ans2)

    chain3 = prompt_template | chat_model
    ans3 = chain3.invoke({"day": "明年中秋節"})
    print(ans3.content)

    chain4 = message | chat_model
    ans4 = chain4.invoke({"day": "明年中秋節"})
    print(ans4.content)


    # prompt_template = PromptTemplate(template="請告訴我今天的日期", input_variables=[])
    # # prompt_template.format()
    # chain = prompt_template | llm
    # print(chain.invoke({}))

