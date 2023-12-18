from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.example_selector import LengthBasedExampleSelector
from typing import Dict, List, Any
# import numpy as np

from utils.config import get_openapi_key
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID


class CustomExampleSelector(BaseExampleSelector):

    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self):
        """Select which examples to use based on the inputs."""
        # 若沒有喵字就忽略
        # return [{list(i.items())[0][0]: list(i.items())[0][1]} for i in self.examples if '喵' in str(i.values())]
        return [{list(i.items())[0][0]: list(i.items())[0][1]} for i in self.examples]


if __name__ == "__main__":
    llm = OpenAI()
    chat_model = ChatOpenAI()

    # dict_examples = [
    #     {'question': '請問如何申請加班', 'answer': '請到EIP點選辦公室事務->加班管理'},
    #     {'question': '請問超過6點怎樣申請加班', 'answer': '若當日欲申請時間超6點要申請加班,請隔日即可申請前天加班'},
    #     {'question': '那裡可查同仁分機?', 'answer': '請到EIP上方點選我要找人'},
    # ]
    #
    # prompt_template1 = PromptTemplate.from_template(template="Q:{question}\nA:{answer}")
    # FewShot_prompt_template = FewShotPromptTemplate(
    #     examples=dict_examples,
    #     example_prompt=prompt_template1,
    #     suffix="Question:{input}",
    #     input_variables=["input"]
    # )
    # q_temp = FewShot_prompt_template.format(input='同仁分機在哪裡')
    # # prompt_template.format(**examples[0]) # 指示輸入examples的問題
    # print(llm.invoke(q_temp))
    # print("-----------------------------")
    # demo_examples = '''
    # 你好 : 你好，喵~'
    # 今天星期幾 : 今天星期六，喵~'
    # 午餐吃甚麼 : 吃便當，喵~'
    # {question} : '
    # '''
    #
    # example_prompt = PromptTemplate.from_template(template=demo_examples)
    # temp2 = example_prompt.format(question="午餐吃什麼好?")
    # print(llm.invoke(temp2))
    # print("-----------------------------")
    # dict_examples2 = [
    #     {'請問如何申請加班': '請到EIP點選辦公室事務->加班管理'},
    #     {'請問超過6點怎樣申請加班': '若當日欲申請時間超6點要申請加班,請隔日即可申請前天加班'},
    #     {'那裡可查同仁分機?': '請到EIP上方點選我要找人'},
    # ]
    #
    # # Initialize example selector.
    # example_selector = CustomExampleSelector(dict_examples2)
    # print(example_selector.select_examples())
    #
    # # Add new example to the set of examples
    # example_selector.add_example({"新人什麼時候會有帳號?": "要等人總處建置資料, 各系統才會自動開立帳號"})
    # print(example_selector.examples)
    # print(example_selector.select_examples())

    print("-----------------------------")
    dict_examples3 = [
        {'question': '請問如何申請加班', 'answer': '請到EIP點選辦公室事務->加班管理'},
        {'question': '請問超過6點怎樣申請加班', 'answer': '若當日欲申請時間超6點要申請加班,請隔日即可申請前天加班'},
        {'question': '那裡可查同仁分機?', 'answer': '請到EIP上方點選我要找人'},
    ]

    example_prompt = PromptTemplate.from_template(template="Question:{question}\nAnswer:{answer}")
    example_selector = LengthBasedExampleSelector(
        # The examples it has available to choose from.
        examples=dict_examples3,
        # The PromptTemplate being used to format the examples.
        example_prompt=example_prompt,
        # Length is measured by the get_text_length function below.
        max_length=25,
        # get_text_length 用於取得字串長度的函數，用於確定包含哪些範例。如果未指定，將作為預設值提供。
        # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
    )

    dynamic_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="你好~你要問什麼問題?",
        suffix="Question: {q}\nAnswer:",
        input_variables=["q"],
    )

    # An example with small input, so it selects all examples.
    short_string = "加班申請"
    ans = llm.invoke(dynamic_prompt.format(q=short_string))
    print(ans)
