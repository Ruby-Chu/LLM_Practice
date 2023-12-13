# from utils.config import get_openapi_key
# import os

from langchain.prompts import PromptTemplate
# from langchain.prompts import ChatPromptTemplate

from datetime import datetime


# OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID


def get_datetime():
    # return datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    return datetime.now().strftime("%Y/%m/%d")


if __name__ == "__main__":
    # 可以使用或不使用的參數

    # Method 1: 在實際狀況下，並非所有的參數都能同時取得的，這時候我們需要在鏈式結構中傳遞提是模板時，僅接受部分的參數。
    prompt_template1 = PromptTemplate.from_template(template="請給我一個{adj1}{obj1}公司名稱")
    partial_prompt1 = prompt_template1.partial(adj1="國際的")
    partial_prompt1.format(obj1="科技業")

    # Method 2: 也可以使用 partialed_variables 初始化提示，這通常在工作流程中更合理。
    prompt_template2 = PromptTemplate.from_template(template="請給我一個{adj1}{obj1}公司名稱",
                                                    partial_variables={"adj1": "國際的"})
    prompt_template2 = prompt_template2.format(obj1="科技業")

    # Method 3
    prompt_template3 = PromptTemplate(template="請給我一個{adj1}{obj1}公司名稱",
                                      input_variables=["obj1"],
                                      partial_variables={"adj1": "國際的"})
    prompt_template3.format(obj1="科技業")

    prompt1 = PromptTemplate(
        template="請告訴我{date} {location}的平均氣溫",
        input_variables=["location", "date"], )
    partial_prompt1 = prompt1.partial(date=get_datetime)
    partial_prompt1.format(location="宜蘭")

    prompt2 = PromptTemplate(
        template="請告訴我{date} {location}的平均氣溫",
        input_variables=["location"],
        partial_variables={"date": get_datetime()})
    # partial_prompt = prompt.partial(date=get_datetime)
    prompt2.format(location="宜蘭")
