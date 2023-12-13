# from utils.config import get_openapi_key
# import os

from langchain.prompts import PromptTemplate

# OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    # no parameter input
    no_input_prompt = PromptTemplate(template="請提供Python 爬蟲課程目錄", input_variables=[])
    no_input_prompt.format()

    # single parameter input
    one_input_prompt = PromptTemplate(template="請提供Python {obj1}課程目錄", input_variables=["obj1"])
    one_input_prompt.format(obj1="爬蟲")

    # multiple parameter input
    multi_input_prompt = PromptTemplate(template="給我一個可愛的{obj1}和{obj2}公司名稱",
                                        input_variables=["obj1", "obj2"])
    multi_input_prompt.format(obj1="兔子", obj2="髮式")

    # 可使用from_template來達到相同的目的，使用此方法可以不用設定input_variables
    prompt_template = PromptTemplate.from_template("給我一個可愛的{obj1}和{obj2}公司名稱")
    prompt_template.format(obj1="兔子", obj2="髮式")
