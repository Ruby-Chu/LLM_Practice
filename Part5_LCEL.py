from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

if __name__ == "__main__":
    # PromptTemplate
    prompt_template1 = PromptTemplate.from_template("請提供{program} {name}課程目錄")
    prompt_value1 = prompt_template1.invoke({"program": "python", "name": "爬蟲"})
    print(prompt_value1)  # text='請提供python 爬蟲課程目錄'
    print(prompt_value1.to_messages())  # [HumanMessage(content='請提供python 爬蟲課程目錄')]
    print(prompt_value1.to_string())  # 請提供python 爬蟲課程目錄

    # ChatPromptTemplate
    prompt_template2 = ChatPromptTemplate.from_template("請告訴我{day}的日期")
    messages_value = prompt_template2.invoke({"day": "中秋節"})
    print(messages_value)
    print(messages_value.to_messages())
    print(messages_value.to_string())
