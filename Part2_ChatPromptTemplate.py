# from utils.config import get_openapi_key
# import os

from langchain.prompts import ChatPromptTemplate

# OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    # ChatModel 包含兩個部分：role (角色) 和 message (訊息)。role 可以指定 system、human、ai 等角色。
    prompt_template = ChatPromptTemplate.from_template("請告訴我{day}的日期")
    prompt = prompt_template.format(day="聖誕節")
    # Human: 請告訴我聖誕節的日期

    # ChatPormptTemplate.from_messages 接受各種訊息表示形式。
    chat_template = ChatPromptTemplate.from_messages(
        [
            ('system', '你是一個{job}的專家'),
            ('human', '你是誰?'),
            ('ai', '您好！這裡是 Siri，是一個人類的好幫手！'),
            ('human', '{question}'),
        ]
    )
    message = chat_template.format_messages(job="法律", question="晚餐吃什麼")
    # Output:
    # [SystemMessage(content='你是一個法律的專家'), HumanMessage(content='你是誰?'), AIMessage(content='您好！這裡是 Siri，是一個人類的好幫手！'), HumanMessage(content='晚餐吃什麼')]
