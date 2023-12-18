from utils.config import get_openapi_key
import os

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    with open('state_of_the_union.txt', encoding='utf-8') as f:
        state_of_the_union = f.read()
        print(state_of_the_union)
        print(type(state_of_the_union))
        print(len(state_of_the_union))

        text_splitter = CharacterTextSplitter(
            chunk_size=500,  # 區塊 (chunk) 最大的大小（透過長度函數測量）。
            chunk_overlap=50,  # 區塊 (chunk) 之間重疊的最大大小。 最好有一些重疊來保持區塊之間的連續性，例如做一個滑動視窗 (sliding window)。
            length_function=len,  # 如何計算區塊 (chunk) 的長度。 預設僅計算字元 (character) 數，不過 token 計數器也很常見。
            add_start_index=True  # boolean 值，表示是否在元資料 (metadata) 中包含每個區塊 (chunk) 在原始文件中的起始位置。
        )
        print(text_splitter)

        doucments = text_splitter.create_documents([state_of_the_union])
        print(doucments[0])
        print('---------------------\n')
        print(doucments[1])
        print('---------------------\n')
