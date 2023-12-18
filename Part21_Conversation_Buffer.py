from utils.config import get_openapi_key
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    memory = ConversationBufferMemory()
    memory.chat_memory.add_user_message("hi!")
    memory.chat_memory.add_ai_message("whats up?")
    print(memory.load_memory_variables({}))

    memory = ConversationBufferMemory()
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    print(memory.load_memory_variables({}))

    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    memory.save_context({"input": "你好~"}, {"output": "你好，請問有什麼問題嗎?"})
    print(memory.load_memory_variables({}))


