from utils.config import get_openapi_key
import os
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID



if __name__ == "__main__":
    memory = ConversationBufferWindowMemory()
    memory.chat_memory.add_user_message("hi!")
    memory.chat_memory.add_ai_message("whats up?")
    memory.load_memory_variables({})

    memory = ConversationBufferWindowMemory()
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    memory.load_memory_variables({})

    memory = ConversationBufferWindowMemory(k=1, return_messages=True)  # k保留前k個資訊
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})
    print(memory.load_memory_variables({}))

    memory = ConversationBufferWindowMemory(k=2, return_messages=True)
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})
    print(memory.load_memory_variables({}))
