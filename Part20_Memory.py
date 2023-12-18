from utils.config import get_openapi_key
import os
from langchain.memory import ChatMessageHistory

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    history = ChatMessageHistory()

    history.add_user_message("hi!")
    history.add_ai_message("whats up?")
    print(history.messages)