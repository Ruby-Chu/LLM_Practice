from utils.config import get_openapi_key
import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain.vectorstores.chroma import Chroma
from chromadb.errors import InvalidDimensionException

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    row_documents = TextLoader('state_of_the_union.txt', encoding='utf-8').load()
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )

    documents = text_splitter.split_documents(row_documents)
    try:
        chroma_db = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())
    except InvalidDimensionException:
        Chroma().delete_collection()
        chroma_db = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

    # with open('state_of_the_union.txt', encoding='utf-8') as f:
    #     state_of_the_union = f.read()
    #     print(state_of_the_union)
    #     print(type(state_of_the_union))
    #     print(len(state_of_the_union))
    #

    #     print(text_splitter)
    #
    #     doucments = text_splitter.create_documents([state_of_the_union])
    #     print(doucments[0])
    #     print('---------------------\n')
    #     print(doucments[1])
    #     print('---------------------\n')
