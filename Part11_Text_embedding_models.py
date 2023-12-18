from utils.config import get_openapi_key
import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# from langchain.vectorstores import Chroma
from langchain.vectorstores.chroma import Chroma
from chromadb.errors import InvalidDimensionException

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID

if __name__ == "__main__":
    # embeddings_model = OpenAIEmbeddings()
    #
    # embedded_doc = embeddings_model.embed_documents(
    #     [
    #         "Hi there!",
    #         "Oh, hello",
    #         "What's your name?",
    #         "My friends call me World",
    #         "Hello World!"
    #     ]
    # )
    #
    # print(len(embedded_doc))
    # print(len(embedded_doc[0]))
    # print(embedded_doc[0][:5])
    #
    # print('------------------------------------\n')
    # embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
    # print(embedded_query[:5])

    raw_documents = TextLoader('state_of_the_union.txt', encoding="utf-8").load()

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )

    # try:
    #     db = Chroma.from_documents(raw_documents, embedding=OpenAIEmbeddings())
    # except InvalidDimensionException:
    #     Chroma().delete_collection()
    #     db = Chroma.from_documents(raw_documents, embedding=OpenAIEmbeddings())
    #
    # query = "What did the president say about Ketanji Brown Jackson"
    # docs = db.similarity_search(query)
    # print(docs[0].page_content)
    #
    # embedding_vector = OpenAIEmbeddings().embed_query(query)
    # docs = db.similarity_search_by_vector(embedding_vector)
    # print(docs[0].page_content)
