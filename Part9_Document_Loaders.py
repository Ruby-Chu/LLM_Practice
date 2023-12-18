# from langchain.output_parsers import PandasDataFrameOutputParser
# from langchain.pydantic_v1 import BaseModel, Field, validator
# from langchain.output_parsers import PydanticOutputParser
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import CSVLoader
# from langchain.document_loaders import UnstructuredMarkdownLoader
# from langchain.document_loaders import UnstructuredHTMLLoader
# from langchain.document_loaders import JSONLoader
# import json
# from pathlib import Path
# from pprint import pprint
from langchain.document_loaders import PyPDFLoader

# Define the metadata extraction function.
# def metadata_func(record: dict, metadata: dict) -> dict:
#     metadata["Question"] = record.get("Question")
#
#     return metadata


if __name__ == "__main__":
    # # 讀取檔案
    # text_loader = TextLoader("README.md", encoding='UTF-8')
    # print(text_loader.load())

    # # 讀取CSV檔案
    # csv_loader = CSVLoader(file_path='data.csv', encoding='UTF-8')
    # data = csv_loader.load()
    # print(data)

    # Customizing the CSV parsing and loading
    # csv2_loader = CSVLoader(file_path='data.csv', encoding='UTF-8', csv_args={
    #     'delimiter': ',',
    #     # 'quotechar': '"',
    #     'fieldnames': ['No.','Question', 'Answer']
    # })
    #
    # data2 = csv2_loader.load()
    # print(data2)

    # Specify a column to identify the document source
    # csv3_loader = CSVLoader(file_path='mlb_teams_2012.csv', source_column="Team")
    #
    # data3 = csv3_loader.load()
    # print(data3)

    # Markdown
    # markdown_path1 = "README.md"
    # markdown_loader1 = UnstructuredMarkdownLoader(file_path=markdown_path1, encodings="UTF-8")
    # data4 = markdown_loader1.load()
    # print(data4)
    # print(len(data4))
    # print(data4[0])

    # markdown_path2 = "README.md"
    # markdown_loader2 = UnstructuredMarkdownLoader(file_path=markdown_path2, encodings="UTF-8", mode="elements")
    # data5 = markdown_loader2.load()
    # print(data5)
    # print(len(data5))
    # print(data5[0])

    # # HTML
    # html_loader = UnstructuredHTMLLoader("Python_documentation.html")
    # data6 = html_loader.load()
    # print(data6)

    # # JSON
    # json_loader = JSONLoader(
    #     file_path='ata.json',
    #     jq_schema='.[].Question',
    #     text_content=False)
    #
    # json_data = json_loader.load()
    # print(json_data)

    # # JSON Metadata
    # loader = JSONLoader(
    #     file_path='data.json',
    #     jq_schema='.[]',
    #     content_key="Answer",
    #     metadata_func=metadata_func
    # )
    #
    # data = loader.load()

    pdf_loader = PyPDFLoader("Attention_is_all_you_need.pdf", )
    pages = pdf_loader.load()
    print(pages)

    print(pages[0])
