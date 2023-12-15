from langchain.prompts import PromptTemplate
# from langchain.prompts.few_shot import FewShotPromptTemplate
# from langchain.prompts.example_selector.base import BaseExampleSelector
# from langchain.prompts.example_selector import LengthBasedExampleSelector
# from typing import Dict, List, Any
# from langchain.pydantic_v1 import BaseModel, Field, validator
# import numpy as np

from utils.config import get_openapi_key
import os
from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.output_parsers import CommaSeparatedListOutputParser
# from langchain.output_parsers import DatetimeOutputParser
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from typing import Any, Dict
# import pandas as pd
# from langchain.output_parsers import PandasDataFrameOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser

OPENAI_API_KEY, ORGANIZATION_ID = get_openapi_key()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["ORGANIZATION_ID"] = ORGANIZATION_ID


# Define your desired data structure
class country_capital(BaseModel):
    setup: str = Field(description="question will give the game of a country")
    punchline: str = Field(description="answer the capital of country")


if __name__ == "__main__":
    model = OpenAI(temperature=0.0)

    # # List parser
    # list_output_parser = CommaSeparatedListOutputParser()
    # list_format_instructions = list_output_parser.get_format_instructions()
    #
    # prompt_template1 = PromptTemplate(
    #     template="請給我3種{subject}.\n{format_instructions}",
    #     input_variables=["subject"],
    #     partial_variables={"format_instructions": list_format_instructions}
    # )
    # prompt1 = prompt_template1.format(subject="口味的餅乾")
    # output1 = model(prompt1)
    # print('----------------------------')
    # print(output1)
    # print('--------List parser---------')
    # print(list_output_parser.parse(output1))
    # print('----------------------------\n')

    # # Datetime parser
    # datetime_output_parser = DatetimeOutputParser()
    # datetime_instructions = datetime_output_parser.get_format_instructions()
    # prompt_template2 = PromptTemplate(
    #     template="回答該項問題: {question} {format_instructions}",
    #     input_variables=["question"],
    #     partial_variables={"format_instructions": datetime_instructions}
    # )
    #
    # prompt2 = prompt_template2.format(question="請問聖誕節在什麼時候?")
    # output2 = model(prompt2)
    # print('----------------------------')
    # print(output2)
    # print('------Datetime parser-------')
    # print(datetime_output_parser.parse(output2))
    # print('----------------------------\n')

    # # Structured output parser
    # response_schemas = [
    #     ResponseSchema(name="answer", description="answer to the user's question"),
    #     ResponseSchema(name="score", description="source used to answer the user's question, should be a webside.")
    # ]
    # struct_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # struct_format_instructions = struct_output_parser.get_format_instructions()
    #
    # prompt_template3 = PromptTemplate(
    #     template="Anser the users question: {question}{format_instructions}",
    #     input_variables=["question"],
    #     partial_variables={"format_instructions": struct_format_instructions}
    # )
    #
    # prompt3 = prompt_template3.format(question="what's the capital of Taiwan?")
    # output3 = model(prompt3)
    # print('----------------------------')
    # print(output3)
    # print('--Structured output parser--')
    # print(struct_output_parser.parse(output3))
    # print('----------------------------')

    # # Define your desired Pandas DataFrame.
    # df = pd.DataFrame(
    #     {
    #         "num_legs": [2, 4, 8, 0],
    #         "num_wings": [2, 0, 0, 0],
    #         "num_specimen_seen": [10, 2, 1, 8]
    #     }
    # )
    #
    # # Set up a parser + inject instructions into the prompt template.
    # pandas_output_parser = PandasDataFrameOutputParser(dataframe=df)
    # pandas_format_instructions = pandas_output_parser.get_format_instructions()
    #
    # # set up the prompt
    # prompt_template4 = PromptTemplate(
    #     template="Answer the users question: {format_instructions} {question}",
    #     input_variables=["question"],
    #     partial_variables={"format_instructions": pandas_format_instructions}
    # )
    #
    # prompt4 = prompt_template4.format_prompt(question="Retrieve the first row.")
    # # or "Retrieve the num_wings column." / "Retrieve the mean of the num_fingers column."
    # output4 = model(prompt4.to_string())
    # print('----------------------------')
    # print(output4)
    # print('--Pandas DataFrame parser--')
    # print(pandas_output_parser.parse(output4))
    # print('----------------------------')
    # for k in pandas_output_parser.parse(output4).keys():
    #     print(pandas_output_parser.parse(output4)[k].to_dict())
    # print('----------------------------')

    # Pydantic (JSON) parser
    # # set up a parser + inject instructions into the prompt template.
    # json_output_parser = PydanticOutputParser(pydantic_object=country_capital)
    # json_format_instuctions = json_output_parser.get_format_instructions()
    #
    # # set up the prompt
    # prompt_template5 = PromptTemplate(
    #     template="Answer the users question: {question} {format_instructions}",
    #     input_variables=["question"],
    #     partial_variables={"format_instructions": json_format_instuctions}
    # )
    #
    # prompt5 = prompt_template5.format_prompt(question="give the name of a country")
    # output5 = model(prompt5.to_string())
    # print('----------------------------')
    # print(output5)
    # print('---Pydantic (JSON) parser---')
    # print(json_output_parser.parse(output5))
    # print('----------------------------')
    # print(json_output_parser.invoke(output5))
    # print('----------------------------')

    # Chain parser - another way
    # Set up a parser + inject instructions into the prompt template.
    output_parser = PydanticOutputParser(pydantic_object=country_capital)
    format_instructions = output_parser.get_format_instructions()

    # Set up the prompt.
    prompt_template = PromptTemplate(
        template="Answer the users question: {question} {format_instructions}",
        input_variables=["question"],
        partial_variables={"format_instructions": format_instructions},
    )
    #
    prompt_and_model = prompt_template | model

    output = prompt_and_model.invoke({'question': 'give the name of a country'})
    print(output)

    chain = prompt_template | model | output_parser
    ans = chain.invoke({'question': 'give the name of a country'})
    print(ans)