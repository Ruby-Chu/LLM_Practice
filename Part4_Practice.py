from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

if __name__ == "__main__":
    # .format()
    prompt1 = PromptTemplate(template="寫出一篇有關{story}，擁有{style}風格的短篇故事",
                             input_variables=["story", "style"])

    prompt1.format(story="冒險", style="魔法")
    # .from_template
    prompt_temp1 = PromptTemplate.from_template(template="寫出一篇有關{story}，擁有{style}風格的短篇故事")
    # .partial()
    prompt2 = prompt_temp1.partial(style="魔法")
    prompt2 = prompt2.format(story="冒險")
    print(prompt2)
