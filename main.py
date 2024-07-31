## Integrate our code with open ai API 
import os 
from constants import MISTRAL_API_KEY
from langchain_mistralai import ChatMistralAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

name_parser = StrOutputParser()

prompt_temp = PromptTemplate(
    input_variables=['name'],
    template= "Name the person who is the author of the book {name}",
)
print(prompt_temp.format(name = 'Cracking the coding interview'))

template_examples = [
    {'word': 'happy', 'antonym': 'sad'},
    {'word': 'shiny', 'antonym': 'dark'} 
]

template = """word: {word}
antonym: {antonym}
"""

prompt_template = PromptTemplate(
    input_variables=['word', 'antonym'],
    template=template
)

## To feed the sample to the model , we need the few-shot prompt template 
from langchain_core.prompts import FewShotPromptTemplate

few_shot_prompt = FewShotPromptTemplate(
    examples = template_examples,
    example_prompt = prompt_template,
    prefix = "Give the Antonym for the input",
    suffix = "word: {input}\nantonym: ",
    input_variables = ['input'],
    example_separator='\n',
)
print(few_shot_prompt.format(input='heavy'))
'''
Give the antonym for the input

Word: happy
Antonym: sad


Word: shiny
Antonym: dark


word: heavy
 antonym: 
'''
model = ChatMistralAI(model="mistral-large-latest", max_tokens=50)
chain = LLMChain(prompt=few_shot_prompt, llm=model)
# print(chain.invoke({'input': 'heavy'}))
print(chain({'input': 'heavy'}))