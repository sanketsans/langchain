## Integrate our code with open ai API 
import os 
from constants import MISTRAL_API_KEY
from langchain_mistralai import ChatMistralAI
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import DatetimeOutputParser, CommaSeparatedListOutputParser
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Memory

# from langchain.memory import ConversationBufferMemory ## to store the conversation memory 
# person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
# dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
# descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

## Prompting messages to be of specific type 

name_parser = StrOutputParser()

## Way I 
# prompt_temp = ChatPromptTemplate.from_messages([
#     ("system", "Tell about the video game") , ("user", "{text}")
# ])
# template = prompt_temp.invoke({"text": "spider man"}) ## output : messages=[SystemMessage(content='Tell about the video game'), HumanMessage(content='spider man')]

## Way II 
prompt_temp = PromptTemplate(
    input_variables=['name'],
    template= "Name the person who is the author of the book {name}",
)
# template = prompt_temp.invoke({'name': 'The way of kings'}) ## output : text='Tell me about the book The way of kings'

model = ChatMistralAI(model="mistral-large-latest", max_tokens=50)

## Chaining it all together 
chain1 = LLMChain(prompt=prompt_temp, llm=model, output_parser=name_parser, verbose=True, output_key='person') #, memory=person_memory) # prompt_temp | model | parser
# chain1 = prompt_temp | model | parser ## - does not seem to work this way, when using with sequentialChains
# print(chain1.invoke({'name': 'the way of kings'})) ## or chain.invoke({'text': 'spider-man'}) - when using prompt 1


## for datetime parsing 
date_parser = DatetimeOutputParser()
date_format_instructions = date_parser.get_format_instructions()
template2 = """Answer the following question:
When was the book {name} launched?
{format_instructions}
"""
prompt_temp2 = PromptTemplate(
    input_variables=['name'],
    template= template2, #'When was the book {name} launched ?',
    partial_variables={'format_instructions': date_format_instructions},
)
chain2 = LLMChain(prompt=prompt_temp2, llm=model, output_parser=date_parser, output_key='dob') #, memory=dob_memory) ## can be used later as the variable name (output_key)
# chain2 = prompt_temp2 | model | date_parser


list_parser = CommaSeparatedListOutputParser()
list_instructions = list_parser.get_format_instructions()
template3 = """Answer the following questions:
List 5 major events on the date {dob}.
{format_instructions}
"""
prompt_temp3 = PromptTemplate(
    input_variables=['dob'],
    template=template3,
    partial_variables={'format_instructions': list_instructions}
)
# chain = prompt_temp3 | model | list_parser
chain3 = LLMChain(prompt=prompt_temp3, llm=model, output_parser=list_parser, verbose=True, output_key='contents') #, memory=descr_memory)

parent = SequentialChain(
    chains=[chain1, chain2, chain3], 
    input_variables=['name', 'name'],
    output_variables=['person', 'dob', 'contents'],
    verbose=True)

# output = parent({'name': 'Dune'})
# print(output['dob'])
# print(output['person'])
# print(output['contents'])

'''
Sample output : 
1965-08-01 00:00:00
The author of the book "Dune" is Frank Herbert. "Dune" is a science fiction novel that was first published in 1965. It is the first book in the Dune series and is often considered one of the
['Watts Riots begin in Los Angeles', 'Beatles release the album "Help!"', 'Indo-Pakistani War of 1965 starts', 'Lyndon B. Johnson signs the Voting Rights Act', 'Cig']
'''
# print(parent.run('Dune')) ## - When using the SimpleSequentialChain



## streamlit framework 
st.title('LangChain demo')
input_text = st.text_input('Enter completion text :')

if input_text:
    output = parent({'name': input_text})
    st.write(output)

    # with st.expander('Book Info'):
    #     st.info(person_memory.buffer)

    with st.expander('Alt Book Info'):
        st.info(str(output['person']))

    with st.expander('List of Events'):
        st.info(output['contents'])

#     messages = [
#         SystemMessage(content="Complete the sentence"),
#         HumanMessage(content=input_text),
#     ]

#     result = model.invoke(messages)
#     ## result.content (can do both)
#     output = parser.invoke(result)
    # st.write(chain.invoke({'name': input_text}))