import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
import streamlit as st

os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0.9)

# App framework
st.title("LangChain GPT")
prompt = st.text_input("What is your question? ")

#text = "What would be a good company name for a company that makes colorful socks?"
#print(llm(text))

titleTemplate = PromptTemplate(
  input_variables=["topic"],
  template="Write me a youtube video title about {topic}",
)

scriptTemplate = PromptTemplate(
  input_variables=["title", 'wikipedia_research'],
  template="Write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia research:{wikipedia_research} ",
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

title_chain = LLMChain(llm=llm, prompt=titleTemplate, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=scriptTemplate, verbose=True, output_key='script', memory=script_memory)

#sequentialChain = SequentialChain(chains=[title_chain, script_chain], verbose=True, input_variables=["topic"], output_variables=["title", "script"])

wiki = WikipediaAPIWrapper()

# display result
if prompt:
  title = title_chain.run(prompt)
  wiki_research = wiki.run(prompt)
  script = script_chain.run(title=title, wikipedia_research=wiki_research)

  st.write(title)
  st.write(script)
  
  #response = sequentialChain({'topic': prompt})
  #st.write(response['title'])
  #st.write(response['script'])
  
  with st.expander('Title History'):
    st.info(title_memory.buffer)

  with st.expander('Script History'):
    st.info(script_memory.buffer)

  with st.expander('Wikipedia Research'): 
    st.info(wiki_research)
