import streamlit as st
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain.chains import create_sql_query_chain
from langchain_mistralai import ChatMistralAI
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
import time
from agent import Agent
# Show title and description.
st.title("📁File Anlyzer")
st.write(
    "Upload your data and start Anlyising! "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
api_key=st.secrets["mistralai_api_key"]
huggingface_api=st.secrets["hugginface_api_key"]
if not api_key:
    st.info("Please add your API key to continue.", icon="🗝️")
else:
    try:
        llm=ChatMistralAI(model_name='mistral-large-latest',api_key=api_key)



    except Exception as e:
         st.error(str(e))
    # Create an OpenAI client.
    

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.csv)", type=("csv")
    )



    # Ask the user for a question via `st.text_area`.


    if uploaded_file :
            try:
            
                df=pd.read_csv(uploaded_file,header=0)
                engine=create_engine('sqlite:///uploaded.db')
                df.to_sql('uploaded_table',con=engine,if_exists='replace',index=False)
                db=SQLDatabase(engine=engine)

                agent=Agent(llm,db)
                app=agent._agent_call()
               
                question = st.text_area(
                "Now ask a question about your data",
                placeholder="Can you give me a short summary?",
                disabled=not uploaded_file,
                )

                if question:
                    
                    inputs = {
                        "question": question
                    }
                    for output in app.stream(inputs):
                        for key, value in output.items():
                            agent_answer=value.get("answer")
                            agent_query=value.get('query')
                            agent_output=value.get('output')
        

        # Stream the response to the app using `st.write_stream`.
                    st.write(agent_answer)
                    ex1=st.expander("Query Used")
                    ex2=st.expander("Query Result")
                    ex1.write(agent_query)
                    ex2.write(agent_output)
                    

            except Exception as e:
                 st.error(str(e))