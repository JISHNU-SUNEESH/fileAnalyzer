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

# Show title and description.
st.title("📄 Document question answering")
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
        llm2=HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            huggingfacehub_api_token=huggingface_api,
            task="text-generation",
            max_new_tokens=512

)

        chat_model=ChatHuggingFace(llm=llm2)

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


                create_query_prompt=PromptTemplate(
                input_variables=["input","table_info","top_k"],
                template="""You are an agent designed to interact with a SQL database.
                    Your job is to create only the sql query based on the user question.
                    Do not produce any other outputs that the correct sql query.
                    The query must not contain "\". The query should be clean and executable

                    question: {input}
                    table_info: {table_info}
                    top_k: {top_k}
                    query:
                    """
                    )

                create_query_chain=create_sql_query_chain(llm,db,create_query_prompt)
                execute_query_chain=QuerySQLDataBaseTool(db=db)
                answer_prompt=PromptTemplate.from_template(
                        """
                        Given the following user question, corresponding SQL query, and SQL result, answer the user question.
                        Provide the answer in a structured format. Don't explain the query used.If the details are represented in tabular format, use the column names as well.

                        Question: {question}
                        SQLQuery: {query}
                        SQL Result: {result}
                        Answer: """
                )

                answer=answer_prompt | chat_model | StrOutputParser()

                chain=(
                    RunnablePassthrough.assign(query=create_query_chain).assign(
                    result=itemgetter("query") | execute_query_chain
                    )|answer
                    )
                chain_1= create_query_chain | execute_query_chain
                question = st.text_area(
                "Now ask a question about your data",
                placeholder="Can you give me a short summary?",
                disabled=not uploaded_file,
                )

                if question:
                    query=create_query_chain.invoke({"question":question})
                    query_result=chain_1.invoke({"question":question})
                    response=chain.invoke({"question":question})

        

        # Stream the response to the app using `st.write_stream`.
                    ex1=st.expander("Query Used")
                    ex2=st.expander("Query Result")
                    ex1.write(query)
                    ex2.write(query_result)
                    st.write(response)

            except Exception as e:
                 st.error(str(e))
