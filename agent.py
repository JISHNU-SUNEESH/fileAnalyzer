# from streamlit_app import db,llm
from langchain.chains import create_sql_query_chain
from langchain.prompts import PromptTemplate
from langchain.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser,BaseOutputParser
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph,START,END
import time

class ErrorGradeParser(BaseOutputParser):

  @property
  def _type(self)->str:
    return "error_grade_parser"

  def parse(self,text:str)->str:
        text=text.strip().lower()
        if text.startswith("yes"):
          return "yes"

        else:
          return "no"


class GraphState(TypedDict):
  query:str
  output:str
  question:str
  answer:str


class Agent:
        def __init__(self,llm,db) -> None:
          self.llm=llm
          self.db=db



        def call_agent(self):
          
            create_query_prompt=PromptTemplate(
            input_variables=["table_info","question","top_k"],
            template="""You are an agent designed to interact with a SQL database.
            Your job is to create only the sql query based on the user question.
            Do not produce any other outputs than the correct sql query.
            The query must not contain "\". The query should be clean and executable

            question: {input}
            table_info: {table_info}
            top_k: {top_k}
            query:
            """
        )

            create_query_chain=create_sql_query_chain(llm=self.llm,db=self.db,prompt=self.create_query_prompt)
            execute_query_chain=QuerySQLDataBaseTool(db=self.db)

            error_re_write_message="""You an SQL query re-writter that resolves the error in
                                            the current sql query executed based on the error
                                            output received.Do not produce any other
                                            outputs than the correct sql query.
                                            The query must not contain "\".
                                            The query should be clean and executable in sqlite.
                                            Only create the modified SQL query as your output."""

            error_re_write_prompt=ChatPromptTemplate.from_messages(
                [
                    ("system",self.error_re_write_message),
                    ("human","Here is the error output:{output}, Formulate an error free SQL query")
                ]
            )

            re_write_chain=self.error_re_write_prompt | self.llm | StrOutputParser()


            answer_prompt=PromptTemplate.from_template(
                        """
                    Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQLQuery: {query}
            SQL Result: {output}
            Answer: """
            )

            answer_chain= self.answer_prompt | self.llm | StrOutputParser()



            error_grade_parser=ErrorGradeParser()

            error_prompt_message="""You are an error grader assessing whether the received sql query output has an error \n
                Give a binary score 'yes' or 'no'. Yes' means that the output is an error ouput.
                Give yes or no,nothing else. No explanations as well. A plain yes or no"""

            error_prompt=ChatPromptTemplate.from_messages([
                ('system',self.error_prompt_message),
                ('human',"Here is the query output: {output}")
            ])

            error_grade_chain=self.error_prompt | self.llm | self.error_grade_parser
            workflow=StateGraph(GraphState)
            workflow.add_node("create_query",create_query)
            workflow.add_node("execute_query",execute_query)
            workflow.add_node("generate",generate)
            workflow.add_node("error_re_write",error_re_write)
            workflow.add_node("grade_error",grade_error)



            workflow.add_edge(START,"create_query")
            workflow.add_edge("create_query","execute_query")
            workflow.add_edge("execute_query","grade_error")
            workflow.add_conditional_edges(
                "grade_error",
                decide_generate,
                {
                    "error_re_write":"error_re_write",
                    "generate":"generate",
                }
                    )
            workflow.add_edge("error_re_write","execute_query")
            workflow.add_edge("generate",END)

            app=workflow.compile()
                
                
            return self.app




def create_query(state:GraphState):
  print("--CREATE QUERY--")
  question=state["question"]
  query=Agent.create_query_chain.invoke({"question":question})
  print(f"QUERY = {query}")
  time.sleep(3)
  return {
      "query":query,
      "question":question
      }

def execute_query(state:GraphState):
  print("--EXECUTE QUERY--")
  question=state["question"]
  query=state["query"]
  output=Agent.execute_query_chain.invoke({"query":query})
  print(f"OUTPUT = {output}")
  time.sleep(3)
  return {
      "output":output,
      "query":query,
      "question":question
      }

def generate(state:GraphState):
  print("--GENERATE--")
  question=state["question"]
  query=state["query"]
  output=state["output"]
  time.sleep(3)
  answer=Agent.answer_chain.invoke({"question":question,"query":query,"output":output})
  print(f"ANSWER = {answer}")
  time.sleep(3)
  return{
      "answer":answer,
      "question":question,
      "query":query,
      "output":output
      }

def error_re_write(state:GraphState):
  print("--RE-WRITE--")
  question=state["question"]
  query=state["query"]
  output=state["output"]
  time.sleep(3)
  query=Agent.re_write_chain.invoke({"output":output})
  print(f"RE-WRTITTEN QUERY: {query} ")
  return{
      "query":query,
      "question":question
  }

def grade_error(state:GraphState):
  print("--GRADE ERROR--")
  question=state["question"]
  query=state["query"]
  output=state["output"]
  grade=Agent.error_grade_chain.invoke({"output":output})
  print(f"Error = {grade}")
  return{
      "grade":grade,
      "question":question,
      "query":query,
      "output":output
  }

def decide_generate(state:GraphState):
  print("--DECIDE GENERATE--")
  grade=state["grade"]
  if grade=="yes":
    return "error_re_write"
  else:
    return "generate"




