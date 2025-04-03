from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from .state import AgentState
from .prompts import REPORT_STRUCTURE_PLANNER_SYSTEM_PROMPT_TEMPLATE
from langchain_core.messages import HumanMessage
from langgraph.graph import END


# <------ REPORT STRUCTURE PLANNER ------->

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

report_structure_planner_system_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(REPORT_STRUCTURE_PLANNER_SYSTEM_PROMPT_TEMPLATE),
    HumanMessagePromptTemplate.from_template(
        template="""
        Topic: {topic}
        Outline: {outline}
        """
    ),
    MessagesPlaceholder(variable_name="messages")
])

report_structure_planner = report_structure_planner_system_prompt | llm

def report_structure_planner_node(state: AgentState, config: RunnableConfig):
    result = report_structure_planner.invoke(state)
    return {"messages": [result]}



# <------ HUMAN FEEDBACK ------->

def human_feedback_node(state: AgentState, config: RunnableConfig):
    human_message = input("Please provide feedback on the report structure (type 'continue' to continue): ")
    return {"messages": [HumanMessage(content=human_message)]}
    
def human_feedback_router(state: AgentState, config: RunnableConfig):
    message = state.get("messages")[-1]
    if message.content == "continue":
        return END
    else:
        return "report_structure_planner"