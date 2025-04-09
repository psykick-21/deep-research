from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage
from langgraph.types import Command, Send
from typing import Literal
from tavily import TavilyClient
from .state import AgentState, ResearchState
from .configuration import Configuration
from .utils import init_llm
from .prompts import (
    REPORT_STRUCTURE_PLANNER_SYSTEM_PROMPT_TEMPLATE,
    SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE,
    SECTION_KNOWLEDGE_SYSTEM_PROMPT_TEMPLATE,
    QUERY_GENERATOR_SYSTEM_PROMPT_TEMPLATE,
    RESULT_ACCUMULATOR_SYSTEM_PROMPT_TEMPLATE,
    REFLECTION_FEEDBACK_SYSTEM_PROMPT_TEMPLATE,
    FINAL_SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE,
    FINAL_REPORT_WRITER_SYSTEM_PROMPT_TEMPLATE
)
from .struct import (
    Sections,
    Queries,
    SearchResult,
    Feedback
)



def report_structure_planner_node(state: AgentState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)

    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

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

    report_structure_planner_llm = report_structure_planner_system_prompt | llm

    result = report_structure_planner_llm.invoke(state)
    return {"messages": [result]}


def human_feedback_node(
        state: AgentState, 
        config: RunnableConfig
) -> Command[Literal["section_formatter", "report_structure_planner"]]:
    
    human_message = input("Please provide feedback on the report structure (type 'continue' to continue): ")
    report_structure = state.get("messages")[-1].content
    if human_message == "continue":
        return Command(
            goto="section_formatter",
            update={"messages": [HumanMessage(content=human_message)], "report_structure": report_structure}
        )
    else:
        return Command(
            goto="report_structure_planner",
            update={"messages": [HumanMessage(content=human_message)]}
        )
    

def section_formatter_node(state: AgentState, config: RunnableConfig) -> Command[Literal["research_agent"]]:

    configurable = Configuration.from_runnable_config(config)
    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

    section_formatter_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="{report_structure}"),
    ])
    section_formatter_llm = section_formatter_system_prompt | llm.with_structured_output(Sections)

    result = section_formatter_llm.invoke(state)
    
    return Command(
        update={"sections": result.sections},
        goto=[
            Send(
                "research_agent",
                {
                    "section": s,
                }
            ) for s in result.sections
        ]
    )


def section_knowledge_node(state: ResearchState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

    section_knowledge_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SECTION_KNOWLEDGE_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="{section}"),
    ])
    section_knowledge_llm = section_knowledge_system_prompt | llm

    result = section_knowledge_llm.invoke(state)

    with open("agent_int_logs.txt", "a") as f:
        f.write(f"Section: {state['section']}\n")
        f.write(f"Knowledge: {result.content}\n\n")

    return {"knowledge": result.content}


def query_generator_node(state: ResearchState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

    query_generator_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            QUERY_GENERATOR_SYSTEM_PROMPT_TEMPLATE.format(max_queries=configurable.max_queries)
        ),
        HumanMessagePromptTemplate.from_template(
            template="Section: {section}\nPrevious Queries: {searched_queries}\nReflection Feedback: {reflection_feedback}"
        ),
    ])
    query_generator_llm = query_generator_system_prompt | llm.with_structured_output(Queries)

    state["reflection_feedback"] = state.get("reflection_feedback", Feedback(feedback=""))
    state["searched_queries"] = state.get("searched_queries", [])

    result = query_generator_llm.invoke(state)

    with open("agent_int_logs.txt", "a") as f:
        f.write(f"Section: {state['section']}\n")
        f.write(f"Generated Queries: {result.queries}\n\n")

    return {"generated_queries": result.queries, "searched_queries": result.queries}


def tavily_search_node(state: ResearchState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)

    tavily_client = TavilyClient()
    queries = state["generated_queries"]
    search_results = []

    for query in queries:
        raw_content = []
        response = tavily_client.search(query=query.query, max_results=configurable.search_depth, include_raw_content=True)
        for result in response["results"]:
            if result['raw_content']:
                raw_content.append(result['raw_content'])
        search_results.append(SearchResult(query=query, raw_content=raw_content) if raw_content else None)

    with open("agent_int_logs.txt", "a") as f:
        f.write(f"Section: {state['section']}\n")
        f.write(f"Search Results: {search_results}\n\n")

    return {"search_results": search_results}


def result_accumulator_node(state: ResearchState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

    result_accumulator_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(RESULT_ACCUMULATOR_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="{search_results}"),
    ])
    result_accumulator_llm = result_accumulator_system_prompt | llm

    result = result_accumulator_llm.invoke(state)

    with open("agent_int_logs.txt", "a") as f:
        f.write(f"Section: {state['section']}\n")
        f.write(f"Accumulated Content: {result.content}\n\n")

    return {**state, "accumulated_content": result.content}


def reflection_feedback_node(
        state: ResearchState, 
        config: RunnableConfig
) -> Command[Literal["final_section_formatter", "query_generator"]]:
    
    configurable = Configuration.from_runnable_config(config)
    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

    reflection_feedback_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(REFLECTION_FEEDBACK_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="Section: {section}\nAccumulated Content: {accumulated_content}"),
    ])
    reflection_feedback_llm = reflection_feedback_system_prompt | llm.with_structured_output(Feedback)

    reflection_count = state["reflection_count"] if "reflection_count" in state else 1
    result = reflection_feedback_llm.invoke(state)
    feedback = result.feedback

    with open("agent_int_logs.txt", "a") as f:
        f.write(f"Section: {state['section']}\n")
        f.write(f"Reflection Feedback: {feedback}\n\n")

    if (feedback == True) or (feedback.lower() == "true") or (reflection_count < configurable.num_reflections):
        return Command(
            update={"reflection_feedback": feedback, "reflection_count": reflection_count},
            goto="final_section_formatter"
        )
    else:
        return Command(
            update={"reflection_feedback": feedback, "reflection_count": reflection_count + 1},
            goto="query_generator"
        )


def final_section_formatter_node(state: ResearchState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

    # Debug: print the state to see what's available
    print("DEBUG - State keys in final_section_formatter_node:", state.keys())
    
    # Check if accumulated_content exists, if not, use an empty string
    accumulated_content = state.get("accumulated_content", "")
    if not accumulated_content and "search_results" in state:
        # Try to build accumulated_content from search_results if available
        accumulated_content = "\n".join([str(result) for result in state["search_results"]])
        print("DEBUG - Built accumulated_content from search_results")

    # Update the state with accumulated_content if it's missing
    state_dict = dict(state)
    state_dict["accumulated_content"] = accumulated_content

    final_section_formatter_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(FINAL_SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="Internal Knowledge: {knowledge}\nSearch Result content: {accumulated_content}"),
    ])
    final_section_formatter_llm = final_section_formatter_system_prompt | llm

    result = final_section_formatter_llm.invoke(state_dict)

    with open("agent_int_logs.txt", "a") as f:
        f.write(f"Section: {state['section']}\n")
        f.write(f"Final Section Content: {result.content}\n\n")

    return {"final_section_content": [result.content]}


def final_report_writer_node(state: AgentState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

    final_report_writer_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(FINAL_REPORT_WRITER_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="Report Structure: {report_structure}\nSection Contents: {final_section_content}"),
    ])
    final_report_writer_llm = final_report_writer_system_prompt | llm
    
    result = final_report_writer_llm.invoke(state)
    
    return {"final_report_content": result.content}