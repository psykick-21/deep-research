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
    FINALIZER_SYSTEM_PROMPT_TEMPLATE
)
from .struct import (
    Sections,
    Queries,
    SearchResult,
    SearchResults,
    Feedback,
    ConclusionAndReferences
)
import time


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
    

def section_formatter_node(state: AgentState, config: RunnableConfig) -> Command[Literal["queue_next_section"]]:

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

    with open("logs/sections.json", "w", encoding="utf-8") as f:
        f.write(result.model_dump_json())
    
    # Initialize the sections queue and current section index
    return Command(
        update={
            "sections": result.sections,
            "current_section_index": 0
        },
        goto="queue_next_section"
    )


def queue_next_section_node(state: AgentState, config: RunnableConfig) -> Command[Literal["research_agent", "finalizer"]]:
    """
    Process sections one at a time to avoid rate limit issues.
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Check if we have sections left to process
    if state["current_section_index"] < len(state["sections"]):
        current_section = state["sections"][state["current_section_index"]]
        
        # Add a delay to avoid rate limits if not the first section
        if state["current_section_index"] > 0:
            print(f"Waiting {configurable.section_delay_seconds} seconds before processing next section to avoid rate limits...")
            time.sleep(configurable.section_delay_seconds)
            
        print(f"Processing section {state['current_section_index'] + 1}/{len(state['sections'])}: {current_section.section_name}")
        
        # Update the index for the next round
        return Command(
            update={"current_section_index": state["current_section_index"] + 1},
            goto=Send("research_agent", {"section": current_section, "current_section_index": state["current_section_index"]})
        )
    else:
        # All sections have been processed
        print(f"All {len(state['sections'])} sections have been processed. Generating final report...")
        return Command(goto="finalizer")


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

    return {"generated_queries": result.queries, "searched_queries": result.queries}


def tavily_search_node(state: ResearchState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)

    tavily_client = TavilyClient()
    queries = state["generated_queries"]
    search_results = []

    for query in queries:
        search_content = []
        response = tavily_client.search(query=query.query, max_results=configurable.search_depth, include_raw_content=True)
        for result in response["results"]:
            if result['raw_content'] and result['url'] and result['title']:
                search_content.append(SearchResult(url=result['url'], title=result['title'], raw_content=result['raw_content']))
        search_results.append(SearchResults(query=query, results=search_content))

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

    return {"accumulated_content": result.content}


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

    final_section_formatter_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(FINAL_SECTION_FORMATTER_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="Internal Knowledge: {knowledge}\nSearch Result content: {accumulated_content}"),
    ])
    final_section_formatter_llm = final_section_formatter_system_prompt | llm

    result = final_section_formatter_llm.invoke(state)

    with open(f"logs/{state['current_section_index']+1}. {state['section'].section_name}.md", "a", encoding="utf-8") as f:
        f.write(f"{result.content}")

    return {"final_section_content": [result.content]}


def finalizer_node(state: AgentState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    llm = init_llm(
        provider=configurable.provider,
        model=configurable.model,
        temperature=configurable.temperature
    )

    extracted_search_results = []
    for search_results in state['search_results']:
        for search_result in search_results.results:
            extracted_search_results.append({"url": search_result.url, "title": search_result.title})

    finalizer_system_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(FINALIZER_SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template(template="Section Contents: {final_section_content}\n\nSearches: {extracted_search_results}"),
    ])
    finalizer_llm = finalizer_system_prompt | llm.with_structured_output(ConclusionAndReferences)

    result = finalizer_llm.invoke({**state, "extracted_search_results": extracted_search_results})

    final_report = "\n\n".join([section_content for section_content in state["final_section_content"]])
    final_report += "\n\n" + result.conclusion
    final_report += "\n\n# References\n\n" + "\n".join(["- "+reference for reference in result.references])
    final_report = f"# {state['topic']}" + final_report

    with open(f"reports/{state['topic']}.md", "w", encoding="utf-8") as f:
        f.write(final_report)

    return {"final_report_content": final_report}