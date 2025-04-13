from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage
from langgraph.types import Command, Send
from typing import Literal, Dict
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
import os


def report_structure_planner_node(state: AgentState, config: RunnableConfig) -> Dict:
    """
    Plans and generates the initial structure of a research report based on a given topic and outline.

    This node uses an LLM to generate a structured outline for the research report. It takes the topic
    and outline from the agent state and produces a detailed report structure that will guide the rest
    of the research and writing process.

    Args:
        state (AgentState): The current state of the agent, containing the topic and outline
        config (RunnableConfig): Configuration object containing LLM settings like provider, model, and temperature

    Returns:
        Dict: A dictionary containing the 'messages' key with the LLM's response about the report structure
    """
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
    """
    Handles human feedback on the generated report structure.

    This node prompts the user for feedback on the report structure and processes their response.
    If the user types 'continue', it proceeds to format the sections. Otherwise, it returns to the
    report structure planner with the feedback for revision.

    Args:
        state (AgentState): The current state containing the generated report structure messages
        config (RunnableConfig): Configuration object (unused in this node)

    Returns:
        Command: A Command object directing the flow either to:
            - "section_formatter" with the approved report structure
            - "report_structure_planner" with feedback for revision
    """
    
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
    """
    Formats the report structure into discrete sections for processing.

    This node takes the approved report structure and uses an LLM to format it into a structured
    Sections object containing individual sections and their subsections. The formatted sections
    are saved to a JSON file for logging and initialized in the state for sequential processing.

    Args:
        state (AgentState): The current state containing the approved report structure
        config (RunnableConfig): Configuration object containing LLM and other settings

    Returns:
        Command: A Command object directing flow to "queue_next_section" with:
            - sections: List of formatted Section objects
            - current_section_index: Initialized to 0 to begin processing
    """

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
    Manages the sequential processing of report sections with rate limiting.

    This node controls the flow of section processing by:
    1. Tracking the current section index
    2. Implementing delays between sections to avoid rate limits
    3. Routing sections to the research agent for processing
    4. Transitioning to report finalization when all sections are complete

    Args:
        state (AgentState): The current state containing sections and section index
        config (RunnableConfig): Configuration object containing delay settings

    Returns:
        Command: A Command object directing flow to either:
            - "research_agent" with the next section to process
            - "finalizer" when all sections are complete
    """
    configurable = Configuration.from_runnable_config(config)
    
    if state["current_section_index"] < len(state["sections"]):
        current_section = state["sections"][state["current_section_index"]]
        
        if state["current_section_index"] > 0:
            print(f"Waiting {configurable.section_delay_seconds} seconds before processing next section to avoid rate limits...")
            time.sleep(configurable.section_delay_seconds)
            
        print(f"Processing section {state['current_section_index'] + 1}/{len(state['sections'])}: {current_section.section_name}")
        
        return Command(
            update={"current_section_index": state["current_section_index"] + 1},
            goto=Send("research_agent", {"section": current_section, "current_section_index": state["current_section_index"]})
        )
    else:
        print(f"All {len(state['sections'])} sections have been processed. Generating final report...")
        return Command(goto="finalizer")


def section_knowledge_node(state: ResearchState, config: RunnableConfig):
    """
    Generates initial knowledge and understanding about a section before conducting research.

    This node uses an LLM to analyze the section details and generate foundational knowledge
    that will guide the subsequent research process. It processes the section information 
    through a system prompt to establish context and requirements.

    Args:
        state (ResearchState): The current research state containing section information
        config (RunnableConfig): Configuration object containing LLM and other settings

    Returns:
        dict: A dictionary containing the generated knowledge with key:
            - knowledge (str): The LLM-generated understanding and context for the section
    """
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
    """
    Generates search queries based on the current section content and research state.

    This node uses an LLM to generate targeted search queries for gathering information about
    the current section. It takes into account any previous queries that have been searched
    and feedback from reflection to avoid redundancy and improve query relevance.

    Args:
        state (ResearchState): The current research state containing section information,
            previous queries, and reflection feedback
        config (RunnableConfig): Configuration object containing LLM and other settings

    Returns:
        dict: A dictionary containing:
            - generated_queries (List[Query]): The newly generated search queries
            - searched_queries (List[Query]): Updated list of all searched queries
    """
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
    """
    Performs web searches using the Tavily search API for each generated query.

    This node takes the generated queries from the previous node and executes searches
    using the Tavily search engine. For each query, it retrieves search results up to
    the configured search depth, extracting the URL, title, and raw content from each result.

    Args:
        state (ResearchState): The current research state containing generated queries
            and other research context
        config (RunnableConfig): Configuration object containing search depth and other settings

    Returns:
        dict: A dictionary containing:
            - search_results (List[SearchResults]): List of search results for each query,
              where each SearchResults object contains the original query and a list of
              SearchResult objects with URL, title and raw content
    """
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
    """
    Accumulates and synthesizes search results into coherent content.

    This node takes the search results from the previous node and uses an LLM to process
    and combine them into a unified, coherent piece of content. The LLM analyzes the 
    search results and extracts relevant information to build knowledge about the section topic.

    Args:
        state (ResearchState): The current research state containing search results
            and other research context
        config (RunnableConfig): Configuration object containing LLM settings and other parameters

    Returns:
        dict: A dictionary containing:
            - accumulated_content (str): The synthesized content generated from processing
              the search results
    """
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
    """
    Evaluates the quality and completeness of accumulated research content and determines next steps.

    This node uses an LLM to analyze the current section's accumulated content and provide feedback
    on whether it adequately covers the section requirements. Based on the feedback and number of
    reflection iterations, it decides whether to proceed to final formatting or generate more queries
    for additional research.

    Args:
        state (ResearchState): The current research state containing the section info and
            accumulated content to evaluate
        config (RunnableConfig): Configuration object containing LLM settings and reflection parameters

    Returns:
        Command: A Command object directing the flow to either:
            - final_section_formatter: If content is sufficient or max reflections reached
            - query_generator: If content needs improvement and more iterations remain
            The Command includes updated reflection feedback and count in its state updates.
    """
    
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
    """
    Formats the final content for a section of the research report.

    This node uses an LLM to take the accumulated research content and internal knowledge
    about the section, and format it into a cohesive, well-structured section of the report.
    The formatted content is both saved to a log file and returned as part of the state.

    Args:
        state (ResearchState): The current research state containing the section info,
            internal knowledge, and accumulated content to format
        config (RunnableConfig): Configuration object containing LLM settings

    Returns:
        dict: A dictionary containing the formatted section content in the 'final_section_content' key
    """

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

    os.makedirs("logs/section_content", exist_ok=True)

    with open(f"logs/section_content/{state['current_section_index']+1}. {state['section'].section_name}.md", "a", encoding="utf-8") as f:
        f.write(f"{result.content}")

    return {"final_section_content": [result.content]}


def finalizer_node(state: AgentState, config: RunnableConfig):
    """
    Finalizes the research report by generating a conclusion, references, and combining all sections.

    This node takes the accumulated section content and search results from the agent state and:
    1. Uses an LLM to generate a conclusion and curated list of references
    2. Combines all section content into a single markdown document
    3. Saves the final report to a file
    
    Args:
        state (AgentState): The current agent state containing all section content and search results
        config (RunnableConfig): Configuration object containing LLM settings

    Returns:
        dict: A dictionary containing the complete report content in the 'final_report_content' key
    """

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
    
    with open(f"reports/{state['topic']}.md", "w", encoding="utf-8") as f:
        f.write(final_report)

    return {"final_report_content": final_report}