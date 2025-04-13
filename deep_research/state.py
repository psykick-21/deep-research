from typing import Annotated, List
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict
from .struct import Section, Feedback, Query, SearchResults


class AgentState(TypedDict):
    topic: str
    outline: str
    messages: Annotated[List[BaseMessage], operator.add]
    report_structure: str
    sections: List[Section]
    current_section_index: int
    final_section_content: Annotated[List[str], operator.add]
    search_results: Annotated[List[SearchResults], operator.add]


class ResearchState(TypedDict):
    section: Section
    knowledge: str
    reflection_feedback: Feedback
    generated_queries: List[Query]
    searched_queries: Annotated[List[Query], operator.add]
    search_results: Annotated[List[SearchResults], operator.add]
    accumulated_content: str
    reflection_count: int
    final_section_content: List[str]
    current_section_index: int