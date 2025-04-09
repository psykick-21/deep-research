from typing import Annotated, List
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict
from .struct import Section, Feedback, Query, SearchResult

class AgentState(TypedDict):
    topic: str
    outline: str
    messages: Annotated[List[BaseMessage], operator.add]
    report_structure: str
    sections: List[Section]
    final_section_content: Annotated[List[str], operator.add]
    final_report_content: str

class ResearchState(TypedDict):
    section: Section
    knowledge: str
    reflection_feedback: Feedback
    generated_queries: List[Query]
    searched_queries: Annotated[List[Query], operator.add]
    search_results: Annotated[List[SearchResult], operator.add]
    accumulated_content: str
    reflection_count: int
    final_section_content: List[str]