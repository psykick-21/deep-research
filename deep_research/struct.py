from pydantic import BaseModel, Field
from typing import List, Union


class Section(BaseModel):
    section_name: str = Field(..., description="The name of this section of the report without its number")
    sub_sections: List[str] = Field(..., description="Comprehensive descriptions of sub-sections, each combining the sub-section title and its bullet points into a fluid, natural-language description")


class Sections(BaseModel):
    sections: List[Section] = Field(..., description="A list of sections")


class Query(BaseModel):
    query: str = Field(..., description="A search query")


class Queries(BaseModel):
    queries: List[Query] = Field(..., description="A list of search queries")


class SearchResult(BaseModel):
    url: str = Field(..., description="The url of the search result")
    title: str = Field(..., description="The title of the search result")
    raw_content: str = Field(..., description="The raw content of the search result")


class SearchResults(BaseModel):
    query: Query = Field(..., description="The search query that was used to retrieve the raw content")
    results: List[SearchResult] = Field(..., description="The search results")


class Feedback(BaseModel):
    feedback: Union[str, bool] = Field(..., description="Feedback on the report structure. If the content is good for the section, return True (boolean), otherwise return a string of feedback on what is missing or incorrect.")


class SectionOutput(BaseModel):
    final_section_content: List[str] = Field(..., description="The final section content")
    search_results: List[SearchResults] = Field(..., description="The search results")


class ConclusionAndReferences(BaseModel):
    conclusion: str = Field(
        ..., 
        description="A comprehensive synthesis of the report's key findings and their implications, offering insightful closure and highlighting the significance of the research without introducing new information not covered in the report sections"
    )
    references: List[str] = Field(
        ..., 
        description="A curated list of 5-6 most relevant and authoritative sources formatted in a consistent academic citation style, selected to support key findings and represent the most critical sources for understanding the report content"
    )