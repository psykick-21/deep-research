from typing import Annotated, List
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel
import operator

class AgentState(BaseModel):
    topic: str
    outline: str
    messages: Annotated[List[BaseMessage], operator.add]