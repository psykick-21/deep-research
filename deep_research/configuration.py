from dataclasses import dataclass
from langchain_core.runnables import RunnableConfig

@dataclass
class Configuration:
    
    @classmethod
    def from_runnable_config(
        cls,
        config: RunnableConfig
    ):
        pass