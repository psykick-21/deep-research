from deep_research.graph import agent_graph
from IPython.display import Image, display
import uuid
import os

def main():
    # TOPIC = "Human Psychology"
    # OUTLINE = "To understand human behaviour about why they always take actions that are selfish and in their best interest, leaving the rest of the world to suffer, including the ones who had supported and loved them all along."
    TOPIC = "LLM Benchmarking"
    OUTLINE = "Since there are so many LLMs out there, I want to understand how they compare to each other in terms of their capabilities, performance, and effectiveness. I have seen so many benchmark scores, but I want to understand the most important ones, what they measure, and what they mean."
    
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "max_queries": 3,
            "search_depth": 3,
            "num_reflections": 3,
            "temperature": 0.7
        }
    }

    os.makedirs("logs", exist_ok=True)
    os.makedirs("report", exist_ok=True)

    for event in agent_graph.stream(
        {"topic": TOPIC, "outline": OUTLINE},
        config=thread,
    ):
        with open("logs/agent_logs.txt", "a") as f:
            f.write(str(event))
            f.write("\n\n\n")

        if "report_structure_planner" in event:
            print("<<< REPORT STRUCTURE PLANNER >>>")
            print(event["report_structure_planner"]["messages"][-1].content)
            print("\n", "="*100, "\n")
        elif "section_formatter" in event:
            pass
        elif "research_agent" in event:
            pass
        elif "human_feedback" in event:
            print("<<< HUMAN FEEDBACK >>>")
            print(event["human_feedback"]["messages"][-1].content)
            print("\n", "="*100, "\n")
        elif "queue_next_section" in event:
            pass
        elif "finalizer" in event:
            print("Final report complete.")
        else:
            print("<<< UNKNOWN EVENT >>>")
            print(event)
            print("\n", "="*100, "\n")
    

if __name__ == "__main__":
    main()