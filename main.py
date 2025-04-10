from deep_research.graph import agent_graph
from IPython.display import Image, display
import uuid

def main():
    TOPIC = "Support Vector Machines"
    OUTLINE = "I am studying machine learning and I want to understand Support Vector Machines."
    
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "max_queries": 2,
            "search_depth": 2,
            "num_reflections": 2,
            "provider": "ollama",
            "model": "gemma3:4b",
            "temperature": 0.3
        }
    }
    for event in agent_graph.stream(
        {"topic": TOPIC, "outline": OUTLINE},
        config=thread,
    ):
        with open("agent_logs.txt", "a") as f:
            f.write(str(event))
            f.write("\n\n\n")

        if "report_structure_planner" in event:
            print("<<< REPORT STRUCTURE PLANNER >>>")
            print(event["report_structure_planner"]["messages"][-1].content)
            print("\n", "="*100, "\n")
        elif "section_formatter" in event:
            print("<<< SECTION FORMATTING >>>")
            print(event["section_formatter"])
            print("\n", "="*100, "\n")
        elif "research_agent" in event:
            # check output of research_agent
            print("<<< RESEARCH AGENT >>>")
            print(event["research_agent"])
            print("\n", "="*100, "\n")
        elif "final_report_writer" in event:
            # check output of final_report_writer
            print("<<< FINAL REPORT WRITER >>>")
            print(event["final_report_writer"])
            print("\n", "="*100, "\n")
        else:
            print("<<< HUMAN FEEDBACK >>>")
            print(event["human_feedback"]["messages"][-1].content)
            print("\n", "="*100, "\n")
    

if __name__ == "__main__":
    main()