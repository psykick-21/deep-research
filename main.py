import uuid
from deep_research.graph import graph

def main():
    TOPIC = "Support Vector Machines"
    OUTLINE = "I am studying machine learning and I want to understand Support Vector Machines."

    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }

    for event in graph.stream(
        {"topic": TOPIC, "outline": OUTLINE},
        config=thread,
    ):
        if "report_structure_planner" in event:
            print("<<< REPORT STRUCTURE PLANNER >>>")
            print(event["report_structure_planner"]["messages"][-1].content)
        else:
            print("<<< HUMAN FEEDBACK >>>")
            print(event["human_feedback"]["messages"][-1].content)

if __name__ == "__main__":
    main()