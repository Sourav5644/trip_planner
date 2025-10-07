import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import dotenv
from dotenv import load_dotenv

# ====================== SETUP ======================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# ====================== STATE ======================
class PlannerState(TypedDict):
    messages: List[HumanMessage]
    city: str
    interests: List[str]
    itinerary: str

# ====================== PROMPT ======================
itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on interests: {interests}."),
    ("human", "Generate a detailed, engaging one-day itinerary.")
])

# ====================== WORKFLOW FUNCTIONS ======================
def input_city(state: PlannerState) -> PlannerState:
    print("Please enter the city you want to visit for your day trip:")
    user_message = input("Your input: ")
    return {
        **state,
        "city": user_message,
        "messages": state["messages"] + [HumanMessage(content=user_message)],
    }

def input_interest(state: PlannerState) -> PlannerState:
    city = state["city"]
    print(f"\nFetching popular activities in {city}...")
    chosen_preferences = llm.invoke(f"List 5 popular activities to do in {city}.")
    print(chosen_preferences.content)

    print(f"\nPlease enter your interests for the trip to {city} (comma-separated):")
    user_message = input("Your input: ")

    return {
        **state,
        "interests": [interest.strip() for interest in user_message.split(",")],
        "messages": state["messages"] + [HumanMessage(content=user_message)],
    }

def create_itinerary(state: PlannerState) -> PlannerState:
    city = state["city"]
    interests = state["interests"]

    print(f"\nCreating an itinerary for {city} based on interests: {', '.join(interests)}...")

    response = llm.invoke(
        itinerary_prompt.format_messages(city=city, interests=", ".join(interests))
    )

    print("\n=== Final Itinerary ===")
    print(response.content)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "itinerary": response.content,
    }

# ====================== WORKFLOW GRAPH ======================
workflow = StateGraph(PlannerState)
workflow.add_node("input_city", input_city)
workflow.add_node("input_interest", input_interest)
workflow.add_node("create_itinerary", create_itinerary)

workflow.set_entry_point("input_city")
workflow.add_edge("input_city", "input_interest")
workflow.add_edge("input_interest", "create_itinerary")
workflow.add_edge("create_itinerary", END)

# No checkpointer (optional) if SqliteSaver is missing
app = workflow.compile(checkpointer=None)

# ====================== MAIN FUNCTION ======================
def travel_planner(user_request: str):
    print(f"Initial Request: {user_request}\n")
    state = {
        "messages": [HumanMessage(content=user_request)],
        "city": "",
        "interests": [],
        "itinerary": "",
    }

    # Run the workflow
    final_state = app.invoke(state)
    print("\n=== Trip Planning Completed ===")
    print(final_state["itinerary"])
    return final_state

# ====================== RUN ======================
if __name__ == "__main__":
    travel_planner("Plan a one-day trip for me.")
