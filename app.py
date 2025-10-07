import streamlit as st
from main_workflow import travel_planner, llm, itinerary_prompt
from langchain_core.messages import HumanMessage, AIMessage

# ----------------------------- Streamlit UI Setup -----------------------------
st.set_page_config(page_title="AI Trip Planner", page_icon="🗺️", layout="centered")

st.title("🧭 AI Travel Planner")
st.write("Plan your perfect **one-day trip** with AI – just enter your city and interests below!")

# ----------------------------- User Inputs -----------------------------
with st.form("trip_form"):
    city = st.text_input("🏙️ Enter the city you want to visit", placeholder="e.g., Manali")
    interests = st.text_input("🎯 Enter your interests (comma-separated)", placeholder="e.g., adventure, food, nature")
    submit = st.form_submit_button("Plan My Trip 🚀")
    

# ----------------------------- Process Inputs -----------------------------
if submit:
    if not city.strip():
        st.warning("Please enter a city name.")
    elif not interests.strip():
        st.warning("Please enter at least one interest.")
    else:
        with st.spinner("✨ Creating your personalized itinerary... Please wait!"):
            # Simulate workflow steps (integrate directly with your backend functions)
            # Step 1: Popular activities
            st.subheader(f"🌍 Popular Activities in {city}")
            activities_response = llm.invoke(f"List 5 popular activities to do in {city}.")
            st.markdown(activities_response.content)

            # Step 2: Generate itinerary
            interests_list = [i.strip() for i in interests.split(",")]
            itinerary_response = llm.invoke(
                itinerary_prompt.format_messages(city=city, interests=", ".join(interests_list))
            )

            # Step 3: Display itinerary
            st.subheader(f"🗓️ Your Personalized {city.title()} Itinerary")
            st.markdown(itinerary_response.content)

        st.success("✅ Trip Planning Completed Successfully!")

# ----------------------------- Footer -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using [LangGraph](https://github.com/langchain-ai/langgraph) and [Groq Llama 3.1](https://groq.com).")
