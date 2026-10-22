import os
from datetime import date

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set.")

if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY is not set.")


model_name = os.getenv("MODEL", "openai/gpt-4o-mini")
today = date.today().strftime("%B %d, %Y")

search_tool = SerperDevTool()


researcher = Agent(
    role="Senior Technology News Researcher",
    goal=(
        "Find and verify the three most important technology news stories "
        "published today."
    ),
    backstory=(
        "You are an experienced technology journalist who searches reliable "
        "sources, checks publication dates, and summarizes complex developments "
        "clearly and accurately."
    ),
    tools=[search_tool],
    llm=model_name,
    verbose=True,
)


writer = Agent(
    role="Concise Technology Newsletter Writer",
    goal=(
        "Turn verified research into an engaging technology newsletter "
        "containing exactly five lines."
    ),
    backstory=(
        "You are a skilled newsletter editor who writes concise, informative "
        "updates for busy readers. You only use the research supplied to you "
        "and never conduct additional research."
    ),
    tools=[],
    llm=model_name,
    verbose=True,
)


find_news = Task(
    description=(
        f"Search the web for the top three technology news stories published "
        f"today, {today}. Verify that each story is current and comes from a "
        f"credible source. Return exactly three numbered items. For each item, "
        f"include the headline, a one-sentence summary, source name, and URL."
    ),
    expected_output=(
        "Exactly three numbered technology news items. Each item must contain "
        "a headline, one-sentence summary, credible source name, and source URL."
    ),
    agent=researcher,
)


write_newsletter = Task(
    description=(
        "Using only the Researcher's output, create a concise technology "
        "newsletter. Do not search for new information. The newsletter must "
        "contain exactly five newline-separated lines."
    ),
    expected_output=(
        "Exactly five lines with no blank lines: "
        "Line 1: 'Tech in 5 Lines'; "
        "Line 2: '1. <first story summary>'; "
        "Line 3: '2. <second story summary>'; "
        "Line 4: '3. <third story summary>'; "
        "Line 5: 'Why it matters: <one concise overall insight>'."
    ),
    agent=writer,
    context=[find_news],
)


crew = Crew(
    agents=[researcher, writer],
    tasks=[find_news, write_newsletter],
    process=Process.sequential,
    verbose=True,
)


result = crew.kickoff()

print("\nFINAL NEWSLETTER\n")
print(result)