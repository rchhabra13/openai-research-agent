"""OpenAI research agent with multi-agent system for comprehensive research.

This module creates a multi-agent system using OpenAI's Agents SDK to conduct
comprehensive research on topics by coordinating between specialized agents:
- Triage Agent: Plans research strategy
- Research Agent: Performs web searches
- Editor Agent: Synthesizes findings into reports
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from agents import (
    Agent,
    Runner,
    WebSearchTool,
    function_tool,
    handoff,
    trace,
)
from dotenv import load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Configuration
load_dotenv()
MODEL_ID: str = "gpt-4o-mini"
POLLING_INTERVAL: int = 1
MAX_POLLING_ATTEMPTS: int = 15
REPORT_MIN_LENGTH: int = 1000
REPORT_WORDS_ESTIMATE: str = "5-10 pages"

# Validate API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Please set your OPENAI_API_KEY environment variable")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="OpenAI Researcher Agent",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("OpenAI Researcher Agent")
st.subheader("Powered by OpenAI Agents SDK")
st.markdown(
    """
    Multi-agent research system for comprehensive topic investigation.
    Coordinates between specialized agents to gather, analyze, and
    synthesize information into detailed reports.
    """
)


# Data models
class ResearchPlan(BaseModel):
    """Research plan structure."""

    topic: str
    search_queries: List[str]
    focus_areas: List[str]


class ResearchReport(BaseModel):
    """Research report structure."""

    title: str
    outline: List[str]
    report: str
    sources: List[str]
    word_count: int


# Tools
@function_tool
def save_important_fact(fact: str, source: Optional[str] = None) -> str:
    """Save important fact during research.

    Args:
        fact (str): The fact to save.
        source (Optional[str]): Source of the fact.

    Returns:
        str: Confirmation message.
    """
    if "collected_facts" not in st.session_state:
        st.session_state.collected_facts = []

    st.session_state.collected_facts.append(
        {
            "fact": fact,
            "source": source or "Not specified",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )

    return f"Fact saved: {fact}"


# Initialize agents
research_agent: Agent = Agent(
    name="Research Agent",
    instructions=(
        "You are a research assistant. Search the web and produce concise "
        "summaries of results. Keep summaries to 2-3 paragraphs, less than "
        "300 words. Capture main points. Be succinct and focus on essence. "
        "Ignore fluff. Return only the summary."
    ),
    model=MODEL_ID,
    tools=[WebSearchTool(), save_important_fact],
)

editor_agent: Agent = Agent(
    name="Editor Agent",
    handoff_description="Senior researcher for comprehensive reports",
    instructions=(
        "You are a senior researcher writing a cohesive research report. "
        "You receive research query and initial research from a research "
        "assistant. Create an outline describing report structure and flow. "
        "Generate detailed markdown report. Aim for 5-10 pages, at least "
        "1000 words."
    ),
    model=MODEL_ID,
    output_type=ResearchReport,
)

triage_agent: Agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are research coordinator. Your job is to: "
        "1. Understand research topic "
        "2. Create research plan with: topic, search_queries (3-5), "
        "focus_areas (3-5) "
        "3. Hand off to Research Agent to gather information "
        "4. Hand off to Editor Agent for comprehensive report "
        "Return plan in structured format."
    ),
    handoffs=[handoff(research_agent), handoff(editor_agent)],
    model=MODEL_ID,
    output_type=ResearchPlan,
)


# Sidebar configuration
with st.sidebar:
    st.header("Research Topic")
    user_topic: str = st.text_input("Enter a topic to research:")

    start_button: bool = st.button(
        "Start Research", type="primary", disabled=not user_topic
    )

    st.divider()
    st.subheader("Example Topics")
    example_topics: List[str] = [
        "Best cruise lines for first-time travelers",
        "Affordable espresso machines for home use",
        "Off-the-beaten-path destinations in India",
    ]

    for topic in example_topics:
        if st.button(topic):
            user_topic = topic
            start_button = True


# Main UI
tab1, tab2 = st.tabs(["Research Process", "Report"])


# Session state initialization
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4().hex[:16])
if "collected_facts" not in st.session_state:
    st.session_state.collected_facts = []
if "research_done" not in st.session_state:
    st.session_state.research_done = False
if "report_result" not in st.session_state:
    st.session_state.report_result = None


async def run_research(topic: str) -> None:
    """Execute research workflow.

    Args:
        topic (str): Research topic.
    """
    st.session_state.collected_facts = []
    st.session_state.research_done = False
    st.session_state.report_result = None

    with tab1:
        message_container = st.container()

    with trace("Research", group_id=st.session_state.conversation_id):
        # Triage phase
        with message_container:
            st.write("Planning research approach...")

        triage_result = await Runner.run(
            triage_agent,
            f"Research this topic thoroughly: {topic}. "
            f"This research will be used to create a comprehensive report.",
        )

        # Extract plan
        if hasattr(triage_result.final_output, "topic"):
            research_plan = triage_result.final_output
            plan_display = {
                "topic": research_plan.topic,
                "search_queries": research_plan.search_queries,
                "focus_areas": research_plan.focus_areas,
            }
        else:
            plan_display = {
                "topic": topic,
                "search_queries": ["Researching " + topic],
                "focus_areas": ["General information about " + topic],
            }

        with message_container:
            st.write("Research Plan:")
            st.json(plan_display)

        # Fact collection phase
        fact_placeholder = message_container.empty()
        previous_fact_count = 0

        for _ in range(MAX_POLLING_ATTEMPTS):
            current_facts = len(st.session_state.collected_facts)
            if current_facts > previous_fact_count:
                with fact_placeholder.container():
                    st.write("Collected Facts:")
                    for fact in st.session_state.collected_facts:
                        st.info(
                            f"**Fact**: {fact['fact']}\n\n"
                            f"**Source**: {fact['source']}"
                        )
                previous_fact_count = current_facts
            await asyncio.sleep(POLLING_INTERVAL)

        # Editor phase
        with message_container:
            st.write("Creating comprehensive research report...")

        try:
            report_result = await Runner.run(
                editor_agent, triage_result.to_input_list()
            )

            st.session_state.report_result = report_result.final_output

            with message_container:
                st.write("✅ Research Complete! Report Generated.")

                if hasattr(report_result.final_output, "report"):
                    report_preview = (
                        report_result.final_output.report[:300] + "..."
                    )
                else:
                    report_preview = str(report_result.final_output)[:300] + "..."

                st.write("Report Preview:")
                st.markdown(report_preview)
                st.write("*See Report tab for full document.*")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            st.error(f"Error generating report: {str(e)}")

            if hasattr(triage_result, "new_items"):
                messages = [
                    item
                    for item in triage_result.new_items
                    if hasattr(item, "content")
                ]
                if messages:
                    raw_content = "\n\n".join(
                        [str(m.content) for m in messages if m.content]
                    )
                    st.session_state.report_result = raw_content

                    with message_container:
                        st.write(
                            "Research completed but report generation had issues."
                        )
                        st.write(
                            "Raw research results available in Report tab."
                        )

    st.session_state.research_done = True


# Run research
if start_button:
    with st.spinner(f"Researching: {user_topic}"):
        try:
            asyncio.run(run_research(user_topic))
        except Exception as e:
            logger.error(f"Research failed: {e}")
            st.error(f"An error occurred during research: {str(e)}")
            st.session_state.report_result = (
                f"# Research on {user_topic}\n\n"
                f"An error occurred. Please try again.\n\n"
                f"Error: {str(e)}"
            )
            st.session_state.research_done = True


# Display results
with tab2:
    if st.session_state.research_done and st.session_state.report_result:
        report = st.session_state.report_result

        if hasattr(report, "title"):
            title = report.title

            if hasattr(report, "outline") and report.outline:
                with st.expander("Report Outline", expanded=True):
                    for i, section in enumerate(report.outline):
                        st.markdown(f"{i+1}. {section}")

            if hasattr(report, "word_count"):
                st.info(f"Word Count: {report.word_count}")

            if hasattr(report, "report"):
                report_content = report.report
                st.markdown(report_content)
            else:
                report_content = str(report)
                st.markdown(report_content)

            if hasattr(report, "sources") and report.sources:
                with st.expander("Sources"):
                    for i, source in enumerate(report.sources):
                        st.markdown(f"{i+1}. {source}")

            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"{title.replace(' ', '_')}.md",
                mime="text/markdown",
            )
        else:
            report_content = str(report)
            title = user_topic.title()

            st.title(title)
            st.markdown(report_content)

            st.download_button(
                label="Download Report",
                data=report_content,
                file_name=f"{title.replace(' ', '_')}.md",
                mime="text/markdown",
            )
