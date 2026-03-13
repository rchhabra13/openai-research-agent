# OpenAI Research Agent

Multi-agent research system using OpenAI's Agents SDK. Conducts comprehensive research by coordinating specialized agents that gather information, analyze content, and synthesize detailed reports. Includes web search integration and automatic fact collection with source attribution.

## Features

- **Multi-Agent Architecture**: Triage, Research, and Editor agents working in coordination
- **Web Search Integration**: Real-time information gathering from the internet
- **Structured Reports**: Markdown-formatted reports with outlines and sources
- **Fact Tracking**: Automatic capture of important findings with source attribution
- **Real-Time Progress**: Live monitoring of research workflow
- **Report Download**: Export findings as markdown documents

## Quick Start

```bash
git clone https://github.com/rchhabra13/opeani_research_agent.git
cd opeani_research_agent
pip install -r requirements.txt

export OPENAI_API_KEY="your-openai-api-key"
streamlit run research_agent.py
```

Access at `http://localhost:8501`. Enter a research topic to begin.

## Agent Architecture

| Agent | Role | Function |
|-------|------|----------|
| Triage | Coordinator | Plans research strategy and queries |
| Research | Searcher | Gathers information from web sources |
| Editor | Writer | Synthesizes findings into comprehensive reports |

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| Model | gpt-4o-mini | Language model for agents |
| Report Length | 1000+ words | Target minimum content |
| Polling Interval | 1 second | Fact collection check frequency |

## Tech Stack

Python, Streamlit, OpenAI Agents SDK, Pydantic, python-dotenv

## License

MIT

**Credit**: Rishi Chhabra (rchhabra13)
