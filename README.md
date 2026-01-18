# AI Agent Toolkit

Build autonomous AI agents with tools, memory, and planning capabilities.

## Features

- 🤖 ReAct-style reasoning loop
- 🛠️ Extensible tool system
- 🧠 Short and long-term memory
- 📋 Task planning and decomposition
- 🔄 Multi-agent orchestration

## Quick Start

```bash
pip install -r requirements.txt
python examples/simple_agent.py
```

## Basic Agent

```python
from agent import Agent, Tool

# Define tools
@Tool
def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Results for: {query}"

@Tool
def calculate(expression: str) -> float:
    """Evaluate a math expression"""
    return eval(expression)

# Create agent
agent = Agent(
    tools=[search_web, calculate],
    model="gpt-4-turbo"
)

# Run
result = agent.run("What is 15% of the US population?")
```

## Multi-Agent System

```python
from agent import Agent, Orchestrator

researcher = Agent(name="researcher", tools=[search_web])
analyst = Agent(name="analyst", tools=[calculate, analyze_data])
writer = Agent(name="writer", tools=[write_document])

orchestrator = Orchestrator(agents=[researcher, analyst, writer])
result = orchestrator.run("Research AI trends and write a summary report")
```

## Architecture

```
User Query
    ↓
┌─────────────────────────────┐
│         Planner             │
│  (decompose into subtasks)  │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│       Executor Loop         │
│  Think → Act → Observe      │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│        Tool Calls           │
│  search, calculate, etc.    │
└─────────────────────────────┘
    ↓
Final Response
```

## License

MIT
