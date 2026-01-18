"""
AI Agent Toolkit
ReAct-style autonomous agents with tools and memory
"""

import json
import inspect
import os
from typing import Callable, List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from functools import wraps


def tool(func: Callable = None, *, name: str = None, description: str = None):
    """
    Decorator to mark a function as a tool for the agent.

    Usage:
        @tool
        def my_function(arg: str) -> str:
            '''Description here'''
            return result

        @tool(name="custom_name", description="Custom description")
        def another_function(x: int) -> int:
            return x * 2
    """
    def decorator(f: Callable) -> 'Tool':
        return Tool(f, name=name, description=description)

    if func is not None:
        # Called without arguments: @tool
        return decorator(func)
    # Called with arguments: @tool(name="...")
    return decorator


class Tool:
    """Wrapper class for tool functions"""

    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or f"Execute {self.name}"

        # Preserve function metadata
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI tool format"""
        sig = inspect.signature(self.func)
        type_hints = getattr(self.func, '__annotations__', {})

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Determine parameter type from annotations
            param_type = "string"
            hint = type_hints.get(param_name)

            if hint == int:
                param_type = "integer"
            elif hint == float:
                param_type = "number"
            elif hint == bool:
                param_type = "boolean"
            elif hint == list or (hasattr(hint, '__origin__') and hint.__origin__ == list):
                param_type = "array"

            properties[param_name] = {"type": param_type}

            # Check if parameter is required (no default value)
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


@dataclass
class Memory:
    """Simple memory store for agent conversations"""
    short_term: List[Dict] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)

    def add(self, role: str, content: str) -> None:
        """Add a message to short-term memory"""
        self.short_term.append({"role": role, "content": content})

    def get_context(self, max_messages: int = 20) -> List[Dict]:
        """Get recent conversation context"""
        return self.short_term[-max_messages:]

    def save_fact(self, key: str, value: Any) -> None:
        """Save a fact to long-term memory"""
        self.long_term[key] = value

    def get_fact(self, key: str, default: Any = None) -> Any:
        """Retrieve a fact from long-term memory"""
        return self.long_term.get(key, default)

    def clear_short_term(self) -> None:
        """Clear short-term memory"""
        self.short_term = []

    def clear_all(self) -> None:
        """Clear all memory"""
        self.short_term = []
        self.long_term = {}


class Agent:
    """ReAct-style AI agent with tools"""

    def __init__(
        self,
        tools: List[Union[Tool, Callable]] = None,
        model: str = "gpt-4-turbo",
        max_iterations: int = 10,
        system_prompt: str = None,
        api_key: str = None
    ):
        # Import here to allow tool-only usage without API key
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.memory = Memory()

        # Convert callables to Tools and build registry
        self.tools: Dict[str, Tool] = {}
        for t in (tools or []):
            if isinstance(t, Tool):
                self.tools[t.name] = t
            elif callable(t):
                tool_obj = Tool(t)
                self.tools[tool_obj.name] = tool_obj

        self.system_prompt = system_prompt or """You are an autonomous AI agent.
You have access to tools to help accomplish tasks.
Think step by step, use tools when needed, and provide a final answer.
Always explain your reasoning before using tools."""

    def add_tool(self, func: Callable, name: str = None, description: str = None) -> None:
        """Add a tool to the agent"""
        tool_obj = Tool(func, name=name, description=description)
        self.tools[tool_obj.name] = tool_obj

    def run(self, task: str, clear_memory: bool = False) -> str:
        """
        Execute a task using the ReAct loop.

        Args:
            task: The task to accomplish
            clear_memory: Whether to clear memory before running

        Returns:
            The agent's final response
        """
        if clear_memory:
            self.memory.clear_short_term()

        self.memory.add("user", task)

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory.get_context()
        ]

        openai_tools = [t.to_openai_tool() for t in self.tools.values()]

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None
            )

            message = response.choices[0].message

            # No tool calls - final response
            if not message.tool_calls:
                content = message.content or ""
                self.memory.add("assistant", content)
                return content

            # Process tool calls
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                # Execute tool
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name](**tool_args)
                    except Exception as e:
                        result = f"Error executing {tool_name}: {str(e)}"
                else:
                    result = f"Error: Unknown tool '{tool_name}'"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        return f"Max iterations ({self.max_iterations}) reached without final answer"


class Orchestrator:
    """Multi-agent orchestration for complex tasks"""

    def __init__(self, agents: Dict[str, Agent] = None, api_key: str = None):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.agents = agents or {}

    def add_agent(self, name: str, agent: Agent) -> None:
        """Add an agent to the orchestrator"""
        self.agents[name] = agent

    def run(self, task: str) -> Dict[str, Any]:
        """
        Orchestrate multiple agents to complete a task.

        Returns dict with 'plan' and 'results' keys.
        """
        if not self.agents:
            return {"error": "No agents configured"}

        # Create execution plan
        plan_response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a task planner. Available agents: {list(self.agents.keys())}.
Create a step-by-step plan assigning each step to an appropriate agent.
Format: One step per line, format: "AGENT_NAME: task description"
"""
                },
                {"role": "user", "content": task}
            ]
        )

        plan = plan_response.choices[0].message.content
        results = {"plan": plan, "agent_results": {}}

        # Execute with each agent
        for name, agent in self.agents.items():
            try:
                result = agent.run(
                    f"As '{name}', contribute to this task: {task}\n\nPlan:\n{plan}",
                    clear_memory=True
                )
                results["agent_results"][name] = result
            except Exception as e:
                results["agent_results"][name] = f"Error: {str(e)}"

        return results


# Built-in tools
@tool
def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression"""
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return f"Error: Invalid characters in expression"

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_current_time() -> str:
    """Get the current date and time"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    # Example usage
    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city"""
        weather = {"nyc": "72°F, sunny", "la": "85°F, clear", "london": "55°F, rainy"}
        return weather.get(city.lower(), f"Weather for {city}: 70°F, partly cloudy")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example")
        print("Example: export OPENAI_API_KEY=sk-...")
    else:
        agent = Agent(tools=[get_weather, calculate, get_current_time])
        result = agent.run("What's the weather in NYC and what's 15 * 23?")
        print(result)
