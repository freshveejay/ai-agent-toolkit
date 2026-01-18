"""
AI Agent Toolkit
ReAct-style autonomous agents with tools and memory
"""

import json
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class Tool:
    """Decorator to create a tool from a function"""
    func: Callable
    name: str = None
    description: str = None

    def __post_init__(self):
        self.name = self.name or self.func.__name__
        self.description = self.description or self.func.__doc__ or ""

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI tool format"""
        import inspect
        sig = inspect.signature(self.func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            param_type = "string"
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"

            properties[param_name] = {"type": param_type}

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
    """Simple memory store for agent"""
    short_term: List[Dict] = field(default_factory=list)
    long_term: Dict[str, Any] = field(default_factory=dict)

    def add(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})

    def get_context(self, max_messages: int = 20) -> List[Dict]:
        return self.short_term[-max_messages:]

    def save_fact(self, key: str, value: Any):
        self.long_term[key] = value

    def get_fact(self, key: str) -> Any:
        return self.long_term.get(key)


class Agent:
    """ReAct-style AI agent with tools"""

    def __init__(
        self,
        tools: List[Tool] = None,
        model: str = "gpt-4-turbo",
        max_iterations: int = 10,
        system_prompt: str = None
    ):
        self.client = OpenAI()
        self.tools = {t.name: t for t in (tools or [])}
        self.model = model
        self.max_iterations = max_iterations
        self.memory = Memory()

        self.system_prompt = system_prompt or """You are an autonomous AI agent.
You have access to tools to help accomplish tasks.
Think step by step, use tools when needed, and provide a final answer."""

    def run(self, task: str) -> str:
        """Execute a task using the ReAct loop"""

        self.memory.add("user", task)

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory.get_context()
        ]

        openai_tools = [t.to_openai_tool() for t in self.tools.values()]

        for i in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None
            )

            message = response.choices[0].message

            # No tool calls - final response
            if not message.tool_calls:
                self.memory.add("assistant", message.content)
                return message.content

            # Process tool calls
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute tool
                if tool_name in self.tools:
                    result = self.tools[tool_name](**tool_args)
                else:
                    result = f"Error: Unknown tool {tool_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        return "Max iterations reached without final answer"


class Orchestrator:
    """Multi-agent orchestration"""

    def __init__(self, agents: List[Agent]):
        self.agents = {a.name if hasattr(a, 'name') else f"agent_{i}": a
                       for i, a in enumerate(agents)}
        self.client = OpenAI()

    def run(self, task: str) -> str:
        """Orchestrate multiple agents to complete a task"""

        # Use GPT to create a plan
        plan_response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": f"You are a task planner. Available agents: {list(self.agents.keys())}. Create a step-by-step plan assigning each step to an agent."},
                {"role": "user", "content": task}
            ]
        )

        plan = plan_response.choices[0].message.content
        results = [f"Plan:\n{plan}\n"]

        # Execute with each agent (simplified)
        for name, agent in self.agents.items():
            result = agent.run(f"As part of this task: {task}\n\nDo your part as {name}.")
            results.append(f"\n{name} result:\n{result}")

        return "\n".join(results)


if __name__ == "__main__":
    # Example usage
    @Tool
    def get_weather(city: str) -> str:
        """Get weather for a city"""
        return f"Weather in {city}: 72°F, sunny"

    @Tool
    def calculate(expression: str) -> str:
        """Calculate a math expression"""
        return str(eval(expression))

    agent = Agent(tools=[get_weather, calculate])
    result = agent.run("What's the weather in NYC and what's 15 * 23?")
    print(result)
