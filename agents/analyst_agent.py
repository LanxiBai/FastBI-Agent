"""Minimal LangGraph workflow for the FastBI analyst agent."""

from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from utils.config import get_model_settings


class AgentState(TypedDict):
    """Data passed between nodes in the analyst workflow."""

    user_input: str
    final_output: str


def create_chat_model() -> BaseChatModel:
    """Create the configured DeepSeek chat model."""
    settings = get_model_settings()
    return ChatOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    )


def analyst_node(state: AgentState) -> dict[str, str]:
    """Ask the LLM to respond as a concise business analyst."""
    model = create_chat_model()
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful business analyst. Respond clearly and concisely. "
                    "This workflow has no data-analysis tools yet."
                )
            ),
            HumanMessage(content=state["user_input"]),
        ]
    )
    return {"final_output": str(response.content)}


workflow = StateGraph(AgentState)
workflow.add_node("analyst", analyst_node)
workflow.add_edge(START, "analyst")
workflow.add_edge("analyst", END)

graph = workflow.compile()


def run_agent(user_input: str) -> str:
    """Run the analyst workflow and return its final text response."""
    if not user_input.strip():
        raise ValueError("user_input must not be empty.")

    result = graph.invoke({"user_input": user_input, "final_output": ""})
    return result["final_output"]
