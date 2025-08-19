# Standart library imports
import os
import sys
from datetime import datetime
from typing import Annotated, TypedDict
from zoneinfo import ZoneInfo

# Thirdparty imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

# from langchain_community.chat_models import ChatOllama


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Local imports
from config import LLMConfig


# Define the state for our graph - contains the history of conversation messages
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# TODO: define a tool function to get the current date
@tool
def get_current_date() -> str:
    """function for get current date"""
    print("--- Calling get_current_date tool ---")

    # return datetime.now().strftime("%A, %B %d, %Y")
    return datetime.now(ZoneInfo("Europe/Kyiv")).strftime("%A, %B %d, %Y")


class SummarizerAgent:
    def __init__(self, model_name: str = "phi3") -> None:
        # 1. Define the tools the agent can use
        self.tools = [get_current_date]  # TODO: define the tools available to the agent

        # 2. Set up the language model with temperature from config
        self.llm = ChatOllama(
            model=model_name, temperature=LLMConfig.TEMPERATURE
        )  # TODO: create ChatOllama instance with model_name and temperature from LLMConfig

        # TODO: If you have problems running local Ollama models, you can use OpenAI or Gemini API from Langchain

        # 3. Bind the tools to the model to enable tool calling capabilities
        # TODO: bind the tools to the llm
        self.model_with_tools = self.llm.bind_tools(self.tools)

        # 4. Build and compile the graph that defines the agent's logic
        self.graph = self._build_graph()
        # self.graph = create_react_agent(self.llm, self.tools)

    def tool_router(self, state: AgentState) -> str:
        """
        Router function that determines the next step in the conversation graph.

        This method examines the last message in the conversation state to decide
        whether the agent should call tools or end the conversation. It's a key
        component of the LangGraph's conditional routing logic.

        Parameters
        ----------
        state : AgentState
            The current state of the agent containing the message history.

        Returns
        -------
        str
            Either "tools" if the last message contains tool calls that need to be
            executed, or END to terminate the conversation flow.

        Notes
        -----
        This function checks if the last message has tool_calls attribute and
        if those tool calls exist, indicating that the model wants to use tools
        to answer the user's question.
        """
        last_message = state["messages"][-1]

        # Check if the last message contains tool calls that need to be executed
        if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
            print("Decision: Call tools.")
            return "tools"
        else:
            print("Decision: End.")
            return END

    def sum_agent(self, state: AgentState) -> AgentState:
        """
        Main agent function that processes messages and generates responses.

        This method creates a prompt template with the system prompt and message
        history, then invokes the language model to generate a response. It handles
        both regular text responses and tool calls based on the user's input.

        Parameters
        ----------
        state : AgentState
            The current state containing the conversation message history.

        Returns
        -------
        AgentState
            The updated state with the agent's response added to the message history.

        Notes
        -----
        This function uses a ChatPromptTemplate that includes the system prompt
        from LLMConfig and a placeholder for the message history. The response
        is automatically added to the state's message list using add_messages.
        """
        # Create prompt template with system instructions and message history
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", LLMConfig.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="history"),
            ]
        )

        # Create the processing chain: prompt -> model -> response
        # TODO: create a chain that uses the prompt_template and model_with_tools
        chain = prompt_template | self.model_with_tools

        # Invoke the chain with the conversation history
        # TODO: invoke the chain with state["messages"] as input
        chain_result = chain.invoke({"history": state["messages"]})

        # Add the model's response to the message history
        # Note: Type checking issues here are due to LangGraph's complex typing
        # The add_messages function properly handles message accumulation
        return {"messages": add_messages(state["messages"], chain_result)}  # type: ignore

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build a cyclic graph for the summarizer agent that can call tools.

        Flow:
            START -> agent -> conditional:
                - If tool call: tools -> agent (loop)
                - If no tool call: END
        """
        # Initialize the state graph
        graph_builder = StateGraph(AgentState)
        tool_node = ToolNode(self.tools)
        graph_builder.add_node("agent", self.sum_agent)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", self.tool_router, {"tools": "tools", END: END})
        graph_builder.add_edge("tools", "agent")

        # Compile and return the runnable graph
        return graph_builder.compile()

    def execute(self, input_message: str) -> str:
        # Initialize the conversation state with the user's message
        initial_state = {"messages": [HumanMessage(content=input_message)]}

        # Process the message through the state graph
        final_state = self.graph.invoke(initial_state)

        # Return the content of the final response message
        return final_state["messages"][-1].content
