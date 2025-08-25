"""
Summarizer Graph Module

This module implements an AI agent using LangGraph that can summarize text
and answer questions using available tools. The agent is built on top of
Ollama's language models and uses a graph-based approach to handle
conversation flow and tool usage.

Classes
-------
AgentState : TypedDict
    State representation for the conversational agent.
SummarizerAgent : class
    Main agent class that handles text summarization and tool-based queries.

Functions
---------
get_current_date : function
    Tool function that returns the current date.

Examples
--------
>>> agent = SummarizerAgent(model_name="phi3")
>>> response = agent.execute("What is today's date?")
>>> print(response)
"""

# Standart library imports
# Standard library imports
from datetime import datetime
from typing import Annotated, TypedDict

# Thirdparty imports
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Third-party imports
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

# Local imports
from config import LLMConfig


# Define the state for our graph - contains the history of conversation messages
class AgentState(TypedDict):
    """
    Represents the state of our conversational agent.

    This class defines the structure of the state that gets passed between
    nodes in the LangGraph. It maintains the conversation history using
    LangChain's message format.

    Attributes
    ----------
    messages : Annotated[list[BaseMessage], add_messages]
        The history of messages in the conversation. Uses LangChain's
        add_messages function to properly handle message accumulation
        and deduplication during graph execution.

    Notes
    -----
    The Annotated type with add_messages enables automatic message
    handling within the LangGraph framework, ensuring proper state
    management across graph nodes.
    """

    messages: Annotated[list[BaseMessage], add_messages]


# define a tool function to get the current date
@tool
def get_current_date() -> str:
    """
    Get the current date in a human-readable format.

    This tool function returns the current date formatted as a string.
    It's designed to be used by the AI agent when users ask about the
    current date or when date information is needed for context.

    Returns
    -------
    str
        The current date formatted as "Weekday, Month DD, YYYY"
        (e.g., "Monday, January 15, 2024").

    Examples
    --------
    >>> get_current_date()
    'Monday, January 15, 2024'

    Notes
    -----
    This function uses Python's datetime module to get the current
    system date and formats it using strftime with the pattern
    "%A, %B %d, %Y".
    """
    print("--- Calling get_current_date tool ---")
    return datetime.now().strftime("%A, %B %d, %Y")


class SummarizerAgent:
    """
    AI agent for text summarization and tool-based question answering.

    This class implements a conversational AI agent that can either summarize
    given text or use available tools to answer specific questions. The agent
    is built using LangGraph for state management and conversation flow control,
    with Ollama as the underlying language model.

    Parameters
    ----------
    model_name : str, optional
        The name of the Ollama model to use for text generation.
        Default is "phi3".

    Attributes
    ----------
    tools : list
        List of tools available to the agent for answering questions.
    llm : ChatOllama
        The language model instance used for text generation.
    model_with_tools : ChatOllama
        The language model bound with available tools.
    graph : CompiledStateGraph
        The compiled state graph that defines the agent's logic flow.

    Methods
    -------
    should_continue(state: AgentState) -> str
        Router function to decide the next step in the conversation flow.
    sum_agent(state: AgentState) -> AgentState
        Main agent function that processes messages and generates responses.
    execute(input_message: str) -> str
        Execute the agent with a given input message and return the response.

    Examples
    --------
    >>> agent = SummarizerAgent(model_name="phi3")
    >>> response = agent.execute("What is today's date?")
    >>> print(response)
    'Wednesday, August 07, 2025'

    >>> response = agent.execute("Summarize this text: The quick brown fox...")
    >>> print(response)
    'Summary of the provided text...'
    """

    def __init__(self, model_name: str = "phi3") -> None:
        """
        Initialize the SummarizerAgent with specified model and tools.

        Parameters
        ----------
        model_name : str, optional
            The name of the Ollama model to use, by default "phi3"
        """
        # 1. Define the tools the agent can use
        self.tools = []

        self.tools.append(get_current_date)

        # 2. Set up the language model with temperature from config
        self.llm = ChatOllama(model=model_name, temperature=LLMConfig.TEMPERATURE)

        # 3. Bind the tools to the model to enable tool calling capabilities
        self.model_with_tools = self.llm.bind_tools(self.tools)

        # 4. Build and compile the graph that defines the agent's logic
        self.graph = self._build_graph()

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
        chain = prompt_template | self.model_with_tools

        # Invoke the chain with the conversation history
        chain_result = chain.invoke({"history": state["messages"]})

        # Add the model's response to the message history
        # Note: Type checking issues here are due to LangGraph's complex typing
        # The add_messages function properly handles message accumulation
        return {"messages": add_messages(state["messages"], chain_result)}  # type: ignore

    def _build_graph(self) -> CompiledStateGraph:
        """
        Build and compile the state graph that defines the agent's logic flow.

        This method constructs a LangGraph StateGraph that defines how the agent
        processes conversations. The graph includes nodes for the main agent logic
        and tool execution, with conditional edges that route between them based
        on whether tools need to be called.

        Returns
        -------
        CompiledStateGraph
            The compiled state graph ready for execution.

        Notes
        -----
        The graph structure:
        - START -> "agent": Initial entry point
        - "agent" -> conditional: Based on should_continue decision
        - "agent" -> "tools" | END: Either call tools or end conversation
        - "tools" -> "agent": Return to agent after tool execution
        """
        # Initialize the state graph with AgentState structure
        graph_builder = StateGraph(AgentState)

        # Add the main agent node that processes messages and generates responses
        graph_builder.add_node("agent", self.sum_agent)

        # Add the tools node that executes any requested tool calls
        tool_node = ToolNode(self.tools)
        graph_builder.add_node("tools", tool_node)

        # Define the conversation flow edges
        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", self.tool_router)
        graph_builder.add_edge("tools", "agent")

        # Compile and return the executable graph
        return graph_builder.compile()

    def execute(self, input_message: str) -> str:
        """
        Execute the agent with a given input message and return the response.

        This method serves as the main entry point for interacting with the agent.
        It takes a user message, processes it through the state graph, and returns
        the agent's final response.

        Parameters
        ----------
        input_message : str
            The user's input message or question to be processed by the agent.

        Returns
        -------
        str
            The agent's response after processing the input through the graph.
            This could be a direct answer, a summary, or the result of tool usage.

        Examples
        --------
        >>> agent = SummarizerAgent()
        >>> response = agent.execute("What is today's date?")
        >>> print(response)
        'Wednesday, August 07, 2025'

        >>> response = agent.execute("Summarize this: Machine learning is...")
        >>> print(response)
        'Machine learning summary...'

        Notes
        -----
        The method initializes the conversation state with the input message
        and invokes the compiled graph to process it. The final response is
        extracted from the last message in the conversation history.
        """
        # Initialize the conversation state with the user's message
        initial_state = {"messages": [HumanMessage(content=input_message)]}

        # Process the message through the state graph
        final_state = self.graph.invoke(initial_state)

        # Return the content of the final response message
        return final_state["messages"][-1].content
