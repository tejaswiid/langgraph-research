from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated, Literal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Define Search Agent
class SearchAgent:
    def __init__(self):
        self.search_tool = TavilySearchResults(max_results=2)
        self.tool_node = ToolNode(tools=[self.search_tool])
    
    def perform_search(self, state: MessagesState):
        messages = state["messages"]
        query = messages[-1].content  # Extract query from the last message
        search_results = self.search_tool.run(query)
        return {"messages": [search_results]}

# Define Reasoning Agent
class ReasoningAgent:
    def __init__(self):
        self.llm = ChatGroq(model_name="Gemma2-9b-It")
    
    def generate_response(self, state: MessagesState):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def router_function(self, state: MessagesState) -> Literal["search", END]:
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "search"  # Route to search
        return END  # End the conversation

# Define Workflow Manager
class MultiAgentRAG:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.reasoning_agent = ReasoningAgent()

    def create_workflow(self):
        # Initialize workflow
        workflow = StateGraph(MessagesState)

        # Add agents to the workflow
        workflow.add_node("reasoning", self.reasoning_agent.generate_response)
        workflow.add_node("search", self.search_agent.tool_node)

        # Define workflow edges
        workflow.add_edge(START, "reasoning")
        workflow.add_conditional_edges(
            "reasoning",
            self.reasoning_agent.router_function,
            {"search": "search", END: END}
        )
        workflow.add_edge("search", "reasoning")

        # Compile workflow
        return workflow.compile()

# Main Execution
if __name__ == "__main__":
    rag_system = MultiAgentRAG()
    workflow_app = rag_system.create_workflow()

    # Example Query
    user_query = {"messages": ["Who is the current Prime Minister of Russia?"]}
    response = workflow_app.invoke(user_query)
    print(response["messages"][-1].content)
