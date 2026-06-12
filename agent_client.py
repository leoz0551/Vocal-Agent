import os
from dotenv import load_dotenv
from textwrap import dedent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.arxiv import ArxivTools
from agno.run.response import RunResponse

load_dotenv()

# Completely disable Agno telemetry to prevent 1-2s synchronous network delays
os.environ["AGNO_TELEMETRY"] = "false"
os.environ["PHIDATA_TELEMETRY"] = "false"

# Define the Knowledge Agent
knowledge_agent_ai = Agent(
    model=OpenAIChat(
        id=os.getenv("MODEL_ID", "openai/gpt-4o-mini"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL", "https://openrouter.ai/api/v1")
    ),
    tools=[
        GoogleSearchTools(),  # For performing web searches
        WikipediaTools(),  # For searching Wikipedia content
        ArxivTools(),  # For searching Arxiv publications
    ],
    instructions=dedent("""\
        You are a knowledge assistant that answers questions concisely. Use the available tools ONLY when necessary:
        - Google Search for general queries and information
        - Wikipedia for facts and history
        - Arxiv for research and papers
        - IMPORTANT: If the user just says a greeting (like 'Hello') or asks a normal conversational question, answer directly WITHOUT using any tools.
        - CRITICAL: When using tools, if a parameter expects a list, you MUST pass a valid JSON list (e.g. []), NOT a string (e.g. '[]').
        Do not use special characters or emojis in your responses.
        Note: Provide clear conversational response in 1-2 sentences and The response should be natural and engaging, and the length depends on what you have to say"""),
    add_datetime_to_instructions=True,
    show_tool_calls=False,
    markdown=True,
    stream=False,
    telemetry=False,
    add_history_to_messages=True,
    num_history_responses=10,
)

def create_streaming_agent(system_prompt: str) -> Agent:
    """Factory method for creating a per-session agent with custom instructions."""
    return Agent(
        model=OpenAIChat(
            id=os.getenv("MODEL_ID", "openai/gpt-4o-mini"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL", "https://openrouter.ai/api/v1")
        ),
        tools=[],  # V2 Realtime voice agent defaults to no tools for fastest response
        instructions=system_prompt,
        add_datetime_to_instructions=True,
        show_tool_calls=False,
        markdown=True,
        stream=True, # Enable streaming for v2
        telemetry=False,
        add_history_to_messages=True,
        num_history_responses=10,
    )

def knowledge_agent_client_stream(agent: Agent, prompt: str):
    """Yields streaming chunks of text from the agent."""
    try:
        response_stream = agent.run(message=prompt, stream=True)
        for chunk in response_stream:
            if isinstance(chunk, RunResponse) and chunk.content:
                yield chunk.content
            elif isinstance(chunk, str):
                yield chunk
    except Exception as e:
        print(f"Error while querying knowledge_agent_stream: {str(e)}")
        yield f"Error: {str(e)}"
def knowledge_agent_client(prompt: str):
    try:
        response = knowledge_agent_ai.run(message=prompt, stream=False)
        if isinstance(response, RunResponse):
            return response.content  
        else:
            print("Error: Invalid response from knowledge_agent_ai.")
            return None  
    except Exception as e:
        print(f"Error while querying knowledge_agent: {str(e)}")
        return None


if __name__ == "__main__":

   # Example usage with a different query
   message = "what is an mcp server use in ai agent"
   print(f'{knowledge_agent_client(message)}')
