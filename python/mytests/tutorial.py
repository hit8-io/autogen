import asyncio # noqa

from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# Define a tool that searches the web for information.
# For simplicity, we will use a mock function here that returns a static string.
async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o-mini",
    model="gpt-4o-mini",
    api_version="2025-01-01-preview",
    azure_endpoint="https://egmni.openai.azure.com/",
)

async def simple():
    # Create an agent
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[web_search],
        system_message="Use tools to solve tasks.",
    )

    # Now we can use await inside this async function
    result = await agent.run(task="Find information on AutoGen")
    print(result.messages) # noqa

async def streaming():
    # Create an agent
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[web_search],
        system_message="Use tools to solve tasks.",
    )

    # Now we can use await inside this async function
    await Console(
        agent.run_stream(task="Find information on AutoGen"),
        output_stats=True,  # Enable stats printing.
    )

async def cot():
    # The response format for the agent as a Pydantic base model.
    class AgentResponse(BaseModel):
        thoughts: str
        response: str

    # Create an agent that uses the OpenAI GPT-4o model.
    agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="Categorize the input as happy, sad, or neutral following the JSON format.",
        # Define the output content type of the agent.
        output_content_type=AgentResponse,
    )

    result = await Console(agent.run_stream(task="I am happy."))

    # Check the last message in the result, validate its type, and print the thoughts and response.
    assert isinstance(result.messages[-1], StructuredMessage[AgentResponse])
    assert isinstance(result.messages[-1].content, AgentResponse)
    print("Thought: ", result.messages[-1].content.thoughts) # noqa
    print("Response: ", result.messages[-1].content.response) # noqa
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(cot())
