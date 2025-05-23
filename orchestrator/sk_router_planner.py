import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments

from agents.multi_judge_agent import run_multi_judge_agents
from agents.literature_agent import run_literature_agent_stream
from agents.document_agent import DocumentQAAgent

load_dotenv()

kernel = Kernel()

azure_chat_completion = AzureChatCompletion(
    deployment_name="gpt-4.1",
    endpoint="https://qiuyu-m9lw5wkq-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    service_id="azure_openai"
)
kernel.add_service(azure_chat_completion)

system_prompt = """
You are an academic AI assistant. 
Select ONLY the most relevant skill for the user's query:
- 'multi_judge_plugin': Recommend classic, seminal, or foundational research papers (does not perform online search, only for classic and must-read requests).
- 'literature_plugin': Retrieve the latest or most up-to-date research papers using online sources like arXiv or web (for requests mentioning recent, latest, 2023/2024, code, datasets, benchmarks, or anything needing external search).
- 'qa_plugin': Handle file uploads, document analysis, and retrieval-augmented generation (RAG) discussions based on user-uploaded documents. Use for answering questions about user files or for RAG-based academic discussions.
"""

async def multi_agent_dispatch_stream(user_input: str) -> str:
    print(f"\n---\nUser input: {user_input}\n---")
    result = await kernel.invoke_prompt(
        prompt=system_prompt + "\nUser: " + user_input,
        arguments=KernelArguments(input=user_input),
        settings=PromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        ),
    )
    print("Result from kernel:", result)
    
    result_str = str(result).lower()

    if "multi_judge_plugin" in result_str:
        async for token in run_multi_judge_agents(user_input):
            yield token
    elif "literature_plugin" in result_str:
        async for token in run_literature_agent_stream(user_input):
            yield token
    elif "qa_plugin" in result_str:
        document_qa_agent = DocumentQAAgent()
        async for token in document_qa_agent.run_document_agent_stream(user_input):
            yield token
    else:
        async for token in run_literature_agent_stream(user_input):
            yield token # default feedback