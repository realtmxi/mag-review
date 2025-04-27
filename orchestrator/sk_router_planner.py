import os
import asyncio
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments

from agents.multi_judge_agent_1d import run_multi_judge_agents
from agents.literature_agent import run_literature_agent
from agents.qa_agent import run_qa_agent

load_dotenv()

kernel = Kernel()

azure_chat_completion = AzureChatCompletion(
    deployment_name="gpt-4.1",
    endpoint="https://qiuyu-m9lw5wkq-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    service_id="azure_openai"
)
kernel.add_service(azure_chat_completion)

# class MultiJudgePlugin(object):
#     """Plugin for classic/seminal paper recommendation."""
#     @kernel_function(name="recommend_classic")
#     async def recommend_classic(self, input: str) -> str:
#         return await run_multi_judge_agents(input)

# class LiteraturePlugin(object):
#     @kernel_function(name="search_latest")
#     async def search_latest(self, input: str) -> str:
#         return await run_literature_agent(input)

# class QAPlugin(object):
#     @kernel_function(name="doc_qa")
#     async def doc_qa(self, input: str) -> str:
#         return await run_qa_agent(input)

# kernel.add_plugin(MultiJudgePlugin(), plugin_name="multi_judge_plugin")
# kernel.add_plugin(LiteraturePlugin(), plugin_name="literature_plugin")
# kernel.add_plugin(QAPlugin(), plugin_name="qa_plugin")

system_prompt = """
You are an academic AI assistant. 
Select ONLY the most relevant skill for the user's query:
- 'multi_judge_plugin': Recommend classic, seminal, or foundational research papers (does not perform online search, only for classic and must-read requests).
- 'literature_plugin': Retrieve the latest or most up-to-date research papers using online sources like arXiv or web (for requests mentioning recent, latest, 2023/2024, code, datasets, benchmarks, or anything needing external search).
- 'qa_plugin': Handle file uploads, document analysis, and retrieval-augmented generation (RAG) discussions based on user-uploaded documents. Use for answering questions about user files or for RAG-based academic discussions.
"""

async def multi_agent_dispatch(user_input: str) -> str:
    print(f"\n---\nUser input: {user_input}\n---")
    result = await kernel.invoke_prompt(
        prompt=system_prompt + "\nUser: " + user_input,
        arguments=KernelArguments(input=user_input),
        # functions=[
        #     kernel.get_function("multi_judge_plugin", "recommend_classic"),
        #     kernel.get_function("literature_plugin", "search_latest"),
        #     kernel.get_function("qa_plugin", "doc_qa"),
        # ],
        settings=PromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto()
        ),
    )
    print("Result from kernel:", result)
    
    result_str = str(result).lower()

    if "recommend_classic" in result_str:
        return await run_multi_judge_agents(user_input)
    elif "search_latest" in result_str:
        return await run_literature_agent(user_input)
    elif "doc_qa" in result_str:
        return await run_qa_agent(user_input)
    else:
        return await run_literature_agent(user_input)  # default fallback