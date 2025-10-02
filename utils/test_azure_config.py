import os
import sys
import json
import openai
from autogen import Agent, ConversableAgent, AssistantAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from utils.openai_client import azure_config_list

def test_azure_openai():
    commander = AssistantAgent(
        name="Commander",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    )

    instructor = MultimodalConversableAgent(
        name="Instructor",
        llm_config={"config_list": [azure_config_list[0]], "max_tokens": 3000},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
    )
    query = "describe the weather in Seattle"
    user_message = f"Please {query}."

    commander.send(
        message=user_message,
        recipient=instructor,
        request_reply=True,
    )

    response = commander._oai_messages[instructor][-1]["content"]
    print("Instructor response:", response)

if __name__ == "__main__":
    test_azure_openai()