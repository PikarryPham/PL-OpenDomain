import json
import logging
import os
from openai import OpenAI
from redis import InvalidResponse
import time
import numpy as np
import openai
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    XLMRobertaTokenizer,
    XLMRobertaModel,
    DistilBertTokenizer,
    DistilBertModel,
)
import torch

from functions import calculate_fixed_monthly_payment, calculate_future_value
from utils import setup_logging
from embeddings import (
    tfidf_embedder,
    bm25_embedder,
    bmx_embedder,
    load_all_base_models,
    get_embedding_tfidf,
    get_embedding_bm25,
    get_embedding_bmx,
    get_embedding_roberta,
    get_embedding_xlm_roberta,
    get_embedding_distilbert,
    get_embedding_hybrid_tfidf_bert,
    get_embedding_hybrid_bm25_bert,
    get_embedding_hybrid_bmx_bert,
)

logger = logging.getLogger(__name__)
setup_logging()

# Định nghĩa các mô hình embedding có sẵn
EMBEDDING_MODELS = {
    "openai": None,  # Sẽ được gán sau khi hàm get_embedding được định nghĩa
    "tfidf": get_embedding_tfidf,
    "bm25": get_embedding_bm25,
    "bmx": get_embedding_bmx,
    "roberta": get_embedding_roberta,
    "xlm-roberta": get_embedding_xlm_roberta,
    "distilbert": get_embedding_distilbert,
    "hybrid_tfidf_bert": get_embedding_hybrid_tfidf_bert,
    "hybrid_bm25_bert": get_embedding_hybrid_bm25_bert,
    "hybrid_bmx_bert": get_embedding_hybrid_bmx_bert,
}

# Mô hình mặc định
DEFAULT_EMBEDDING_MODEL = "openai"

# Tải các mô hình đã lưu khi khởi động
try:
    logger.info("Loading saved embedding models...")
    # Thử tải các mô hình đã lưu
    load_all_base_models()
    logger.info("Successfully loaded saved embedding models")
except Exception as e:
    logger.warning(f"Could not load saved models: {e}")
    logger.info("Will use base models without loading saved versions")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", default=None)


def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)


client = get_openai_client()


def openai_chat_complete(messages=(), model="gpt-4o-mini", raw=False):
    logger.info("Chat complete for {}".format(messages))
    response = client.chat.completions.create(model=model, messages=messages)
    if raw:
        return response.choices[0].message
    output = response.choices[0].message
    logger.info("Chat complete output: ".format(output))
    return output.content


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    logger.info("Receive embedding for text: {}".format(text))
    return (
        client.embeddings.create(input=text, model=model, encoding_format="float")
        .data[0]
        .embedding
    )


# Cập nhật hàm get_embedding vào EMBEDDING_MODELS
EMBEDDING_MODELS["openai"] = get_embedding


def gen_doc_prompt(docs):
    """
    Document:
    Title: Lộ trình học AI ...
    Content: ....
    """
    doc_prompt = ""
    for doc in docs:
        doc_prompt += f"Title: {doc['title']} \n Content: {doc['content']} \n"

    return "Document: \n + {}".format(doc_prompt)


def generate_conversation_text(conversations):
    conversation_text = ""
    for conversation in conversations:
        logger.info("Generate conversation: {}".format(conversation))
        role = conversation.get("role", "user")
        content = conversation.get("content", "")
        conversation_text += f"{role}: {content}\n"
    return conversation_text


def detect_user_intent(history, message):
    # Convert history to list messages
    history_messages = generate_conversation_text(history)
    logger.info(f"History messages: {history_messages}")
    # Update documents to prompt
    user_prompt = f"""
    Given following conversation and follow up question, rephrase the follow up question to a standalone question in the question's language.

    Chat History:
    {history_messages}

    Original Question: {message}

    Answer:
    """
    openai_messages = [
        {"role": "system", "content": "You are an amazing virtual assistant"},
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"Rephrase input messages: {openai_messages}")
    # call openai
    return openai_chat_complete(openai_messages)


def detect_route(history, message):
    logger.info(f"Detect route on history messages: {history}")
    # Update documents to prompt
    user_prompt = f"""
    Given the following chat history and the user's latest message, determine whether the user's intent is to ask for a frequently asked question ("bank_faq") or to inquire about loans and savings ("loan_savings") or to play some games ("play_game"). \n
    Provide only the classification label as your response.

    Chat History:
    {history}

    Latest User Message:
    {message}

    Classification (choose either "bank_faq" or "loan_savings" or "play_game"):
    """
    openai_messages = [
        {
            "role": "system",
            "content": "You are a highly intelligent assistant that helps classify customer queries",
        },
        {"role": "user", "content": user_prompt},
    ]
    logger.info(f"Route output: {openai_messages}")
    # call openai
    return openai_chat_complete(openai_messages)


def get_financial_tools():
    tools = []
    logger.info(f"Financial tools: {tools}")
    return tools


available_tools = {
    "calculate_fixed_monthly_payment": calculate_fixed_monthly_payment,
    "calculate_future_value": calculate_future_value,
}


def get_financial_agent_answer(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()

    # Execute the chat completion request
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    # Attempt to extract response details
    if not resp.choices:
        logger.error("No choices available in the response.")
        return {
            "role": "assistant",
            "content": "An error occurred, please try again later.",
        }

    choice = resp.choices[0]
    return choice


def convert_tool_calls_to_json(tool_calls):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "arguments": json.dumps(call.function.arguments),
                    "name": call.function.name,
                },
            }
            for call in tool_calls
        ],
    }


def get_financial_agent_handle(messages, model="gpt-4o", tools=None):
    if not tools:
        tools = get_financial_tools()
    choice = get_financial_agent_answer(messages, model, tools)

    resp_content = choice.message.content
    resp_tool_calls = choice.message.tool_calls
    # Prepare the assistant's message
    if resp_content:
        return resp_content

    elif resp_tool_calls:
        logger.info(f"Process the tools call: {resp_tool_calls}")
        # List to hold tool response messages
        tool_messages = []
        # Iterate through each tool call and execute the corresponding function
        for tool_call in resp_tool_calls:
            # Display the tool call details
            logger.info(
                f"Tool call: {tool_call.function.name}({tool_call.function.arguments})"
            )
            # Retrieve the tool function from available tools
            tool = available_tools[tool_call.function.name]
            # Parse the arguments for the tool function
            tool_args = json.loads(tool_call.function.arguments)
            # Execute the tool function and get the result
            result = tool(**tool_args)
            tool_args["result"] = result
            # Append the tool's response to the tool_messages list
            tool_messages.append(
                {
                    "role": "tool",  # Indicate this message is from a tool
                    "content": json.dumps(tool_args),  # The result of the tool function
                    "tool_call_id": tool_call.id,  # The ID of the tool call
                }
            )
        # Update the new message to get response from LLM
        # Append the tool messages to the existing messages
        # Check here: https://platform.openai.com/docs/guides/function-calling
        next_messages = (
            messages + [convert_tool_calls_to_json(resp_tool_calls)] + tool_messages
        )
        return get_financial_agent_handle(next_messages, model, tools)
    else:
        raise InvalidResponse(f"The response is invalid: {choice}")
