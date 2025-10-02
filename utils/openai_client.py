import os
import requests
import base64
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import openai

# Option A: Azure Managed Identity (recommended on servers)
managed_identity_client_id = "YOUR_MANAGED_IDENTITY_CLIENT_ID"  # TODO
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(managed_identity_client_id=managed_identity_client_id), "https://cognitiveservices.azure.com/.default")
AZURE_ENDPOINT = "https://<your-openai-resource>.openai.azure.com/"  # TODO
API_VERSION = "YOUR_API_VERSION"  # TODO if needed
client = AzureOpenAI(
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    azure_ad_token_provider=token_provider
)
# Azure API configurations
azure_config_list = [
    {
        "model": "gpt-4o",
        "api_type": "azure",
        "max_retries": 10,
        "azure_ad_token_provider": token_provider,
        "base_url": AZURE_ENDPOINT,
        "api_version": API_VERSION,
    },
]

# Option B: OpenAI API Key (good for local/personal use)
# Uncomment the following two lines for replacement
# OPENAI_API_KEY = "YOUR_AZURE_OPENAI_API_KEY"  # TODO
# client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_openai_response_text_only(prompt, temp=0.5):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API request: {e}")
        return None



def get_openai_response_base64(prompt, image_path):
    try:
        with open(image_path, "rb") as image_file:
            img_b64_str = base64.b64encode(image_file.read()).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64_str}"
                            },
                        },],}],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API request: {e}")
        return None

def get_openai_response_base64_with_multiple_images(prompt, image_paths):
    try:
        images_b64 = []
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                img_b64_str = base64.b64encode(image_file.read()).decode("utf-8")
                images_b64.append(img_b64_str)
        message_content = [{"type": "text", "text": prompt}]
        for img_b64_str in images_b64:
            message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64_str}"
                    },}
            )
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            messages=[{
                    "role": "user",
                    "content": message_content,
                }],
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI API request: {e}")
        return None