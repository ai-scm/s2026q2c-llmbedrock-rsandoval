import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Aseguramos que tome la variable de entorno o use us-east-1 por defecto
aws_region = os.getenv("AWS_REGION", "us-east-1")

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region
)

MODEL_ID = os.getenv("MODEL_ID")

# Función para leer el archivo del system prompt de forma segura
def load_system_prompt():
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Eres un asistente experto en AWS Certified Cloud Practitioner."

def invoke_model(messages):
    system_prompt = load_system_prompt()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "system": system_prompt, 
        "messages": messages,
        "temperature": 0.1
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body)
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]
