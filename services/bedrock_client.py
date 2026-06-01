import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()
aws_region = os.getenv("AWS_REGION", "us-east-1")

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region
)

MODEL_ID = os.getenv("MODEL_ID")

def load_system_prompt():
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Eres un asistente experto en AWS Certified Cloud Practitioner."


# 🛠️ FUNCIÓN DE GUARDRAIL DINÁMICO (No hardcoded)
def is_aws_related(last_user_message):
    """
    Le pregunta al modelo en un micro-segundo si el texto es de AWS o no.
    Retorna True si es válido, False si debe ser bloqueado.
    """
    eval_system_prompt = (
        "Analiza el texto del usuario. Responde ÚNICAMENTE con la palabra 'TRUE' "
        "si la pregunta está relacionada con AWS, Cloud Computing, tecnología de nube o saludos iniciales. "
        "Responde ÚNICAMENTE con la palabra 'FALSE' si habla de cualquier otra cosa ajena "
        "(como comida, pizza, Scrum, fútbol, tareas del hogar, etc.). No agregues nada más."
    )
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 5,
            "system": eval_system_prompt,
            "messages": [{"role": "user", "content": last_user_message}],
            "temperature": 0.0 # Cero creatividad para asegurar precisión
        })
        
        response = bedrock.invoke_model(modelId=MODEL_ID, body=body)
        result = json.loads(response["body"].read())["content"][0]["text"].strip().upper()
        
        return "TRUE" in result
    except Exception:
        return True # Si el guardrail falla por timeout, permite pasar por seguridad


def invoke_model(messages):
    # 1. Obtener el último mensaje que escribió el alumno
    last_user_text = messages[-1]["content"] if messages else ""
    
    # 2. Validar con nuestro Guardrail inteligente si el tema es permitido
    if not is_aws_related(last_user_text):
        # Si el Guardrail detecta que habla de otra cosa (como pizza),
        # corta la ejecución inmediatamente y devuelve la frase del archivo de texto
        system_rules = load_system_prompt()
        # Buscamos la frase estándar dentro de tu prompt.txt dinámicamente para no quemarla en código
        if "FRASE DE RECHAZO ESTÁNDAR:" in system_rules:
            frase_rechazo = system_rules.split("FRASE DE RECHAZO ESTÁNDAR:")[1].split("EJEMPLOS")[0].strip()
            return frase_rechazo.replace('"', '')
        return "Lo siento, solo puedo responder dudas sobre la certificación AWS Cloud Practitioner."

    # 3. Si la pregunta es válida de AWS, continúa el flujo normal con todo el historial
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