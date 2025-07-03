# test_advanced_vision.py (v2.1)

import os
import base64
import logging
from dotenv import load_dotenv
from openai import OpenAI

# --- 1. CONFIGURACIÓN (sin cambios) ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY: raise ValueError("OPENROUTER_API_KEY no encontrada.")
try:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    logger.info("Cliente de OpenRouter configurado.")
except Exception as e:
    logger.error(f"Error al configurar cliente: {e}"); exit()

# --- 2. LÓGICA DE LA PRUEBA (con el cambio clave) ---

def encode_image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error al codificar la imagen '{image_path}': {e}"); return None

def describe_sketch_with_advanced_model(image_base64: str, model_id: str):
    logger.info(f"Enviando boceto para descripción al modelo: {model_id}...")
    prompt_text = "Describe este boceto de manera objetiva y literal. Enfócate en las formas y las líneas. Sé conciso."
    
    # --- CAMBIO CLAVE EN LA ESTRUCTURA DEL PROMPT ---
    # En lugar de una lista de contenidos, separamos el texto y la imagen
    # en mensajes consecutivos dentro de la misma llamada.
    # Este formato a veces es mejor interpretado por ciertos modelos multimodales.
    messages_payload = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages_payload
        )
        description = response.choices[0].message.content
        logger.info("Descripción recibida del modelo.")
        return description
    except Exception as e:
        logger.error(f"Error en la llamada a la API de OpenRouter: {e}")
        return f"Error: {e}"

# --- 3. EJECUCIÓN PRINCIPAL (sin cambios) ---
if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(current_directory, "test", "image.jpeg")
    ADVANCED_MODEL_ID = "mistralai/mistral-small-3.2-24b-instruct:free"

    logger.info("--- INICIANDO PRUEBA DE VISIÓN AVANZADA (v2.1) ---")
    b64_image = encode_image_to_base64(IMAGE_PATH)
    if b64_image:
        sketch_description = describe_sketch_with_advanced_model(b64_image, ADVANCED_MODEL_ID)
        print("\n" + "="*50)
        print(f"  ANÁLISIS DEL BOCETO CON '{ADVANCED_MODEL_ID}'")
        print("="*50)
        print(f"Ruta de la imagen: {IMAGE_PATH}")
        print("-" * 50)
        print("Descripción generada:")
        print(sketch_description)
        print("="*50)
    logger.info("--- PRUEBA FINALIZADA ---")