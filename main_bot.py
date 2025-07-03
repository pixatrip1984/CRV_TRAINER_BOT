# main_bot.py

# ==============================================================================
#                      PROTOCOLO NAUTILUS - TELEGRAM BOT v2.5
#                         (Versión Estable y Corregida)
# ==============================================================================

import os
import logging
import random
import base64
import asyncio
from io import BytesIO
from typing import Dict, Optional, Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from PIL import Image

# Dependencias de IA Local
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Dependencias de Telegram
from telegram import Update, WebAppInfo, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    ConversationHandler,
    filters,
)

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

if not TELEGRAM_TOKEN: raise ValueError("TELEGRAM_TOKEN no encontrado.")

CANVAS_URL = "https://pixatrip1984.github.io/nautilus-canvas/"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler('nautilus_bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

(FASE_1_GESTALT, FASE_2_SENSORIAL, FASE_3_BOCETO, FASE_4_CONCEPTUAL, FINALIZAR) = range(5)

user_sessions: Dict[int, Dict[str, Any]] = {}
telegram_app: Optional[Application] = None

# --- 2. CONFIGURACIÓN DE IA ---
openrouter_client: Optional[OpenAI] = None
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
else:
    logger.warning("OPENROUTER_API_KEY no encontrada.")

blip_processor: Optional[BlipProcessor] = None
blip_model: Optional[BlipForConditionalGeneration] = None
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
MISTRAL_CLOUD_MODEL_ID = "mistralai/mistral-small-3.2-24b-instruct:free" # Usamos el alias que sabemos que funciona

def initialize_blip_model():
    global blip_processor, blip_model
    try:
        logger.info(f"Cargando modelo local: {BLIP_MODEL_ID}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
        blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID).to(device)
        logger.info(f"✅ Modelo local cargado en dispositivo: {device}")
    except Exception as e:
        logger.error(f"💥 Error al cargar modelo local: {e}", exc_info=True)
        blip_model = None

# Modelos Pydantic
class DrawingSubmission(BaseModel):
    imageData: str
    userId: int

class APIResponse(BaseModel):
    status: str
    message: Optional[str] = None

# --- 3. POOL DE OBJETIVOS (CORREGIDO) ---
TARGET_POOL = [
    {"id": "T001", "name": "Las Pirámides de Giza con la Esfinge", "url": "https://images.unsplash.com/photo-1539650116574-75c0c6d05be5?w=800&h=600&fit=crop"},
    {"id": "T002", "name": "El Puente Golden Gate en la niebla", "url": "https://images.unsplash.com/photo-1545229763-8a6a4a51abf2?w=800&h=600&fit=crop"},
    {"id": "T003", "name": "La Torre Eiffel", "url": "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=800&h=600&fit=crop"},
    {"id": "T004", "name": "Paisaje Montañoso con Nubes", "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop"},
    {"id": "T005", "name": "El Transbordador Espacial despegando", "url": "https://images.unsplash.com/photo-1446776877081-d282a0f896e2?w=800&h=600&fit=crop"},
]

# --- 4. SERVIDOR API (FastAPI) ---
app_fastapi = FastAPI(title="Protocolo Nautilus API", version="2.5.0")
app_fastapi.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app_fastapi.post("/submit_drawing", response_model=APIResponse)
async def submit_drawing(submission: DrawingSubmission):
    user_id = submission.userId
    logger.info(f"Recibiendo dibujo de usuario {user_id}")
    if not telegram_app: raise HTTPException(status_code=503, detail="Servicio Telegram no inicializado")
    if user_id not in user_sessions: raise HTTPException(status_code=404, detail="No hay sesión activa")
    chat_id = user_sessions[user_id].get("chat_id")
    if not chat_id: raise HTTPException(status_code=404, detail="No se encontró chat_id")
    try:
        header, encoded = submission.imageData.split(",", 1)
        image_data = base64.b64decode(encoded)
        if not image_data: raise ValueError("Imagen vacía")
        user_sessions[user_id]["session_data"]["fase3_boceto_bytes"] = image_data
        await telegram_app.bot.send_photo(chat_id=chat_id, photo=image_data, caption="🎨 <b>Boceto recibido</b>", parse_mode='HTML')
        await telegram_app.bot.send_message(chat_id=chat_id, text="¡Excelente!\n\n<b>FASE 4: CONCEPTUAL</b>\nDescribe las cualidades intangibles y conceptos abstractos.", parse_mode='HTML')
        return APIResponse(status="ok")
    except Exception as e:
        logger.error(f"Error en submit_drawing para {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. FUNCIONES DE IA ESPECIALIZADAS ---
def describe_objective_with_blip(image_bytes: bytes) -> str:
    if not blip_model: return "Modelo de visión local no disponible."
    try:
        logger.info("Describiendo OBJETIVO con BLIP local...")
        raw_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = blip_processor(raw_image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs, max_new_tokens=75)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logger.error(f"Error al describir OBJETIVO con BLIP: {e}")
        return "Error al procesar la imagen objetivo."

async def describe_sketch_with_mistral(image_bytes: bytes) -> str:
    if not openrouter_client: return "Modelo de visión en la nube no disponible."
    try:
        logger.info("Describiendo BOCETO con Mistral en la nube...")
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt_text = "Describe este boceto de manera objetiva y literal. Enfócate en las formas, las líneas y la composición. Sé conciso."
        response = await asyncio.to_thread(openrouter_client.chat.completions.create, model=MISTRAL_CLOUD_MODEL_ID, messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}]}])
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error al describir BOCETO con Mistral: {e}")
        return "Error al procesar el boceto."

async def get_text_analysis_with_mistral(user_transcript: str, target_desc: str, sketch_desc: str) -> str:
    if not openrouter_client: return "Análisis de texto no disponible."
    logger.info("Generando análisis de texto con Mistral...")
    system_prompt = "Eres un 'Analista de Protocolo Nautilus', un experto imparcial en evaluar sesiones de Visión Remota. Tu tono es alentador pero riguroso."
    user_prompt = f"""Analiza la siguiente sesión. Compara 3 piezas de texto: la transcripción, la descripción del boceto y la descripción del objetivo.
**1. TRANSCRIPCIÓN DEL VIDENTE:**\n---\n{user_transcript}\n---
**2. DESCRIPCIÓN DEL BOCETO (IA):**\n---\n{sketch_desc}\n---
**3. DESCRIPCIÓN DEL OBJETIVO (IA):**\n---\n{target_desc}\n---
**INSTRUCCIONES:** Crea un informe en Markdown con: Resumen General, Correlaciones Directas (Texto y Boceto), Correlaciones Conceptuales, Discrepancias Notables, y Puntuación de Precisión (1.0-10.0) con justificación."""
    try:
        response = await asyncio.to_thread(openrouter_client.chat.completions.create, model=MISTRAL_CLOUD_MODEL_ID, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.4, max_tokens=2048)
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generando análisis de texto con Mistral: {e}")
        return "Error: El servicio de análisis de texto no está disponible."

# --- 6. HANDLERS DE TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    user_sessions[user.id] = {"chat_id": update.effective_chat.id, "session_data": {}}
    selected_target = random.choice(TARGET_POOL)
    user_sessions[user.id]["session_data"]["target"] = selected_target
    target_ref = f"PN-{random.randint(1000, 9999)}-{random.choice('WXYZ')}"
    logger.info(f"Usuario {user.id} ({user.first_name}) inició sesión. Objetivo: {selected_target['name']}")
    await update.message.reply_html(f"Hola {user.mention_html()}.\nBienvenido al <b>Protocolo Nautilus v2.5</b>.\n\nTu objetivo es: <code>{target_ref}</code>\n\n<b>FASE 1: GESTALT</b>\nDescribe tus impresiones primarias.")
    return FASE_1_GESTALT

async def fase_1_gestalt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_sessions[update.effective_user.id]["session_data"]["fase1"] = update.message.text
    await update.message.reply_html("✅ Fase 1 registrada.\n\n<b>FASE 2: DATOS SENSORIALES</b>\nDescribe colores, texturas, sonidos, etc.")
    return FASE_2_SENSORIAL

async def fase_2_sensorial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_sessions[update.effective_user.id]["session_data"]["fase2"] = update.message.text
    keyboard = [[InlineKeyboardButton("🎨 Abrir Lienzo Nautilus", web_app=WebAppInfo(url=CANVAS_URL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_html("✅ Datos sensoriales registrados.\n\n<b>FASE 3: BOCETO</b>\nPresiona el botón para dibujar.", reply_markup=reply_markup)
    return FASE_3_BOCETO

async def fase_4_conceptual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_sessions[update.effective_user.id]["session_data"]["fase4"] = update.message.text
    await update.message.reply_html("✅ Fase 4 registrada.\n\n🎯 <b>Sesión completa</b>\n¿Listo? Envía /finalizar.")
    return FINALIZAR

async def finalizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        await update.message.reply_text("❌ No se encontró una sesión activa. Por favor, /start.")
        return ConversationHandler.END
    await update.message.reply_text("⏳ Finalizando... Contactando IAs Local y en la Nube...")
    
    session_data = user_sessions[user_id].get("session_data", {})
    target_info = session_data.get("target")
    if not target_info:
        await update.message.reply_text("❌ Error al recuperar objetivo. Por favor, /start.")
        return ConversationHandler.END

    logger.info(f"Usuario {user_id} finalizó. Objetivo: {target_info['name']}.")
    user_transcript = (f"Fase 1: {session_data.get('fase1', 'N/A')}\nFase 2: {session_data.get('fase2', 'N/A')}\nFase 4: {session_data.get('fase4', 'N/A')}")
    
    try:
        response = requests.get(target_info["url"], timeout=10)
        response.raise_for_status()
        target_desc = describe_objective_with_blip(response.content)
    except Exception as e:
        logger.error(f"No se pudo descargar/describir el objetivo: {e}")
        target_desc = "Error al procesar la imagen objetivo."

    user_drawing_bytes = session_data.get("fase3_boceto_bytes")
    sketch_desc = "El usuario no proporcionó un boceto."
    if user_drawing_bytes:
        sketch_desc = await describe_sketch_with_mistral(user_drawing_bytes)

    session_analysis = await get_text_analysis_with_mistral(user_transcript, target_desc, sketch_desc)
    
    await context.bot.send_photo(chat_id=user_id, photo=target_info["url"], caption=f"🎯 <b>El objetivo era:</b>\n{target_info['name']}", parse_mode='HTML')
    if "Error:" in session_analysis:
        await context.bot.send_message(chat_id=user_id, text=f"⚠️ No se pudo generar el análisis automático.\n{session_analysis}", parse_mode='HTML')
    else:
        await context.bot.send_message(chat_id=user_id, text=session_analysis, parse_mode='Markdown')
    
    await update.message.reply_html("🙏 <b>¡Gracias por participar!</b>\nPara una nueva sesión, envía /start.")
    del user_sessions[user_id]
    return ConversationHandler.END

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id in user_sessions: del user_sessions[user_id]
    await update.message.reply_text("❌ Sesión cancelada. Envía /start para empezar de nuevo.")
    return ConversationHandler.END

# --- 7. EJECUCIÓN CONCURRENTE ---
def setup_telegram_application() -> Application:
    global telegram_app
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            FASE_1_GESTALT: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_1_gestalt)],
            FASE_2_SENSORIAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_2_sensorial)],
            FASE_3_BOCETO: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_4_conceptual)],
            FASE_4_CONCEPTUAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_4_conceptual)],
            FINALIZAR: [CommandHandler("finalizar", finalizar)],
        },
        fallbacks=[CommandHandler("cancelar", cancelar)],
        allow_reentry=True
    )
    app.add_handler(conv_handler)
    telegram_app = app
    return app

async def run_services():
    initialize_blip_model()
    app = setup_telegram_application()
    config = uvicorn.Config(app_fastapi, host=FASTAPI_HOST, port=FASTAPI_PORT, log_level="info")
    server = uvicorn.Server(config)
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        logger.info("🤖 Bot de Telegram funcionando...")
        await server.serve()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

def main():
    logger.info("🚀 Iniciando Protocolo Nautilus v2.5 (Estable)...")
    try:
        asyncio.run(run_services())
    except KeyboardInterrupt:
        logger.info("👋 Protocolo Nautilus cerrado por el usuario.")
    except Exception as e:
        logger.error(f"💥 Error fatal en main: {e}", exc_info=True)

if __name__ == "__main__":
    main()