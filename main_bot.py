# main_bot.py

# ==============================================================================
#                      PROTOCOLO NAUTILUS - TELEGRAM BOT v3.0
#                   (Sistema Profesional con Búsqueda Ética)
# ==============================================================================

import os
import logging
import random
import base64
import asyncio
import re
from io import BytesIO
from typing import Dict, Optional, Any, List, Tuple

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
MISTRAL_CLOUD_MODEL_ID = "mistralai/mistral-small-3.2-24b-instruct:free"

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

# --- 3. SISTEMA DE COORDENADAS PROFESIONAL ---
def generate_professional_coordinates() -> str:
    """
    Genera coordenadas siguiendo el estándar de programas formales de percepción remota.
    Formato: XXXX-XXXX o similar, sin prefijos fijos para evitar sesgos.
    """
    formats = [
        # Formato clásico: 4-4 dígitos
        lambda: f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
        # Formato extendido: 4-5 dígitos  
        lambda: f"{random.randint(1000, 9999)}-{random.randint(10000, 99999)}",
        # Formato militar: 5-4 dígitos
        lambda: f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}",
        # Formato secuencial: 6 dígitos
        lambda: f"{random.randint(100000, 999999)}",
        # Formato alfanumérico: XXXX-YZ
        lambda: f"{random.randint(1000, 9999)}-{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(1, 9)}",
        # Formato de laboratorio: XXXXX
        lambda: f"{random.randint(10000, 99999)}",
    ]
    
    selected_format = random.choice(formats)
    return selected_format()

# --- FUNCIONES AUXILIARES PARA MANEJO DE TEXTO ---
def clean_markdown_for_telegram(text: str) -> str:
    """
    Limpia el markdown para evitar errores de parsing en Telegram.
    """
    # Escapar caracteres problemáticos
    text = text.replace('_', '\\_')
    text = text.replace('*', '\\*')
    text = text.replace('[', '\\[')
    text = text.replace(']', '\\]')
    text = text.replace('(', '\\(')
    text = text.replace(')', '\\)')
    text = text.replace('~', '\\~')
    text = text.replace('`', '\\`')
    text = text.replace('>', '\\>')
    text = text.replace('#', '\\#')
    text = text.replace('+', '\\+')
    text = text.replace('-', '\\-')
    text = text.replace('=', '\\=')
    text = text.replace('|', '\\|')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('.', '\\.')
    text = text.replace('!', '\\!')
    
    return text

def strip_markdown(text: str) -> str:
    """
    Elimina todo el markdown dejando solo texto plano.
    """
    import re
    # Eliminar headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Eliminar bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Eliminar código
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Eliminar links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Limpiar caracteres especiales
    text = re.sub(r'[_~>#+\-=|{}\.!]', '', text)
    
    return text

# Pool corregido de objetivos con URLs verificadas
VERIFIED_SAFE_TARGETS = [
    {
        "name": "Puente de Piedra Ancestral",
        "url": "https://images.unsplash.com/photo-1516026672322-bc52d61a55d5?w=800&h=600&fit=crop",
        "description": "Construcción histórica integrando naturaleza y arquitectura"
    },
    {
        "name": "Templo Dorado de Kyoto en Otoño",
        "url": "https://images.unsplash.com/photo-1545569341-9eb8b30979d9?w=800&h=600&fit=crop",
        "description": "Arquitectura tradicional japonesa rodeada de naturaleza"
    },
    {
        "name": "Catedral Gótica con Luz Natural",
        "url": "https://images.unsplash.com/photo-1520637836862-4d197d17c93a?w=800&h=600&fit=crop",
        "description": "Majestuosa arquitectura religiosa histórica"
    },
    {
        "name": "Faro Histórico al Amanecer",
        "url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&h=600&fit=crop",
        "description": "Estructura marítima clásica con significado de guía"
    },
    {
        "name": "Biblioteca Antigua con Cúpula",
        "url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop",
        "description": "Arquitectura dedicada al conocimiento y la sabiduría"
    },
    {
        "name": "Jardín de Meditación Zen",
        "url": "https://images.unsplash.com/photo-1547036967-23d11aacaee0?w=800&h=600&fit=crop",
        "description": "Espacio diseñado para la contemplación y paz interior"
    },
    {
        "name": "Observatorio Astronómico Histórico",
        "url": "https://images.unsplash.com/photo-1464822759844-d150baec0494?w=800&h=600&fit=crop",
        "description": "Estructura dedicada al estudio del cosmos"
    },
    {
        "name": "Cascada en Bosque Primigenio",
        "url": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop",
        "description": "Fuerza natural en ecosistema preservado"
    },
    {
        "name": "Molino de Viento Tradicional",
        "url": "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800&h=600&fit=crop",
        "description": "Tecnología histórica en armonía con elementos naturales"
    },
    {
        "name": "Anfiteatro Romano entre Colinas",
        "url": "https://images.unsplash.com/photo-1539650116574-75c0c6d05be5?w=800&h=600&fit=crop",
        "description": "Arquitectura clásica diseñada para reunión y cultura"
    }
]
SAFE_SEARCH_CATEGORIES = {
    "arquitectura_historica": [
        "ancient stone temple peaceful", "medieval cathedral architecture", 
        "historic monastery mountains", "traditional wooden bridge nature",
        "classic lighthouse coastline", "ancient amphitheater ruins",
        "historic windmill countryside", "traditional pagoda garden"
    ],
    "paisajes_naturales": [
        "serene mountain lake reflection", "peaceful waterfall forest",
        "ancient oak tree meadow", "pristine alpine valley",
        "rolling green hills sunrise", "tranquil river bend",
        "misty morning forest path", "gentle coastal cliffs"
    ],
    "arte_cultura": [
        "classical marble sculpture garden", "historic fountain plaza",
        "ancient stone circle monument", "traditional art museum",
        "peaceful cultural garden", "historic observatory dome",
        "classical music hall architecture", "ancient library ruins"
    ],
    "monumentos_positivos": [
        "peace memorial garden", "historic lighthouse beacon",
        "ancient astronomical observatory", "traditional cultural center",
        "historic university campus", "peaceful meditation garden",
        "classical architectural marvel", "ancient healing temple"
    ],
    "naturaleza_simbolica": [
        "centuries old tree wisdom", "sacred mountain peak",
        "pristine crystal cave", "ancient hot springs",
        "peaceful butterfly garden", "serene bamboo forest",
        "majestic aurora landscape", "tranquil zen garden"
    ]
}

async def search_safe_target_with_duckduckgo(search_term: str) -> Optional[Dict[str, str]]:
    """
    Busca una imagen usando DuckDuckGo con términos éticos y seguros.
    Por ahora usa el pool verificado de objetivos seguros.
    """
    try:
        # Usar el pool verificado de objetivos seguros
        selected_target = random.choice(VERIFIED_SAFE_TARGETS)
        logger.info(f"Objetivo seleccionado del pool verificado: {selected_target['name']}")
        return selected_target
        
    except Exception as e:
        logger.error(f"Error en búsqueda de objetivo: {e}")
        return None

async def select_ethical_target() -> Dict[str, str]:
    """
    Selecciona un objetivo ético usando criterios de seguridad psicológica.
    """
    try:
        # Seleccionar directamente del pool verificado
        target = random.choice(VERIFIED_SAFE_TARGETS)
        logger.info(f"Objetivo ético seleccionado: {target['name']}")
        return target
            
    except Exception as e:
        logger.error(f"Error seleccionando objetivo ético: {e}")
        # Fallback seguro garantizado
        return VERIFIED_SAFE_TARGETS[0]  # Siempre tenemos al menos uno

# Modelos Pydantic
class DrawingSubmission(BaseModel):
    imageData: str
    userId: int
    targetCoordinates: Optional[str] = None

class APIResponse(BaseModel):
    status: str
    message: Optional[str] = None

# --- 5. SERVIDOR API (FastAPI) ---
app_fastapi = FastAPI(title="Protocolo Nautilus API", version="3.0.0")
app_fastapi.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app_fastapi.post("/submit_drawing", response_model=APIResponse)
async def submit_drawing(submission: DrawingSubmission):
    user_id = submission.userId
    logger.info(f"Recibiendo dibujo de usuario {user_id} con coordenadas: {submission.targetCoordinates}")
    
    if not telegram_app: 
        raise HTTPException(status_code=503, detail="Servicio Telegram no inicializado")
    if user_id not in user_sessions: 
        raise HTTPException(status_code=404, detail="No hay sesión activa")
    
    chat_id = user_sessions[user_id].get("chat_id")
    if not chat_id: 
        raise HTTPException(status_code=404, detail="No se encontró chat_id")
    
    try:
        header, encoded = submission.imageData.split(",", 1)
        image_data = base64.b64decode(encoded)
        if not image_data: 
            raise ValueError("Imagen vacía")
        
        # Guardamos la imagen y comenzamos el análisis inmediatamente
        user_sessions[user_id]["session_data"]["fase3_boceto_bytes"] = image_data
        
        # Análisis inmediato del boceto para optimización de tiempo
        logger.info(f"Iniciando análisis inmediato del boceto para usuario {user_id}")
        sketch_desc = await describe_sketch_with_mistral(image_data)
        user_sessions[user_id]["session_data"]["sketch_description"] = sketch_desc
        logger.info(f"Análisis del boceto completado para usuario {user_id}")
        
        # Obtener las coordenadas del objetivo de la sesión
        target_ref = user_sessions[user_id]["session_data"].get("target_ref", "????-????")
        
        await telegram_app.bot.send_photo(
            chat_id=chat_id, 
            photo=image_data, 
            caption="🎨 <b>Boceto recibido y procesado</b>", 
            parse_mode='HTML'
        )
        
        await telegram_app.bot.send_message(
            chat_id=chat_id, 
            text=f"¡Excelente!\n\n<b>FASE 4: DATOS CONCEPTUALES</b>\n<b>Coordenadas del objetivo:</b> <code>{target_ref}</code>\n\nAhora describe las <b>cualidades intangibles</b> y <b>conceptos abstractos</b> que percibes. Estas impresiones pueden incluir:\n\n• Sensaciones emocionales\n• Propósito o función\n• Atmósfera o energía\n• Significado simbólico\n• Contexto temporal o cultural\n\n<i>Recuerda: las impresiones son sutiles y pueden parecer como \"recuerdos descoloridos\". Confía en tus primeras intuiciones.</i>", 
            parse_mode='HTML'
        )
        
        return APIResponse(status="ok")
        
    except Exception as e:
        logger.error(f"Error en submit_drawing para {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. FUNCIONES DE IA ESPECIALIZADAS ---
def describe_objective_with_blip(image_bytes: bytes) -> str:
    if not blip_model: 
        return "Modelo de visión local no disponible."
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
    if not openrouter_client: 
        return "Modelo de visión en la nube no disponible."
    try:
        logger.info("Describiendo BOCETO con Mistral en la nube...")
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt_text = """Analiza este boceto de percepción remota de manera objetiva y detallada. 

Describe específicamente:
- Formas y líneas principales
- Composición y distribución espacial  
- Elementos estructurales identificables
- Patrones o texturas sugeridas
- Proporciones y relaciones entre elementos
- Cualquier detalle técnico o arquitectónico aparente

Sé conciso pero exhaustivo en la descripción visual."""
        
        response = await asyncio.to_thread(
            openrouter_client.chat.completions.create, 
            model=MISTRAL_CLOUD_MODEL_ID, 
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt_text}, 
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            temperature=0.2,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error al describir BOCETO con Mistral: {e}")
        return "Error al procesar el boceto."

async def get_professional_analysis_with_mistral(user_transcript: str, target_desc: str, sketch_desc: str, target_name: str, coordinates: str) -> str:
    if not openrouter_client: 
        return "Análisis de texto no disponible."
    
    logger.info("Generando análisis profesional con Mistral...")
    
    system_prompt = """Eres un Analista Senior de Percepción Remota con experiencia en protocolos científicos estándar. 
Tu evaluación debe ser profesional, constructiva y basada en criterios establecidos de correlación en percepción remota.
Usa terminología apropiada del campo y mantén un tono alentador pero riguroso."""
    
    user_prompt = f"""ANÁLISIS DE SESIÓN DE PERCEPCIÓN REMOTA

**COORDENADAS DEL OBJETIVO:** {coordinates}
**OBJETIVO REAL:** {target_name}

**DATOS DEL PERCEPTOR:**
---
{user_transcript}
---

**ANÁLISIS DETALLADO DEL BOCETO (IA Visual):**
---
{sketch_desc}
---

**DESCRIPCIÓN DEL OBJETIVO REAL (IA BLIP):**
---
{target_desc}
---

**INSTRUCCIONES PARA EL ANÁLISIS:**

Genera un informe profesional en Markdown siguiendo la estructura estándar de evaluación:

## 📋 Resumen Ejecutivo
- Evaluación general de la precisión de la sesión
- Puntuación preliminar y justificación metodológica

## 🎨 Análisis del Boceto
### Descripción Técnica
- Análisis detallado de lo que la IA detectó en el dibujo
- Elementos estructurales y compositivos identificados
- Características técnicas y proporciones

### Interpretación Perceptual
- Correlación entre elementos dibujados y objetivo real
- Análisis de la traducción subconsciente de datos

## 📊 Correlaciones Identificadas
### Correspondencias Directas
- Elementos específicos que coinciden exactamente
- Detalles estructurales alineados
- Características físicas acertadas

### Correspondencias Conceptuales
- Temas y conceptos abstractos que coinciden
- Impresiones atmosféricas correctas
- Datos intangibles acertados

### Correspondencias Sensoriales
- Datos táctiles, dimensionales y texturales correctos
- Información sobre densidad, masa y escala

## 🎯 Elementos Únicos del Perceptor
- Datos proporcionados que no aparecen en descripciones de referencia
- Interpretación de estos elementos únicos
- Posible significado o relevancia

## ⚠️ Discrepancias y Desviaciones
- Elementos que no corresponden al objetivo
- Posibles interpretaciones alternativas
- Análisis de ruido vs. datos válidos

## 📈 Evaluación Cuantitativa
**Escala de Precisión: 1.0 - 10.0**
- **Puntuación Final:** [X.X/10.0]
- **Justificación Metodológica:** Explicación detallada de la calificación
- **Criterios de Evaluación:** Factores considerados en la puntuación

## 💡 Retroalimentación Constructiva
- Fortalezas identificadas en la sesión
- Áreas de mejora para futuras sesiones
- Recomendaciones específicas para el desarrollo

**NOTA:** Mantén un enfoque científico y objetivo, reconociendo que la percepción remota implica datos sutiles y a menudo fragmentarios."""

    try:
        response = await asyncio.to_thread(
            openrouter_client.chat.completions.create, 
            model=MISTRAL_CLOUD_MODEL_ID, 
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}
            ], 
            temperature=0.3, 
            max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generando análisis profesional con Mistral: {e}")
        return "Error: El servicio de análisis profesional no está disponible."

# --- 7. HANDLERS DE TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    user_sessions[user.id] = {"chat_id": update.effective_chat.id, "session_data": {}}
    
    # Seleccionar objetivo ético y generar coordenadas profesionales
    selected_target = await select_ethical_target()
    target_ref = generate_professional_coordinates()
    
    # Guardar datos de la sesión
    user_sessions[user.id]["session_data"]["target"] = selected_target
    user_sessions[user.id]["session_data"]["target_ref"] = target_ref
    
    logger.info(f"Usuario {user.id} ({user.first_name}) inició sesión. Objetivo: {selected_target['name']}, Coordenadas: {target_ref}")
    
    await update.message.reply_html(
        f"Hola {user.mention_html()}.\n"
        f"Bienvenido al <b>Protocolo Nautilus v3.0</b> - <i>Percepción Remota Controlada</i>\n\n"
        f"<b>Coordenadas del objetivo:</b> <code>{target_ref}</code>\n\n"
        f"<b>FASE 1: IMPRESIONES GESTALT</b>\n"
        f"Describe tus <b>primeras impresiones</b> sobre el objetivo. Estas pueden incluir:\n\n"
        f"• Sensaciones táctiles (rugoso, suave, frío, cálido)\n"
        f"• Impresiones dimensionales (grande, pequeño, alto, ancho)\n"
        f"• Datos primitivos de forma o estructura\n\n"
        f"<i>Nota: Las impresiones son sutiles, como \"recuerdos descoloridos\". Confía en tus primeras intuiciones sin analizarlas.</i>"
    )
    return FASE_1_GESTALT

async def fase_1_gestalt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    target_ref = user_sessions[user_id]["session_data"]["target_ref"]
    
    user_sessions[user_id]["session_data"]["fase1"] = update.message.text
    
    await update.message.reply_html(
        f"✅ <b>Impresiones Gestalt registradas</b>\n\n"
        f"<b>FASE 2: DATOS SENSORIALES</b>\n"
        f"<b>Coordenadas del objetivo:</b> <code>{target_ref}</code>\n\n"
        f"Describe datos sensoriales más específicos:\n\n"
        f"• <b>Colores:</b> Tonalidades o matices percibidos\n"
        f"• <b>Texturas:</b> Características de superficie\n"
        f"• <b>Sonidos:</b> Impresiones auditivas\n"
        f"• <b>Densidad/Masa:</b> Sensación de peso o solidez\n"
        f"• <b>Temperatura:</b> Sensaciones térmicas\n\n"
        f"<i>Los datos pueden acumularse gradualmente. Reporta lo que percibas sin forzar.</i>"
    )
    return FASE_2_SENSORIAL

async def fase_2_sensorial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    target_ref = user_sessions[user_id]["session_data"]["target_ref"]
    
    user_sessions[user_id]["session_data"]["fase2"] = update.message.text
    
    # Crear URL del WebApp con las coordenadas
    webapp_url = f"{CANVAS_URL}?target={target_ref}"
    
    keyboard = [[InlineKeyboardButton("🎨 Abrir Canvas de Percepción", web_app=WebAppInfo(url=webapp_url))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_html(
        f"✅ <b>Datos sensoriales registrados</b>\n\n"
        f"<b>FASE 3: BOCETO PERCEPTUAL</b>\n"
        f"<b>Coordenadas del objetivo:</b> <code>{target_ref}</code>\n\n"
        f"Ahora traduce tus impresiones en un <b>boceto</b>. Este dibujo debe:\n\n"
        f"• Representar las formas e impresiones percibidas\n"
        f"• Mostrar relaciones espaciales básicas\n"
        f"• Incluir elementos estructurales intuidos\n\n"
        f"<i>No busques perfección artística. El boceto es una herramienta de traducción de datos perceptuales.</i>", 
        reply_markup=reply_markup
    )
    return FASE_3_BOCETO

async def fase_4_conceptual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    target_ref = user_sessions[user_id]["session_data"]["target_ref"]
    
    user_sessions[user_id]["session_data"]["fase4"] = update.message.text
    
    await update.message.reply_html(
        f"✅ <b>Datos conceptuales registrados</b>\n\n"
        f"🎯 <b>Sesión de Percepción Remota Completada</b>\n"
        f"<b>Coordenadas finales:</b> <code>{target_ref}</code>\n\n"
        f"<b>Datos recopilados:</b>\n"
        f"• Impresiones Gestalt ✅\n"
        f"• Datos Sensoriales ✅\n"
        f"• Boceto Perceptual ✅\n"
        f"• Datos Conceptuales ✅\n\n"
        f"¿Listo para ver el <b>análisis profesional</b> y el <b>objetivo real</b>?\n\n"
        f"Envía /finalizar para obtener tu evaluación completa."
    )
    return FINALIZAR

async def finalizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        await update.message.reply_text("❌ No se encontró una sesión activa. Por favor, /start.")
        return ConversationHandler.END
    
    await update.message.reply_text("⏳ <b>Procesando análisis profesional...</b>\n<i>Contactando sistemas de IA Local y en la Nube...</i>", parse_mode='HTML')
    
    session_data = user_sessions[user_id].get("session_data", {})
    target_info = session_data.get("target")
    target_ref = session_data.get("target_ref", "????-????")
    
    if not target_info:
        await update.message.reply_text("❌ Error al recuperar objetivo. Por favor, /start.")
        return ConversationHandler.END

    logger.info(f"Usuario {user_id} finalizó. Objetivo: {target_info['name']}, Coordenadas: {target_ref}")
    
    # Crear transcripción completa del perceptor
    user_transcript = (
        f"FASE 1 - Impresiones Gestalt:\n{session_data.get('fase1', 'N/A')}\n\n"
        f"FASE 2 - Datos Sensoriales:\n{session_data.get('fase2', 'N/A')}\n\n"
        f"FASE 4 - Datos Conceptuales:\n{session_data.get('fase4', 'N/A')}"
    )
    
    # Describir objetivo con BLIP
    try:
        response = requests.get(target_info["url"], timeout=15)
        response.raise_for_status()
        target_desc = describe_objective_with_blip(response.content)
    except Exception as e:
        logger.error(f"No se pudo descargar/describir el objetivo: {e}")
        target_desc = "Error al procesar la imagen objetivo."

    # Obtener descripción del boceto (ya analizada o analizarla ahora)
    sketch_desc = session_data.get("sketch_description", "El perceptor no proporcionó un boceto.")
    if sketch_desc == "El perceptor no proporcionó un boceto.":
        user_drawing_bytes = session_data.get("fase3_boceto_bytes")
        if user_drawing_bytes:
            sketch_desc = await describe_sketch_with_mistral(user_drawing_bytes)

    # Generar análisis profesional completo
    session_analysis = await get_professional_analysis_with_mistral(
        user_transcript, target_desc, sketch_desc, target_info['name'], target_ref
    )
    
    # Enviar revelación del objetivo
    await context.bot.send_photo(
        chat_id=user_id, 
        photo=target_info["url"], 
        caption=(
            f"🎯 <b>REVELACIÓN DEL OBJETIVO</b>\n\n"
            f"<b>Coordenadas:</b> <code>{target_ref}</code>\n"
            f"<b>Objetivo Real:</b> {target_info['name']}\n\n"
            f"<i>{target_info.get('description', 'Objetivo de percepción remota controlada')}</i>"
        ), 
        parse_mode='HTML'
    )
    
    # Enviar análisis profesional con manejo de errores mejorado
    if "Error:" in session_analysis:
        await context.bot.send_message(
            chat_id=user_id, 
            text=f"⚠️ <b>No se pudo generar el análisis profesional</b>\n\n{session_analysis}", 
            parse_mode='HTML'
        )
    else:
        try:
            # Limpiar el markdown para evitar errores de parsing
            cleaned_analysis = clean_markdown_for_telegram(session_analysis)
            
            # Dividir el análisis en partes si es muy largo
            if len(cleaned_analysis) > 4000:
                parts = [cleaned_analysis[i:i+4000] for i in range(0, len(cleaned_analysis), 4000)]
                for i, part in enumerate(parts):
                    header = f"📊 <b>ANÁLISIS PROFESIONAL - Parte {i+1}/{len(parts)}</b>\n\n" if i == 0 else ""
                    await context.bot.send_message(
                        chat_id=user_id, 
                        text=header + part, 
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(1)  # Pausa para evitar límites de tasa
            else:
                await context.bot.send_message(
                    chat_id=user_id, 
                    text=f"📊 <b>ANÁLISIS PROFESIONAL</b>\n\n{cleaned_analysis}", 
                    parse_mode='Markdown'
                )
        except Exception as e:
            logger.error(f"Error enviando análisis con Markdown: {e}")
            # Fallback: enviar como texto plano
            plain_analysis = strip_markdown(session_analysis)
            await context.bot.send_message(
                chat_id=user_id, 
                text=f"📊 <b>ANÁLISIS PROFESIONAL</b>\n\n{plain_analysis}", 
                parse_mode='HTML'
            )
    
    # Mensaje de cierre profesional
    await update.message.reply_html(
        f"🙏 <b>Sesión de Percepción Remota Completada</b>\n\n"
        f"Gracias por participar en este protocolo controlado. Tus datos han sido procesados "
        f"siguiendo estándares profesionales de evaluación.\n\n"
        f"<b>Recuerda:</b> La percepción remota es una habilidad que se desarrolla con práctica. "
        f"Cada sesión proporciona datos valiosos para tu crecimiento en esta disciplina.\n\n"
        f"Para una nueva sesión, envía /start"
    )
    
    del user_sessions[user_id]
    return ConversationHandler.END

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id in user_sessions: 
        del user_sessions[user_id]
    await update.message.reply_text("❌ Sesión cancelada. Envía /start para comenzar una nueva sesión de percepción remota.")
    return ConversationHandler.END

# --- 8. COMANDOS ADICIONALES ---
async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Proporciona información sobre el protocolo de percepción remota."""
    info_text = """
🧠 <b>PROTOCOLO NAUTILUS - Información</b>

<b>¿Qué es la Percepción Remota?</b>
Es la capacidad de obtener información sobre un objetivo distante o inaccesible usando medios extrasensoriales. No se trata de "ver" el objetivo, sino de percibir datos sutiles que se manifiestan como:

• <b>Datos sensoriales:</b> tacto, gusto, olfato
• <b>Datos dimensionales:</b> tamaño, masa, densidad
• <b>Datos estructurales:</b> formas, relaciones espaciales
• <b>Datos conceptuales:</b> propósito, emociones, significado

<b>El Proceso:</b>
Las impresiones llegan como "ráfagas suaves de información", similares a recuerdos descoloridos. Rara vez son imágenes claras o visualmente impactantes.

<b>Seguridad:</b>
Este protocolo usa únicamente objetivos seguros: lugares históricos, arquitectura, paisajes naturales y monumentos culturales. Se evitan temas controversiales o traumáticos.

<b>Metodología:</b>
Seguimos protocolos estándar de percepción remota controlada con coordenadas aleatorias y análisis objetivo mediante IA.
"""
    await update.message.reply_html(info_text)

async def estadisticas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra estadísticas básicas del sistema."""
    active_sessions = len(user_sessions)
    stats_text = f"""
📊 <b>ESTADÍSTICAS DEL SISTEMA</b>

• <b>Sesiones activas:</b> {active_sessions}
• <b>Versión del protocolo:</b> 3.0
• <b>IA Local:</b> {'🟢 Activa' if blip_model else '🔴 Inactiva'}
• <b>IA en la Nube:</b> {'🟢 Activa' if openrouter_client else '🔴 Inactiva'}

<b>Características:</b>
✅ Coordenadas profesionales aleatorias
✅ Selección ética de objetivos
✅ Análisis inmediato de bocetos
✅ Terminología científica apropiada
✅ Protocolos de seguridad psicológica
"""
    await update.message.reply_html(stats_text)

# --- 9. CONFIGURACIÓN FINAL DE LA APLICACIÓN ---
def setup_telegram_application() -> Application:
    global telegram_app
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Conversation handler principal
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
    
    # Handlers adicionales
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("estadisticas", estadisticas))
    app.add_handler(CommandHandler("stats", estadisticas))  # Alias
    
    telegram_app = app
    return app

async def run_services():
    """Ejecuta todos los servicios de manera concurrente."""
    initialize_blip_model()
    app = setup_telegram_application()
    config = uvicorn.Config(app_fastapi, host=FASTAPI_HOST, port=FASTAPI_PORT, log_level="info")
    server = uvicorn.Server(config)
    
    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        logger.info("🤖 Protocolo Nautilus v3.0 funcionando...")
        logger.info("📡 Sistema de percepción remota controlada activo")
        await server.serve()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

def main():
    """Función principal de ejecución."""
    logger.info("🚀 Iniciando Protocolo Nautilus v3.0 - Sistema Profesional")
    logger.info("🔬 Implementando protocolos de percepción remota controlada")
    logger.info("🛡️ Sistema de selección ética de objetivos activado")
    
    try:
        asyncio.run(run_services())
    except KeyboardInterrupt:
        logger.info("👋 Protocolo Nautilus cerrado por el usuario.")
    except Exception as e:
        logger.error(f"💥 Error fatal en main: {e}", exc_info=True)

if __name__ == "__main__":
    main()