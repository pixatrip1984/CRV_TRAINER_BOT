# main_bot.py

# ==============================================================================
#                      PROTOCOLO NAUTILUS - TELEGRAM BOT v2.0
#                      (Arquitectura Híbrida con FastAPI)
# ==============================================================================

import os
import logging
import random
import base64
import asyncio
import threading
from datetime import datetime
from io import BytesIO
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn

from telegram import Update, WebAppInfo, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    ConversationHandler,
    filters,
)

# --- 1. CONFIGURACIÓN INICIAL Y VARIABLES GLOBALES ---

# Carga las variables de entorno desde el archivo .env
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

# Validar que el token existe
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN no encontrado. Verifica tu archivo .env")

# URL de tu lienzo en GitHub Pages
CANVAS_URL = "https://pixatrip1984.github.io/nautilus-canvas/"

# Configuración de logging mejorada
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nautilus_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Definición de estados para el ConversationHandler
(
    FASE_1_GESTALT,
    FASE_2_SENSORIAL,
    FASE_3_BOCETO,
    FASE_4_CONCEPTUAL,
    FINALIZAR,
) = range(5)

# Diccionario para mapear user_id con chat_id (memoria temporal)
user_chat_mapping: Dict[int, int] = {}

# Instancia global de la aplicación de Telegram para uso en FastAPI
telegram_app: Optional[Application] = None

# --- 2. MODELOS PYDANTIC PARA FASTAPI ---

class CanvasSize(BaseModel):
    width: int
    height: int

class DrawingSubmission(BaseModel):
    imageData: str
    userId: int
    timestamp: str
    canvasSize: Optional[CanvasSize] = None
    
    @field_validator('imageData')
    @classmethod
    def validate_image_data(cls, v):
        if not v.startswith('data:image/'):
            raise ValueError('imageData debe ser un Data URL válido')
        return v
    
    @field_validator('userId')
    @classmethod
    def validate_user_id(cls, v):
        if v <= 0:
            raise ValueError('userId debe ser un entero positivo')
        return v

class APIResponse(BaseModel):
    status: str
    message: str = ""
    timestamp: str = datetime.now().isoformat()

# --- 3. POOL DE OBJETIVOS PREDEFINIDO ---
TARGET_POOL = [
    {
        "id": "T001", "name": "Las Pirámides de Giza con la Esfinge",
        "url": "https://images.unsplash.com/photo-1539650116574-75c0c6d05be5?w=800&h=600&fit=crop",
    },
    {
        "id": "T002", "name": "El Puente Golden Gate en la niebla", 
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",
    },
    {
        "id": "T003", "name": "La Torre Eiffel de noche",
        "url": "https://images.unsplash.com/photo-1511739001486-6bfe10ce785f?w=800&h=600&fit=crop",
    },
    {
        "id": "T004", "name": "La Cascada Seljalandsfoss en Islandia",
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",
    },
    {
        "id": "T005", "name": "El Transbordador Espacial despegando",
        "url": "https://images.unsplash.com/photo-1446776877081-d282a0f896e2?w=800&h=600&fit=crop",
    },
    {
        "id": "T006", "name": "Un Faro en Acantilado",
        "url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&h=600&fit=crop",
    },
    {
        "id": "T007", "name": "Montañas con Lago Alpino",
        "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",
    },
]

# --- 4. CONFIGURACIÓN DE FASTAPI ---

app_fastapi = FastAPI(
    title="Protocolo Nautilus API",
    description="API para recibir dibujos del lienzo de Visión Remota",
    version="2.0.0"
)

# Configuración de CORS para permitir peticiones desde el frontend
app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios concretos
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app_fastapi.get("/")
async def root():
    """Endpoint de estado de la API."""
    return {"message": "Protocolo Nautilus API funcionando", "version": "2.0.0"}

@app_fastapi.get("/health")
async def health_check():
    """Endpoint de verificación de salud del servicio."""
    return {
        "status": "healthy",
        "telegram_bot": "running" if telegram_app else "not_initialized",
        "timestamp": datetime.now().isoformat()
    }

@app_fastapi.post("/submit_drawing", response_model=APIResponse)
async def submit_drawing(submission: DrawingSubmission):
    """
    Endpoint principal para recibir dibujos desde la Mini App.
    Procesa la imagen y la envía al chat de Telegram correspondiente.
    """
    try:
        logger.info(f"Recibiendo dibujo de usuario {submission.userId}")
        
        # Verificar que tenemos la instancia del bot de Telegram
        if not telegram_app or not telegram_app.bot:
            logger.error("Instancia del bot de Telegram no disponible")
            raise HTTPException(status_code=500, detail="Bot de Telegram no disponible")
        
        # Buscar el chat_id correspondiente al user_id
        chat_id = user_chat_mapping.get(submission.userId)
        if not chat_id:
            logger.error(f"No se encontró chat_id para user_id {submission.userId}")
            raise HTTPException(status_code=404, detail="Usuario no encontrado en sesión activa")
        
        # Procesar la imagen
        try:
            # Extraer los datos base64 de la imagen
            header, encoded = submission.imageData.split(",", 1)
            image_data = base64.b64decode(encoded)
            image_stream = BytesIO(image_data)
            
            # Validar que la imagen se decodificó correctamente
            if len(image_data) == 0:
                raise ValueError("Imagen vacía después de decodificar")
                
        except Exception as e:
            logger.error(f"Error procesando imagen de usuario {submission.userId}: {e}")
            raise HTTPException(status_code=400, detail="Error procesando la imagen")
        
        # Enviar la imagen al chat de Telegram
        try:
            await telegram_app.bot.send_photo(
                chat_id=chat_id,
                photo=image_stream,
                caption="🎨 <b>Boceto de Fase 3 recibido</b>\n\nProcesando...",
                parse_mode='HTML'
            )
            logger.info(f"Imagen enviada exitosamente al chat {chat_id}")
            
            # Enviar mensaje para la Fase 4
            await telegram_app.bot.send_message(
                chat_id=chat_id,
                text="¡Excelente! Tu boceto ha sido registrado.\n\n"
                     "<b>FASE 4: CONCEPTUAL</b>\n"
                     "Ahora describe las cualidades intangibles, "
                     "impresiones abstractas y conceptos que percibes.",
                parse_mode='HTML'
            )
            
        except Exception as e:
            logger.error(f"Error enviando mensajes a Telegram para usuario {submission.userId}: {e}")
            raise HTTPException(status_code=500, detail="Error comunicándose con Telegram")
        
        return APIResponse(
            status="ok",
            message="Dibujo recibido y procesado correctamente"
        )
        
    except HTTPException:
        # Re-lanzar HTTPExceptions tal como están
        raise
    except Exception as e:
        logger.error(f"Error inesperado en submit_drawing: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# --- 5. FUNCIONES DE LOS COMANDOS Y FASES DE LA CONVERSACIÓN ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Inicia una nueva sesión de visión remota al recibir /start."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    # Limpiar datos de sesión previa
    context.user_data.clear()
    context.user_data["session_data"] = {}
    
    # Mapear user_id con chat_id para la comunicación desde FastAPI
    user_chat_mapping[user.id] = chat_id
    logger.info(f"Mapeando user_id {user.id} -> chat_id {chat_id}")
    
    # Seleccionar objetivo aleatorio
    selected_target = random.choice(TARGET_POOL)
    context.user_data["target"] = selected_target
    
    # Generar referencia única de objetivo
    target_ref = f"PN-{random.randint(1000, 9999)}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
    context.user_data["target_ref"] = target_ref
    
    logger.info(f"Usuario {user.id} ({user.first_name}) inició sesión. Objetivo: {selected_target['name']} ({selected_target['id']})")
    
    await update.message.reply_html(
        f"Hola {user.mention_html()}.\n"
        f"Bienvenido al <b>Protocolo Nautilus v2.0</b>.\n\n"
        f"Tu objetivo es: <code>{target_ref}</code>\n\n"
        f"<b>FASE 1: GESTALT</b>\n"
        f"Describe tus impresiones primarias y percepciones generales."
    )
    return FASE_1_GESTALT

async def fase_1_gestalt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 1 y pide la Fase 2."""
    context.user_data["session_data"]["fase1"] = update.message.text
    context.user_data["session_data"]["fase1_timestamp"] = datetime.now().isoformat()
    
    logger.info(f"Usuario {update.effective_user.id} completó Fase 1.")
    
    await update.message.reply_html(
        "✅ <b>Fase 1 registrada</b>\n\n"
        "<b>FASE 2: DATOS SENSORIALES</b>\n"
        "Describe colores, texturas, sonidos, olores, "
        "sensaciones táctiles y cualquier impresión sensorial."
    )
    return FASE_2_SENSORIAL

async def fase_2_sensorial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 2 y presenta la Fase 3 (Boceto)."""
    context.user_data["session_data"]["fase2"] = update.message.text
    context.user_data["session_data"]["fase2_timestamp"] = datetime.now().isoformat()
    
    logger.info(f"Usuario {update.effective_user.id} completó Fase 2.")
    
    keyboard = [[
        InlineKeyboardButton("🎨 Abrir Lienzo Nautilus", web_app=WebAppInfo(url=CANVAS_URL))
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_html(
        "✅ <b>Datos sensoriales registrados</b>\n\n"
        "<b>FASE 3: BOCETO</b>\n"
        "Presiona el botón para abrir el lienzo y dibuja las formas, "
        "estructuras y elementos principales que percibes.\n\n"
        "💡 <i>Cuando termines tu dibujo, presiona 'Enviar Dibujo' en la Mini App.</i>",
        reply_markup=reply_markup
    )
    return FASE_3_BOCETO

async def fase_4_conceptual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 4 y pide confirmación para finalizar."""
    context.user_data["session_data"]["fase4"] = update.message.text
    context.user_data["session_data"]["fase4_timestamp"] = datetime.now().isoformat()
    
    logger.info(f"Usuario {update.effective_user.id} completó Fase 4.")
    
    await update.message.reply_html(
        "✅ <b>Fase 4 registrada</b>\n\n"
        "🎯 <b>Sesión completa</b>\n"
        "Toda la información del protocolo ha sido registrada.\n\n"
        "¿Estás listo para finalizar y revelar el objetivo?\n"
        "Envía /finalizar cuando estés preparado."
    )
    return FINALIZAR

async def finalizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Revela el objetivo y termina la sesión."""
    user_id = update.effective_user.id
    
    await update.message.reply_text("🔄 Finalizando sesión y revelando el objetivo...")
    
    target_info = context.user_data.get("target")
    if not target_info:
        logger.error(f"Usuario {user_id} intentó finalizar sin un objetivo.")
        await update.message.reply_text("❌ Error al recuperar el objetivo. Por favor, envía /start para comenzar de nuevo.")
        return ConversationHandler.END
    
    logger.info(f"Usuario {user_id} finalizó. Revelando {target_info['name']}.")
    
    # Enviar imagen del objetivo
    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=target_info["url"],
        caption=f"🎯 <b>El objetivo era:</b>\n{target_info['name']}\n\n"
                f"<b>ID del objetivo:</b> {target_info['id']}",
        parse_mode='HTML'
    )
    
    # Mensaje de cierre
    await update.message.reply_html(
        "🙏 <b>¡Gracias por participar en el Protocolo Nautilus!</b>\n\n"
        "Tu sesión de Visión Remota ha sido completada.\n"
        "Para iniciar una nueva sesión, envía /start."
    )
    
    # Limpiar datos y mapeo
    if user_id in user_chat_mapping:
        del user_chat_mapping[user_id]
        logger.info(f"Limpiado mapeo para user_id {user_id}")
    
    context.user_data.clear()
    return ConversationHandler.END

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancela la sesión actual y limpia los datos."""
    user = update.effective_user
    logger.info(f"Usuario {user.id} canceló la sesión.")
    
    # Limpiar mapeo
    if user.id in user_chat_mapping:
        del user_chat_mapping[user.id]
    
    context.user_data.clear()
    await update.message.reply_text(
        "❌ Sesión cancelada.\n\n"
        "Para empezar una nueva sesión, envía /start."
    )
    return ConversationHandler.END

async def mensaje_no_reconocido(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes fuera del flujo de conversación."""
    await update.message.reply_text(
        "🤔 No entiendo ese mensaje.\n\n"
        "Comandos disponibles:\n"
        "• /start - Iniciar nueva sesión\n"
        "• /cancelar - Cancelar sesión actual\n"
        "• /finalizar - Finalizar sesión (solo en Fase 4)"
    )

# --- 6. FUNCIONES UTILITARIAS ---

def setup_telegram_application() -> Application:
    """Configura y retorna la aplicación de Telegram."""
    logger.info("Configurando aplicación de Telegram...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # ConversationHandler principal
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            FASE_1_GESTALT: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_1_gestalt)],
            FASE_2_SENSORIAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_2_sensorial)],
            FASE_3_BOCETO: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_4_conceptual)],  # Nota: Salta directo a Fase 4 porque Fase 3 se maneja via API
            FASE_4_CONCEPTUAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_4_conceptual)],
            FINALIZAR: [CommandHandler("finalizar", finalizar)],
        },
        fallbacks=[CommandHandler("cancelar", cancelar)],
        allow_reentry=True
    )
    
    # Agregar manejadores
    app.add_handler(conv_handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, mensaje_no_reconocido))
    
    logger.info("Aplicación de Telegram configurada correctamente.")
    return app

async def run_telegram_bot():
    """Ejecuta el bot de Telegram."""
    global telegram_app
    
    try:
        telegram_app = setup_telegram_application()
        logger.info("Iniciando bot de Telegram...")
        
        # Inicializar y ejecutar el bot
        await telegram_app.initialize()
        await telegram_app.start()
        await telegram_app.updater.start_polling()
        
        logger.info("🤖 Bot de Telegram funcionando...")
        
        # Mantener el bot corriendo indefinidamente
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Cerrando bot de Telegram...")
        
    except Exception as e:
        logger.error(f"Error en el bot de Telegram: {e}")
        raise
    finally:
        if telegram_app and telegram_app.updater.running:
            await telegram_app.updater.stop()
        if telegram_app:
            await telegram_app.stop()
            await telegram_app.shutdown()

def run_fastapi_server():
    """Ejecuta el servidor FastAPI en un hilo separado."""
    logger.info("Iniciando servidor FastAPI...")
    
    try:
        uvicorn.run(
            app_fastapi,
            host=FASTAPI_HOST,
            port=FASTAPI_PORT,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Error en el servidor FastAPI: {e}")
        raise

async def run_concurrent_services():
    """Ejecuta tanto el bot de Telegram como FastAPI concurrentemente."""
    logger.info("🚀 Iniciando Protocolo Nautilus v2.0...")
    logger.info(f"FastAPI servidor: http://{FASTAPI_HOST}:{FASTAPI_PORT}")
    logger.info("Asegúrate de que ngrok esté funcionando y configurado en script.js")
    
    # Crear un hilo para FastAPI
    fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    fastapi_thread.start()
    
    # Esperar un momento para que FastAPI se inicie
    await asyncio.sleep(2)
    
    try:
        # Ejecutar el bot de Telegram
        await run_telegram_bot()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Recibida señal de interrupción. Cerrando servicios...")
    except Exception as e:
        logger.error(f"Error ejecutando servicios: {e}")
        raise

# --- 7. FUNCIÓN PRINCIPAL ---

def main():
    """Función principal que inicia todos los servicios."""
    try:
        # Verificar configuración
        logger.info("Verificando configuración...")
        logger.info(f"Token de Telegram: {'✅ Configurado' if TELEGRAM_TOKEN else '❌ Faltante'}")
        logger.info(f"FastAPI Host: {FASTAPI_HOST}")
        logger.info(f"FastAPI Port: {FASTAPI_PORT}")
        
        if not TELEGRAM_TOKEN:
            logger.error("❌ TELEGRAM_TOKEN no configurado. Verifica tu archivo .env")
            return
        
        # Ejecutar servicios concurrentes
        asyncio.run(run_concurrent_services())
        
    except KeyboardInterrupt:
        logger.info("👋 Protocolo Nautilus cerrado por el usuario.")
    except Exception as e:
        logger.error(f"💥 Error fatal: {e}")
        raise

if __name__ == "__main__":
    main()