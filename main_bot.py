# main_bot.py

# ==============================================================================
#                      PROTOCOLO NAUTILUS - TELEGRAM BOT v1.0
#                      (Versión Limpia y Estable)
# ==============================================================================

import os
import logging
import random
import base64
from io import BytesIO

from dotenv import load_dotenv

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

# Validar que el token existe
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN no encontrado. Verifica tu archivo .env")

# URL de tu lienzo en GitHub Pages
CANVAS_URL = "https://pixatrip1984.github.io/nautilus-canvas/"

# Configuración de logging para ver qué está pasando
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
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

# --- 2. POOL DE OBJETIVOS PREDEFINIDO ---
TARGET_POOL = [
    {
        "id": "T001", "name": "Las Pirámides de Giza con la Esfinge",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Le_Sphinx_et_les_Pyramides_de_Gizeh.jpg/1280px-Le_Sphinx_et_les_Pyramides_de_Gizeh.jpg",
    },
    {
        "id": "T002", "name": "El Puente Golden Gate en la niebla",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Golden_Gate_Bridge_at_sunset_1.jpg/1920px-Golden_Gate_Bridge_at_sunset_1.jpg",
    },
    {
        "id": "T003", "name": "La Torre Eiffel de noche",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_de_Nuit_-_Paris_2007_v2.jpg/1024px-Tour_Eiffel_de_Nuit_-_Paris_2007_v2.jpg",
    },
    {
        "id": "T004", "name": "La Cascada Seljalandsfoss en Islandia",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Seljalandsfoss_in_July_2021.jpg/1024px-Seljalandsfoss_in_July_2021.jpg",
    },
    {
        "id": "T005", "name": "El Transbordador Espacial despegando",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/The_Space_Shuttle_Challenger_lifts_off_-_GPN-2000-001293.jpg/1024px-The_Space_Shuttle_Challenger_lifts_off_-_GPN-2000-001293.jpg",
    },
]

# --- 3. FUNCIONES DE LOS COMANDOS Y FASES DE LA CONVERSACIÓN ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Inicia una nueva sesión de visión remota al recibir /start."""
    user = update.effective_user
    context.user_data.clear()
    context.user_data["session_data"] = {}

    selected_target = random.choice(TARGET_POOL)
    context.user_data["target"] = selected_target

    target_ref = f"PN-{random.randint(1000, 9999)}-{random.choice('XYZ')}"
    context.user_data["target_ref"] = target_ref

    logger.info(f"Usuario {user.id} ({user.first_name}) inició sesión. Objetivo: {selected_target['name']} ({selected_target['id']})")

    await update.message.reply_html(
        rf"Hola {user.mention_html()}."
        f"\nBienvenido al <b>Protocolo Nautilus</b>."
        f"\n\nTu objetivo es: <code>{target_ref}</code>"
        "\n\n<b>FASE 1: GESTALT</b>\nDescribe tus impresiones primarias.",
    )
    return FASE_1_GESTALT

async def fase_1_gestalt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 1 y pide la Fase 2."""
    context.user_data["session_data"]["fase1"] = update.message.text
    logger.info(f"Usuario {update.effective_user.id} completó Fase 1.")
    await update.message.reply_html(
        "Recibido.\n\n<b>FASE 2: DATOS SENSORIALES</b>\nDescribe colores, texturas, sonidos, etc."
    )
    return FASE_2_SENSORIAL

async def fase_2_sensorial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 2 y pide la Fase 3 (Boceto)."""
    context.user_data["session_data"]["fase2"] = update.message.text
    logger.info(f"Usuario {update.effective_user.id} completó Fase 2.")

    keyboard = [[
        InlineKeyboardButton("Abrir Lienzo Nautilus 🎨", web_app=WebAppInfo(url=CANVAS_URL))
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_html(
        "Datos sensoriales guardados.\n\n<b>FASE 3: BOCETO</b>\n"
        "Presiona el botón para dibujar las formas principales. "
        "Cuando termines, pulsa 'Enviar Dibujo'.",
        reply_markup=reply_markup,
    )
    return FASE_3_BOCETO

async def fase_3_boceto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la imagen del lienzo (web_app_data) y pide la Fase 4."""
    if not update.effective_message or not update.effective_message.web_app_data:
        await update.message.reply_text("Por favor, usa el botón 'Abrir Lienzo' para enviar tu boceto.")
        return FASE_3_BOCETO
    
    try:
        data_url = update.effective_message.web_app_data.data
        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image_stream = BytesIO(image_data)
        
        context.user_data["session_data"]["fase3_boceto_bytes"] = image_stream.getvalue()
        
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=image_stream,
            caption="Boceto recibido."
        )
        logger.info(f"Usuario {update.effective_user.id} completó Fase 3 (Boceto).")
        
        await update.message.reply_html(
            "¡Excelente! Ahora, <b>FASE 4: CONCEPTUAL</b>\n"
            "Describe las cualidades, intangibles e impresiones abstractas."
        )
        return FASE_4_CONCEPTUAL

    except Exception as e:
        logger.error(f"Error procesando web app data de {update.effective_user.id}: {e}")
        await update.message.reply_text("Hubo un error recibiendo el dibujo. Por favor, inténtalo de nuevo.")
        return FASE_3_BOCETO

async def fase_4_conceptual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 4 y pide confirmación para finalizar."""
    context.user_data["session_data"]["fase4"] = update.message.text
    logger.info(f"Usuario {update.effective_user.id} completó Fase 4.")
    await update.message.reply_text(
        "Toda la información ha sido registrada.\n\n"
        "¿Listo para finalizar? Envía /finalizar."
    )
    return FINALIZAR

async def finalizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """(Versión Prototipo) Revela el objetivo y termina la sesión."""
    await update.message.reply_text("Sesión finalizada. Revelando el objetivo...")
    
    target_info = context.user_data.get("target")
    if not target_info:
        logger.error(f"Usuario {update.effective_user.id} intentó finalizar sin un objetivo.")
        await update.message.reply_text("Error al recuperar el objetivo. Por favor, /start.")
        return ConversationHandler.END

    logger.info(f"Usuario {update.effective_user.id} finalizó. Revelando {target_info['name']}.")

    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=target_info["url"],
        caption=f"<b>El objetivo era: {target_info['name']}</b>",
        parse_mode='HTML'
    )
    
    await update.message.reply_text("¡Gracias por participar! Para una nueva sesión, envía /start.")
    context.user_data.clear()
    return ConversationHandler.END

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancela la sesión actual y limpia los datos."""
    user = update.effective_user
    logger.info(f"Usuario {user.id} canceló la sesión.")
    context.user_data.clear()
    await update.message.reply_text("Sesión cancelada. Para empezar de nuevo, envía /start.")
    return ConversationHandler.END

# --- 4. FUNCIÓN PRINCIPAL Y EJECUCIÓN DEL BOT ---

def main() -> None:
    """Función principal que construye y ejecuta el bot."""
    logger.info("Iniciando Bot de Protocolo Nautilus...")
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            FASE_1_GESTALT: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_1_gestalt)],
            FASE_2_SENSORIAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_2_sensorial)],
            FASE_3_BOCETO: [MessageHandler(filters.StatusUpdate.WEB_APP_DATA, fase_3_boceto)],
            FASE_4_CONCEPTUAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_4_conceptual)],
            FINALIZAR: [CommandHandler("finalizar", finalizar)],
        },
        fallbacks=[CommandHandler("cancelar", cancelar)],
        allow_reentry=True
    )

    app.add_handler(conv_handler)
    
    logger.info("El bot está configurado y listo para escuchar...")
    app.run_polling()


if __name__ == "__main__":
    main()