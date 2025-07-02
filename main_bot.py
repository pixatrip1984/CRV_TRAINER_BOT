# main_bot.py (Versión con Estrategia de Guardado Local y Envío Manual)

# ==============================================================================
#                      PROTOCOLO NAUTILUS - TELEGRAM BOT v1.1
# ==============================================================================

import os
import logging
import random
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

# Configuración de logging para depuración
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Definición de estados para la conversación
(
    FASE_1_GESTALT,
    FASE_2_SENSORIAL,
    FASE_3_BOCETO, # Este estado ahora esperará un mensaje de tipo FOTO
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
    """Inicia una nueva sesión al recibir /start."""
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
        "\n\n<b>FASE 1: GESTALT</b>\nDescribe tus impresiones primarias (ej: 'Estructura artificial', 'Agua y tierra').",
    )
    return FASE_1_GESTALT

async def fase_1_gestalt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 1 y pide la Fase 2."""
    context.user_data["session_data"]["fase1"] = update.message.text
    logger.info(f"Usuario {update.effective_user.id} completó Fase 1.")
    await update.message.reply_html(
        "Recibido.\n\n<b>FASE 2: DATOS SENSORIALES</b>\nDescribe solo sensaciones: colores, texturas, sonidos, olores."
    )
    return FASE_2_SENSORIAL

async def fase_2_sensorial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 2 y PIDE al usuario que envíe el boceto como una foto."""
    context.user_data["session_data"]["fase2"] = update.message.text
    logger.info(f"Usuario {update.effective_user.id} completó Fase 2.")
    
    keyboard = [[
        InlineKeyboardButton("Abrir Lienzo Nautilus 🎨", web_app=WebAppInfo(url=CANVAS_URL))
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_html(
        "Datos sensoriales guardados.\n\n<b>FASE 3: BOCETO DIMENSIONAL</b>\n\n"
        "1. Presiona el botón para abrir el lienzo.\n"
        "2. Dibuja y luego presiona 'Finalizar y Preparar para Enviar'.\n"
        "3. Guarda la imagen en tu dispositivo (PC o móvil).\n"
        "4. Cierra la ventana del lienzo y <b>envía la imagen que guardaste aquí en el chat.</b>",
        reply_markup=reply_markup,
    )
    return FASE_3_BOCETO

async def fase_3_boceto_recibido(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la FOTO del boceto y pide la Fase 4."""
    if not update.message.photo:
        await update.message.reply_text("Por favor, envía tu boceto como una foto para continuar.")
        return FASE_3_BOCETO

    photo_file = await update.message.photo[-1].get_file()
    boceto_bytearray = await photo_file.download_as_bytearray()
    
    context.user_data["session_data"]["fase3_boceto_bytes"] = bytes(boceto_bytearray)
    logger.info(f"Usuario {update.effective_user.id} completó Fase 3 (Boceto recibido como foto).")

    await update.message.reply_html(
        "Boceto recibido con éxito.\n\n"
        "¡Excelente! Ahora, <b>FASE 4: CONCEPTUAL</b>\n"
        "Describe las cualidades, intangibles y tus impresiones emocionales."
    )
    return FASE_4_CONCEPTUAL

async def fase_4_conceptual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Recibe la Fase 4 y pide confirmación para finalizar."""
    context.user_data["session_data"]["fase4"] = update.message.text
    logger.info(f"Usuario {update.effective_user.id} completó Fase 4.")
    await update.message.reply_text(
        "Toda la información ha sido registrada.\n\n"
        "Envía /finalizar para revelar el objetivo o /cancelar para abortar."
    )
    return FINALIZAR

async def finalizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Revela el objetivo y termina la sesión."""
    await update.message.reply_text("Sesión finalizada. Revelando el objetivo...")
    
    target_info = context.user_data.get("target")
    if not target_info:
        logger.error(f"Usuario {update.effective_user.id} intentó finalizar sin objetivo.")
        await update.message.reply_text("Error al recuperar el objetivo. Por favor, inicia de nuevo con /start.")
        return ConversationHandler.END

    logger.info(f"Usuario {update.effective_user.id} finalizó sesión. Revelando {target_info['name']}.")
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
    """Cancela la sesión actual."""
    user = update.effective_user
    logger.info(f"Usuario {user.id} canceló la sesión.")
    context.user_data.clear()
    await update.message.reply_text("Sesión cancelada. Puedes empezar de nuevo cuando quieras con /start.")
    return ConversationHandler.END

async def manejar_entrada_incorrecta(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes que no se esperan en una fase concreta."""
    # Podrías añadir lógica para saber en qué fase está el usuario
    await update.message.reply_text("La entrada no es válida para esta fase. Por favor, sigue las instrucciones.")

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
            # ¡EL GRAN CAMBIO ESTÁ AQUÍ! Ahora espera una FOTO.
            FASE_3_BOCETO: [MessageHandler(filters.PHOTO, fase_3_boceto_recibido)],
            FASE_4_CONCEPTUAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_4_conceptual)],
            FINALIZAR: [CommandHandler("finalizar", finalizar)],
        },
        fallbacks=[
            CommandHandler("cancelar", cancelar),
            # Maneja cualquier otro mensaje que no sea un comando para dar una respuesta útil
            MessageHandler(filters.ALL, manejar_entrada_incorrecta)
        ],
        allow_reentry=True
    )

    app.add_handler(conv_handler)
    
    logger.info("El bot está configurado y listo para escuchar...")
    app.run_polling()


if __name__ == "__main__":
    main()