# main_bot.py - Protocolo Nautilus v3.0 (Upload Bridge Architecture)

import os
import logging
import random
from datetime import datetime

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    Application, CommandHandler, ContextTypes, MessageHandler,
    ConversationHandler, filters,
)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN no encontrado. Verifica tu archivo .env")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Estados
FASE_1_GESTALT, FASE_2_SENSORIAL, FASE_3_BOCETO, FASE_4_CONCEPTUAL, FINALIZAR = range(5)

# Pool de objetivos
TARGET_POOL = [
    {"id": "T001", "name": "Las Pirámides de Giza", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Le_Sphinx_et_les_Pyramides_de_Gizeh.jpg/1280px-Le_Sphinx_et_les_Pyramides_de_Gizeh.jpg"},
    {"id": "T002", "name": "Golden Gate", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Golden_Gate_Bridge_at_sunset_1.jpg/1920px-Golden_Gate_Bridge_at_sunset_1.jpg"},
    {"id": "T003", "name": "Torre Eiffel", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_de_Nuit_-_Paris_2007_v2.jpg/1024px-Tour_Eiffel_de_Nuit_-_Paris_2007_v2.jpg"},
]

import time
CANVAS_URL = f"https://pixatrip1984.github.io/nautilus-canvas/index.html?cache_bust={int(time.time())}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    context.user_data.clear()
    
    target = random.choice(TARGET_POOL)
    context.user_data["target"] = target
    context.user_data["target_ref"] = f"PN-{random.randint(1000, 9999)}-{random.choice('XYZ')}"
    
    logger.info(f"Usuario {user.id} inició sesión. Objetivo: {target['name']}")
    
    await update.message.reply_html(
        f"Hola {user.mention_html()}.\n"
        f"<b>Protocolo Nautilus v3.0</b>\n\n"
        f"Objetivo: <code>{context.user_data['target_ref']}</code>\n\n"
        f"<b>FASE 1: GESTALT</b>\nDescribe impresiones primarias."
    )
    return FASE_1_GESTALT

async def fase_1_gestalt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["fase1"] = update.message.text
    await update.message.reply_html("<b>FASE 2: SENSORIALES</b>\nDescribe colores, texturas, sonidos.")
    return FASE_2_SENSORIAL

async def fase_2_sensorial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["fase2"] = update.message.text
    
    keyboard = [[InlineKeyboardButton("🎨 Abrir Lienzo", web_app=WebAppInfo(url=CANVAS_URL))]]
    
    await update.message.reply_html(
        "<b>FASE 3: BOCETO</b>\nDibuja formas principales.\n\n"
        "💡 <i>Presiona 'Enviar Dibujo' cuando termines.</i>",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return FASE_3_BOCETO

async def handle_drawing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Maneja el dibujo recibido como foto"""
    if update.message.photo:
        # Guardar la foto
        photo = update.message.photo[-1]
        context.user_data["drawing_file_id"] = photo.file_id
        
        await update.message.reply_html(
            "✅ <b>Boceto recibido</b>\n\n"
            "<b>FASE 4: CONCEPTUAL</b>\n"
            "Describe cualidades intangibles e impresiones abstractas."
        )
        return FASE_4_CONCEPTUAL
    
    await update.message.reply_text("Por favor, usa el lienzo para enviar tu dibujo.")
    return FASE_3_BOCETO

async def fase_4_conceptual(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data["fase4"] = update.message.text
    await update.message.reply_text("Sesión completa. Envía /finalizar para revelar el objetivo.")
    return FINALIZAR

async def finalizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    target = context.user_data.get("target")
    if not target:
        await update.message.reply_text("Error. Envía /start para comenzar.")
        return ConversationHandler.END
    
    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=target["url"],
        caption=f"🎯 <b>El objetivo era:</b>\n{target['name']}",
        parse_mode='HTML'
    )
    
    context.user_data.clear()
    return ConversationHandler.END

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Sesión cancelada. Envía /start para empezar.")
    return ConversationHandler.END

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            FASE_1_GESTALT: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_1_gestalt)],
            FASE_2_SENSORIAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_2_sensorial)],
            FASE_3_BOCETO: [
                MessageHandler(filters.PHOTO, handle_drawing),
                MessageHandler(filters.TEXT & ~filters.COMMAND, fase_4_conceptual)
            ],
            FASE_4_CONCEPTUAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, fase_4_conceptual)],
            FINALIZAR: [CommandHandler("finalizar", finalizar)],
        },
        fallbacks=[CommandHandler("cancelar", cancelar)],
        allow_reentry=True
    )
    
    app.add_handler(conv_handler)
    
    logger.info("🚀 Protocolo Nautilus v3.0 iniciado")
    app.run_polling()

if __name__ == "__main__":
    main()