# main_bot.py

# ==============================================================================
#                      PROTOCOLO NAUTILUS - TELEGRAM BOT v3.2
#                    (Sistema de Ranking + Código Optimizado)
# ==============================================================================

import os
import logging
import random
import base64
import asyncio
import re
import json
import hashlib
from io import BytesIO
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime

import requests
from bs4 import BeautifulSoup
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

from database_manager import NautilusDB

# --- 1. CONFIGURACIÓN INICIAL ---
# REEMPLAZA ESTE BLOQUE COMPLETO

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PUBLIC_BOT_URL = os.getenv("PUBLIC_BOT_URL") # <-- AÑADIDO: Carga la URL pública
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

if not TELEGRAM_TOKEN: 
    raise ValueError("TELEGRAM_TOKEN no encontrado en el archivo .env.")

# <-- AÑADIDO: Validación crítica para la URL pública
if not PUBLIC_BOT_URL:
    raise ValueError("PUBLIC_BOT_URL no encontrada en el archivo .env. ¡Es necesaria para que la WebApp funcione!")

CANVAS_URL = "https://pixatrip1984.github.io/nautilus-canvas/"
DATA_FILE = "nautilus_research_data.json"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler('nautilus_bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

(FASE_1_GESTALT, FASE_2_SENSORIAL, FASE_3_BOCETO, FASE_4_CONCEPTUAL, FINALIZAR) = range(5)

user_sessions: Dict[int, Dict[str, Any]] = {}
telegram_app: Optional[Application] = None

# Base de datos de rankings
nautilus_db: Optional[NautilusDB] = None
session_start_times: Dict[int, datetime] = {}  # Para trackear tiempos de sesión

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
    """Genera coordenadas siguiendo el estándar de programas formales de percepción remota."""
    formats = [
        lambda: f"{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
        lambda: f"{random.randint(1000, 9999)}-{random.randint(10000, 99999)}",
        lambda: f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}",
        lambda: f"{random.randint(100000, 999999)}",
        lambda: f"{random.randint(1000, 9999)}-{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{random.randint(1, 9)}",
        lambda: f"{random.randint(10000, 99999)}",
    ]
    return random.choice(formats)()

# --- 4. SISTEMA DE PSEUDÓNIMOS Y DATOS ---
def get_user_pseudonym(user_id: int) -> str:
    """Genera un pseudónimo consistente para un user_id."""
    hash_input = f"nautilus_{user_id}_salt_2025"
    hash_digest = hashlib.md5(hash_input.encode()).hexdigest()
    
    prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Theta", "Lambda", "Sigma", "Omega", 
                "Nova", "Quasar", "Pulsar", "Nebula", "Cosmos", "Astral", "Stellar", "Lunar", "Solar", "Vortex"]
    suffixes = ["Explorer", "Seeker", "Voyager", "Navigator", "Observer", "Perceiver", "Sensor", "Scanner", 
                "Detector", "Finder", "Hunter", "Tracker", "Reader", "Viewer", "Seer", "Oracle", "Mystic"]
    
    prefix_idx = int(hash_digest[:2], 16) % len(prefixes)
    suffix_idx = int(hash_digest[2:4], 16) % len(suffixes)
    number = int(hash_digest[4:6], 16) % 1000
    
    return f"{prefixes[prefix_idx]}{suffixes[suffix_idx]}{number:03d}"

def save_session_data(user_id: int, session_data: dict, score: float):
    """Guarda los datos de la sesión para research futuro."""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"sessions": [], "high_score_targets": []}
        
        session_record = {
            "timestamp": datetime.now().isoformat(),
            "user_pseudonym": get_user_pseudonym(user_id),
            "coordinates": session_data.get("target_ref", "unknown"),
            "target_name": session_data.get("target", {}).get("name", "unknown"),
            "target_url": session_data.get("target", {}).get("url", ""),
            "score": score,
            "phases": {
                "gestalt": session_data.get("fase1", ""),
                "sensorial": session_data.get("fase2", ""),
                "conceptual": session_data.get("fase4", ""),
            }
        }
        
        data["sessions"].append(session_record)
        
        if score > 7.0:
            target_exists = any(t["url"] == session_record["target_url"] for t in data["high_score_targets"])
            if not target_exists:
                data["high_score_targets"].append({
                    "name": session_record["target_name"],
                    "url": session_record["target_url"],
                    "average_score": score,
                    "success_count": 1
                })
            else:
                for target in data["high_score_targets"]:
                    if target["url"] == session_record["target_url"]:
                        target["success_count"] += 1
                        target["average_score"] = (target["average_score"] + score) / 2
                        break
        
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Datos de sesión guardados para {get_user_pseudonym(user_id)}")
        
    except Exception as e:
        logger.error(f"Error guardando datos de sesión: {e}")

# --- 5. FUNCIONES DE UTILIDAD PARA IMÁGENES ---
def validate_image_url_basic(url: str) -> bool:
    """Validación básica mejorada de URL de imagen."""
    if not url or not isinstance(url, str):
        return False
    
    if not url.startswith(('http://', 'https://')):
        return False
    
    url_lower = url.lower()
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.svg']
    has_image_ext = any(ext in url_lower for ext in image_extensions)
    
    trusted_domains = [
        'unsplash.com', 'pixabay.com', 'pexels.com', 'wikimedia.org',
        'flickr.com', 'imgur.com', 'cloudinary.com', 'amazonaws.com'
    ]
    is_trusted_domain = any(domain in url_lower for domain in trusted_domains)
    
    forbidden_patterns = [
        'profile', 'avatar', 'user', 'people', 'person', 'face',
        'social', 'facebook', 'instagram', 'twitter', 'selfie'
    ]
    has_forbidden = any(pattern in url_lower for pattern in forbidden_patterns)
    
    return (has_image_ext or is_trusted_domain) and not has_forbidden

async def validate_image_content_with_llm(image_url: str) -> bool:
    """Valida que el contenido de una imagen sea apropiado usando Mistral Vision."""
    if not openrouter_client:
        return True
    
    try:
        response = await asyncio.to_thread(requests.get, image_url, timeout=10)
        if response.status_code != 200:
            return False
            
        image_b64 = base64.b64encode(response.content).decode('utf-8')
        
        validation_prompt = """Analiza esta imagen y determina si es apropiada para percepción remota.

CRITERIOS DE APROBACIÓN:
✅ Paisajes naturales, arquitectura histórica, monumentos, jardines
✅ Lugares públicos sin personas visibles
✅ Estructuras, edificios, formaciones naturales

CRITERIOS DE RECHAZO:
❌ Personas, rostros, cuerpos humanos (incluso parciales)
❌ Contenido violento, traumático o controvertido
❌ Lugares de guerra, desastres, accidentes
❌ Contenido privado o inapropiado

Responde ÚNICAMENTE con "APROPIADA" o "RECHAZADA"."""

        response = await asyncio.to_thread(
            openrouter_client.chat.completions.create,
            model=MISTRAL_CLOUD_MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": validation_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            temperature=0.1,
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        is_appropriate = "APROPIADA" in result
        
        logger.info(f"Validación de imagen: {result} ({'✅' if is_appropriate else '❌'})")
        return is_appropriate
        
    except Exception as e:
        logger.error(f"Error validando contenido de imagen: {e}")
        return False

def get_fallback_target() -> Dict[str, str]:
    """Objetivos de emergencia completamente seguros y específicos."""
    emergency_targets = [
        {
            "name": "Templo de Piedra Ancestral",
            "url": "https://images.unsplash.com/photo-1520637836862-4d197d17c93a?w=800&h=600&fit=crop",
            "description": "Templo de piedra ancestral con arquitectura histórica - objetivo de emergencia validado",
            "validation_description": "Estructura de piedra antigua con columnas y escalinatas"
        },
        {
            "name": "Lago de Montaña Sereno",
            "url": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop",
            "description": "Lago cristalino en valle montañoso con reflejos naturales - objetivo de emergencia validado",
            "validation_description": "Cuerpo de agua transparente rodeado de montañas y vegetación"
        },
        {
            "name": "Faro Histórico Costero",
            "url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&h=600&fit=crop",
            "description": "Faro de piedra histórico en acantilado costero - objetivo de emergencia validado",
            "validation_description": "Torre cilíndrica de piedra blanca sobre formación rocosa costera"
        },
        {
            "name": "Puente de Arco de Piedra",
            "url": "https://images.unsplash.com/photo-1516026672322-bc52d61a55d5?w=800&h=600&fit=crop",
            "description": "Puente histórico de arco de piedra sobre río - objetivo de emergencia validado",
            "validation_description": "Estructura arqueada de piedra gris con soportes cilíndricos"
        }
    ]
    
    selected = random.choice(emergency_targets)
    logger.info(f"🛡️ Usando objetivo de emergencia: {selected['name']}")
    return selected

# --- 6. GENERACIÓN INTELIGENTE DE TÉRMINOS CON LLM ---
async def generate_search_term_with_llm() -> str:
    """Usa Mistral para generar términos de búsqueda éticos y seguros de manera inteligente."""
    if not openrouter_client:
        fallback_terms = [
            "ancient stone bridge peaceful landscape",
            "historic lighthouse coastal scenery",
            "traditional wooden temple garden",
            "serene mountain lake reflection",
            "classical marble fountain courtyard"
        ]
        return random.choice(fallback_terms)
    
    try:
        logger.info("Generando término de búsqueda con Mistral...")
        
        system_prompt = """Eres un especialista en percepción remota y selección ética de objetivos. Tu tarea es generar términos de búsqueda únicos y seguros para encontrar imágenes apropiadas para sesiones de percepción remota controlada."""
        
        user_prompt = """Genera un término de búsqueda único y específico para encontrar una imagen ética apropiada para percepción remota.

**CRITERIOS OBLIGATORIOS DE SEGURIDAD:**
• SOLO lugares, arquitectura, paisajes naturales, monumentos históricos
• NUNCA personas, rostros, cuerpos humanos, multitudes
• NUNCA contenido violento, traumático, controvertido o perturbador
• NUNCA sitios de guerra, desastres, accidentes, cementerios
• NUNCA contenido religioso controvertido o símbolos polarizantes
• NUNCA ubicaciones privadas o con posibles problemas de privacidad

**TIPOS APROPIADOS (elige uno):**
1. **Arquitectura Histórica:** templos antiguos, catedrales, puentes de piedra, faros, observatorios, bibliotecas clásicas, castillos, monasterios
2. **Paisajes Naturales:** montañas, lagos, cascadas, bosques, valles, praderas, costas rocosas, formaciones geológicas
3. **Jardines y Espacios:** jardines zen, patios históricos, plazas públicas, fuentes clásicas, laberintos de jardín
4. **Monumentos Culturales:** estatuas (sin personas), obeliscos, arcos triunfales, columnas históricas, estructuras astronómicas

**INSTRUCCIONES:**
- Genera SOLO el término de búsqueda (máximo 4-5 palabras en inglés)
- Sé específico pero no demasiado restrictivo
- Incluye adjetivos descriptivos que sugieran tranquilidad
- Evita términos ambiguos que puedan retornar contenido inapropiado
- Cada término debe ser único y creativo
- NO uses comillas ni caracteres especiales

**EJEMPLOS DE BUENOS TÉRMINOS:**
- ancient stone lighthouse peaceful coast
- serene mountain temple garden
- historic marble fountain courtyard
- tranquil forest waterfall scene
- classical observatory dome architecture

**FORMATO DE RESPUESTA:**
Responde ÚNICAMENTE con el término de búsqueda sin comillas, sin explicaciones adicionales.

Genera ahora un término único y seguro:"""

        response = await asyncio.to_thread(
            openrouter_client.chat.completions.create,
            model=MISTRAL_CLOUD_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=50
        )
        
        search_term = response.choices[0].message.content.strip()
        search_term = search_term.replace('"', '').replace("'", '').replace('`', '').strip()
        
        forbidden_words = [
            'person', 'people', 'human', 'man', 'woman', 'child', 'face', 'body',
            'war', 'battle', 'death', 'grave', 'cemetery', 'disaster', 'accident',
            'violence', 'blood', 'weapon', 'gun', 'bomb', 'fire', 'destruction'
        ]
        
        search_term_lower = search_term.lower()
        if any(word in search_term_lower for word in forbidden_words):
            logger.warning(f"Término generado contiene palabras prohibidas: {search_term}")
            return "peaceful ancient stone temple"
        
        if len(search_term.split()) > 6:
            logger.warning(f"Término generado muy largo: {search_term}")
            search_term = " ".join(search_term.split()[:5])
        
        logger.info(f"Término de búsqueda generado: '{search_term}'")
        return search_term
        
    except Exception as e:
        logger.error(f"Error generando término con LLM: {e}")
        return "ancient peaceful stone temple"

# --- 7. FUNCIONES DE BÚSQUEDA DE IMÁGENES ---
async def search_duckduckgo_images_real(query: str, max_results: int = 5) -> List[str]:
    """Busca imágenes usando la librería duckduckgo_search."""
    try:
        from duckduckgo_search import DDGS
        import time
        
        logger.info(f"Buscando imágenes con DDGS para: '{query}'")
        
        def search_images():
            results = []
            try:
                with DDGS() as ddgs:
                    search_params = {
                        "keywords": query,
                        "region": "us-en",
                        "safesearch": "On",
                        "size": "Medium",
                        "type_image": None,
                        "layout": None,
                        "license_image": None,
                        "max_results": max_results * 3
                    }
                    
                    for r in ddgs.images(**search_params):
                        try:
                            image_url = r.get("image")
                            if image_url and validate_image_url_basic(image_url):
                                url_lower = image_url.lower()
                                forbidden_url_parts = ['person', 'people', 'face', 'human', 'man', 'woman']
                                if not any(part in url_lower for part in forbidden_url_parts):
                                    results.append(image_url)
                                    logger.debug(f"URL válida encontrada: {image_url[:100]}...")
                                    
                                    if len(results) >= max_results:
                                        break
                            
                            time.sleep(0.1)
                            
                        except Exception as e:
                            logger.debug(f"Error procesando resultado individual: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error en búsqueda DDGS detallado: {e}")
                
            return results
        
        image_urls = await asyncio.to_thread(search_images)
        logger.info(f"DDGS encontró {len(image_urls)} imágenes válidas")
        return image_urls
        
    except ImportError:
        logger.error("Librería duckduckgo_search no instalada")
        return []
    except Exception as e:
        logger.error(f"Error general en DDGS: {e}")
        return []

async def search_duckduckgo_images_fallback(query: str, max_results: int = 5) -> List[str]:
    """Fallback mejorado con web scraping."""
    try:
        logger.info(f"Usando fallback de web scraping para: '{query}'")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
            'DNT': '1'
        }
        
        search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&t=h_&iax=images&ia=images&safe=strict"
        
        response = await asyncio.to_thread(requests.get, search_url, headers=headers, timeout=20)
        response.raise_for_status()
        
        logger.debug(f"Respuesta HTTP: {response.status_code}, Tamaño: {len(response.content)} bytes")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        image_urls = []
        
        img_tags = soup.find_all('img', src=True)
        for img in img_tags:
            src = img.get('src')
            if src and validate_image_url_basic(src) and 'http' in src:
                if len(image_urls) < max_results:
                    image_urls.append(src)
                    logger.debug(f"Imagen encontrada (directa): {src[:100]}...")
        
        if len(image_urls) < max_results:
            data_attrs = ['data-src', 'data-original', 'data-lazy', 'data-image']
            for attr in data_attrs:
                elements = soup.find_all(attrs={attr: True})
                for element in elements:
                    url = element.get(attr)
                    if url and validate_image_url_basic(url) and 'http' in url:
                        if len(image_urls) < max_results:
                            image_urls.append(url)
                            logger.debug(f"Imagen encontrada ({attr}): {url[:100]}...")
        
        if len(image_urls) < max_results:
            json_urls = extract_images_from_scripts(soup)
            for url in json_urls:
                if validate_image_url_basic(url):
                    if len(image_urls) < max_results:
                        image_urls.append(url)
                        logger.debug(f"Imagen encontrada (JSON): {url[:100]}...")
        
        if not image_urls:
            logger.info("No se encontraron imágenes, probando términos alternativos")
            simple_terms = query.split()[:2]
            simple_query = " ".join(simple_terms)
            
            if simple_query != query:
                return await search_unsplash_alternative(simple_query, max_results)
        
        logger.info(f"Web scraping encontró {len(image_urls)} imágenes válidas")
        return image_urls
        
    except Exception as e:
        logger.error(f"Error en fallback de web scraping: {e}")
        return []

def extract_images_from_scripts(soup) -> List[str]:
    """Extrae URLs de imágenes desde scripts JSON mejorado."""
    urls = []
    try:
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string and 'http' in script.string:
                patterns = [
                    r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif|svg)(?:\?[^\s"\'<>]*)?',
                    r'"image":\s*"(https?://[^\s"\'<>]+)"',
                    r'"url":\s*"(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif))"',
                    r'"src":\s*"(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif))"'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, script.string, re.IGNORECASE)
                    for match in matches:
                        url = match if isinstance(match, str) else match[0] if isinstance(match, tuple) else str(match)
                        if validate_image_url_basic(url):
                            urls.append(url)
                            
    except Exception as e:
        logger.debug(f"Error extrayendo de scripts: {e}")
    return urls

async def search_unsplash_alternative(query: str, max_results: int = 5) -> List[str]:
    """Búsqueda alternativa usando Unsplash como último recurso."""
    try:
        logger.info(f"Usando Unsplash como alternativa para: '{query}'")
        
        unsplash_alternatives = {
            'ancient temple': 'https://images.unsplash.com/photo-1520637836862-4d197d17c93a?w=800&h=600&fit=crop',
            'stone bridge': 'https://images.unsplash.com/photo-1516026672322-bc52d61a55d5?w=800&h=600&fit=crop',
            'historic bridge': 'https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=800&h=600&fit=crop',
            'lighthouse': 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&h=600&fit=crop',
            'mountain lake': 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop',
            'waterfall': 'https://images.unsplash.com/photo-1447752875215-b2761acb3c5d?w=800&h=600&fit=crop',
            'garden': 'https://images.unsplash.com/photo-1547036967-23d11aacaee0?w=800&h=600&fit=crop',
            'fountain': 'https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=800&h=600&fit=crop',
            'cathedral': 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop',
            'castle': 'https://images.unsplash.com/photo-1464822759844-d150baec0494?w=800&h=600&fit=crop'
        }
        
        query_lower = query.lower()
        for key, url in unsplash_alternatives.items():
            if key in query_lower:
                logger.info(f"Encontrada alternativa de Unsplash para '{key}': {url}")
                return [url]
        
        default_urls = list(unsplash_alternatives.values())
        return [random.choice(default_urls)]
        
    except Exception as e:
        logger.error(f"Error en búsqueda alternativa de Unsplash: {e}")
        return []

async def search_duckduckgo_images(query: str, max_results: int = 5) -> List[str]:
    """Función principal que coordina todas las estrategias de búsqueda de imágenes."""
    logger.info(f"Iniciando búsqueda de imágenes para: '{query}'")
    
    results = await search_duckduckgo_images_real(query, max_results)
    if results:
        logger.info(f"✅ DDGS exitoso: {len(results)} imágenes")
        return results
    
    logger.info("DDGS falló, intentando web scraping...")
    results = await search_duckduckgo_images_fallback(query, max_results)
    if results:
        logger.info(f"✅ Web scraping exitoso: {len(results)} imágenes")
        return results
    
    if len(query.split()) > 2:
        simple_query = " ".join(query.split()[:2])
        logger.info(f"Probando términos simplificados: '{simple_query}'")
        
        results = await search_duckduckgo_images_real(simple_query, max_results)
        if results:
            logger.info(f"✅ Términos simplificados exitoso: {len(results)} imágenes")
            return results
    
    logger.info("Todas las búsquedas fallaron, usando Unsplash alternativo...")
    results = await search_unsplash_alternative(query, max_results)
    if results:
        logger.info(f"✅ Unsplash alternativo exitoso: {len(results)} imágenes")
        return results
    
    logger.warning("Todas las estrategias de búsqueda fallaron")
    return []

async def select_ethical_target_dynamic() -> Dict[str, str]:
    """Selecciona un objetivo ético usando LLM para generar términos de búsqueda."""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"Intento {attempt + 1} de selección de objetivo")
            
            search_term = await generate_search_term_with_llm()
            image_urls = await search_duckduckgo_images(search_term)
            
            for url in image_urls:
                if await validate_image_content_with_llm(url):
                    target_name = " ".join(word.capitalize() for word in search_term.split()[:3])
                    
                    target = {
                        "name": target_name,
                        "url": url,
                        "description": f"Objetivo generado dinámicamente: {search_term}",
                        "search_term": search_term,
                        "generation_method": "llm_dynamic"
                    }
                    
                    logger.info(f"✅ Objetivo dinámico seleccionado: {target_name}")
                    return target
            
            logger.warning(f"Intento {attempt + 1}: No se encontraron imágenes válidas para '{search_term}'")
            
        except Exception as e:
            logger.error(f"Error en intento {attempt + 1}: {e}")
    
    logger.warning("Todos los intentos fallaron, usando objetivo de emergencia")
    return get_fallback_target()

# === SISTEMA DE ANÁLISIS PERCEPTUAL MEJORADO ===
# Agregar estas funciones al main_bot.py

async def analyze_target_with_mistral(image_bytes: bytes) -> Dict[str, Any]:
    """
    El LLM realiza un protocolo completo de percepción remota viendo la imagen objetivo.
    Esto genera datos esperables para comparar con las percepciones del usuario.
    """
    if not openrouter_client:
        return {"error": "Modelo de análisis en la nube no disponible"}
    
    try:
        logger.info("Realizando análisis perceptual completo con Mistral...")
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        analysis_prompt = """Actúa como un EXPERTO EN PERCEPCIÓN REMOTA con 20 años de experiencia. 

Tu tarea es analizar esta imagen como si fueras un perceptor experimentado realizando el protocolo completo de 4 fases. Debes generar DATOS PERCEPTUALES ESPERABLES que un vidente competente podría captar del objetivo.

**IMPORTANTE:** No describas solo lo que ves visualmente. Infiere y deduce sensaciones, impresiones y datos sutiles que se derivarían lógicamente del entorno y contexto.

**FASE 1 - DATOS GESTALT (Impresiones Primitivas):**
• Sensaciones táctiles esperables basadas en materiales y entorno
• Impresiones dimensionales deducibles de la escala y proporciones
• Datos primitivos de forma y estructura
• Sensaciones de densidad y masa aparente

**FASE 2 - DATOS SENSORIALES (Información Específica):**
• Colores dominantes y tonalidades presentes
• Texturas de superficie lógicas según materiales
• Sonidos ambientales esperables del entorno
• Sensaciones térmicas basadas en contexto (agua, sombra, materiales)
• Impresiones de humedad/sequedad según el ambiente
• Sensaciones de movimiento o estatismo

**FASE 4 - DATOS CONCEPTUALES (Información Abstracta):**
• Emociones que evoca el lugar (tranquilidad, grandeza, misterio)
• Propósito funcional o histórico del sitio
• Atmósfera y energía del lugar
• Significado simbólico o cultural
• Contexto temporal (antiguo/moderno)
• Sensación de actividad humana (habitado/abandonado)

**ANÁLISIS CONTEXTUAL ADICIONAL:**
• Factores ambientales que influirían en las sensaciones
• Elementos que podrían generar impresiones específicas
• Detalles sutiles que un perceptor experimentado captaría

Responde en formato JSON con la siguiente estructura:
{
  "gestalt": {
    "tactiles": ["sensación1", "sensación2"],
    "dimensionales": ["impresión1", "impresión2"],
    "estructurales": ["dato1", "dato2"],
    "densidad_masa": "descripción"
  },
  "sensoriales": {
    "colores": ["color1", "color2"],
    "texturas": ["textura1", "textura2"],
    "sonidos": ["sonido1", "sonido2"],
    "termicas": ["sensación1", "sensación2"],
    "humedad": "descripción",
    "movimiento": "descripción"
  },
  "conceptuales": {
    "emocionales": ["emoción1", "emoción2"],
    "funcional": "propósito",
    "atmosfera": "descripción",
    "simbolico": "significado",
    "temporal": "contexto",
    "actividad": "descripción"
  },
  "contextual": {
    "factores_ambientales": ["factor1", "factor2"],
    "impresiones_sutiles": ["impresión1", "impresión2"],
    "elementos_clave": ["elemento1", "elemento2"]
  }
}"""

        response = await asyncio.to_thread(
            openrouter_client.chat.completions.create,
            model=MISTRAL_CLOUD_MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }],
            temperature=0.3,
            max_tokens=2000
        )
        
        # Parsear respuesta JSON
        response_text = response.choices[0].message.content.strip()
        
        # Limpiar respuesta si tiene markdown
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        perceptual_data = json.loads(response_text)
        logger.info("✅ Análisis perceptual completo generado")
        
        return perceptual_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parseando respuesta JSON del análisis: {e}")
        return {"error": "Error en formato de respuesta del análisis"}
    except Exception as e:
        logger.error(f"Error en análisis perceptual con Mistral: {e}")
        return {"error": f"Error en análisis: {str(e)}"}

async def get_enhanced_professional_analysis(user_transcript: str, target_perceptual_data: Dict, 
                                           sketch_desc: str, target_name: str, coordinates: str) -> str:
    """
    Genera análisis profesional mejorado comparando datos del usuario vs datos esperables del objetivo.
    """
    if not openrouter_client:
        return "Análisis de texto no disponible."
    
    logger.info("Generando análisis profesional mejorado con comparación perceptual...")
    
    system_prompt = """Eres un Analista Senior de Percepción Remota con experiencia en protocolos científicos estándar y evaluación perceptual avanzada.

Tu evaluación debe ser profesional, constructiva y basada en criterios establecidos de correlación perceptual. Debes evaluar no solo coincidencias literales, sino capacidades perceptuales reales.

IMPORTANTE: Tienes acceso a datos perceptuales esperables generados por análisis experto del objetivo. Usa esto para evaluar si el perceptor captó sensaciones, impresiones y datos que DEBERÍAN haberse percibido del entorno."""
    
    # Convertir datos perceptuales a texto legible
    perceptual_summary = format_perceptual_data_for_analysis(target_perceptual_data)
    
    user_prompt = f"""ANÁLISIS AVANZADO DE SESIÓN DE PERCEPCIÓN REMOTA

**COORDENADAS DEL OBJETIVO:** {coordinates}
**OBJETIVO REAL:** {target_name}

**DATOS DEL PERCEPTOR:**
---
{user_transcript}
---

**ANÁLISIS DEL BOCETO:**
---
{sketch_desc}
---

**DATOS PERCEPTUALES ESPERABLES DEL OBJETIVO:**
---
{perceptual_summary}
---

**INSTRUCCIONES PARA ANÁLISIS MEJORADO:**

Genera un informe profesional evaluando CAPACIDADES PERCEPTUALES REALES:

## 📋 Resumen Ejecutivo
- Evaluación de precisión perceptual vs datos esperables
- Análisis de capacidades demostradas
- Puntuación preliminar y justificación

## 🎯 Análisis de Correspondencias Perceptuales

### Correspondencias Directas (Datos Captados Correctamente)
- Sensaciones táctiles que coinciden con el entorno
- Impresiones dimensionales acertadas
- Datos sensoriales lógicos captados
- Elementos conceptuales coherentes con el objetivo

### Correspondencias Inferidas (Capacidad Deductiva)
- Sensaciones derivadas lógicamente del entorno que el perceptor captó
- Impresiones contextualmente apropiadas
- Datos sutiles que requieren experiencia perceptual

### Correspondencias Abstractas (Percepción Avanzada)
- Captación de atmósfera y energía del lugar
- Percepción de propósito o función
- Datos conceptuales y emocionales apropiados

## 🧠 Evaluación de Capacidades Perceptuales

### Capacidades Demostradas
- Tipos de percepción exitosa (táctil, dimensional, emocional)
- Nivel de precisión en diferentes categorías
- Indicadores de experiencia perceptual

### Oportunidades de Desarrollo
- Áreas sensoriales no exploradas adecuadamente
- Tipos de datos que podrían mejorarse
- Técnicas para fortalecer capacidades específicas

## 📊 Análisis del Boceto vs Realidad Perceptual
- Traducción visual de impresiones percibidas
- Coherencia entre descripciones verbales y representación gráfica
- Elementos únicos captados solo en el boceto

## ⚠️ Análisis de Desviaciones
- Datos que no corresponden al objetivo (ruido vs. información válida)
- Posibles interpretaciones alternativas
- Factores que podrían haber influido en las percepciones

## 📈 Evaluación Cuantitativa Mejorada
**Criterios de Evaluación:**
- **Precisión Sensorial (30%):** Captación de datos táctiles, térmicos, texturales
- **Precisión Dimensional (25%):** Percepción de escala, proporciones, estructura
- **Precisión Conceptual (25%):** Atmósfera, propósito, contexto temporal
- **Capacidad Inferencial (20%):** Deducción de datos lógicos del entorno

**Puntuación Final:** [X.X/10.0]

## 💡 Retroalimentación Específica para Desarrollo
- Fortalezas perceptuales identificadas
- Técnicas específicas para mejorar áreas débiles
- Ejercicios recomendados para el siguiente nivel
- Patrones de mejora observados

**NOTA:** Evalúa capacidades perceptuales reales, no solo coincidencias superficiales. Un perceptor competente debería captar sensaciones lógicamente derivables del entorno objetivo."""

    try:
        response = await asyncio.to_thread(
            openrouter_client.chat.completions.create,
            model=MISTRAL_CLOUD_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generando análisis profesional mejorado: {e}")
        return "Error: El servicio de análisis profesional mejorado no está disponible."

def format_perceptual_data_for_analysis(perceptual_data: Dict) -> str:
    """Convierte los datos perceptuales JSON a texto legible para el análisis."""
    if "error" in perceptual_data:
        return f"Error en análisis perceptual: {perceptual_data['error']}"
    
    try:
        formatted = []
        
        # Datos Gestalt
        if "gestalt" in perceptual_data:
            gestalt = perceptual_data["gestalt"]
            formatted.append("**DATOS GESTALT ESPERABLES:**")
            formatted.append(f"• Tactiles: {', '.join(gestalt.get('tactiles', []))}")
            formatted.append(f"• Dimensionales: {', '.join(gestalt.get('dimensionales', []))}")
            formatted.append(f"• Estructurales: {', '.join(gestalt.get('estructurales', []))}")
            formatted.append(f"• Densidad/Masa: {gestalt.get('densidad_masa', 'N/A')}")
            formatted.append("")
        
        # Datos Sensoriales
        if "sensoriales" in perceptual_data:
            sensoriales = perceptual_data["sensoriales"]
            formatted.append("**DATOS SENSORIALES ESPERABLES:**")
            formatted.append(f"• Colores: {', '.join(sensoriales.get('colores', []))}")
            formatted.append(f"• Texturas: {', '.join(sensoriales.get('texturas', []))}")
            formatted.append(f"• Sonidos: {', '.join(sensoriales.get('sonidos', []))}")
            formatted.append(f"• Térmicas: {', '.join(sensoriales.get('termicas', []))}")
            formatted.append(f"• Humedad: {sensoriales.get('humedad', 'N/A')}")
            formatted.append(f"• Movimiento: {sensoriales.get('movimiento', 'N/A')}")
            formatted.append("")
        
        # Datos Conceptuales
        if "conceptuales" in perceptual_data:
            conceptuales = perceptual_data["conceptuales"]
            formatted.append("**DATOS CONCEPTUALES ESPERABLES:**")
            formatted.append(f"• Emocionales: {', '.join(conceptuales.get('emocionales', []))}")
            formatted.append(f"• Funcional: {conceptuales.get('funcional', 'N/A')}")
            formatted.append(f"• Atmósfera: {conceptuales.get('atmosfera', 'N/A')}")
            formatted.append(f"• Simbólico: {conceptuales.get('simbolico', 'N/A')}")
            formatted.append(f"• Temporal: {conceptuales.get('temporal', 'N/A')}")
            formatted.append(f"• Actividad: {conceptuales.get('actividad', 'N/A')}")
            formatted.append("")
        
        # Contexto
        if "contextual" in perceptual_data:
            contextual = perceptual_data["contextual"]
            formatted.append("**ANÁLISIS CONTEXTUAL:**")
            formatted.append(f"• Factores Ambientales: {', '.join(contextual.get('factores_ambientales', []))}")
            formatted.append(f"• Impresiones Sutiles: {', '.join(contextual.get('impresiones_sutiles', []))}")
            formatted.append(f"• Elementos Clave: {', '.join(contextual.get('elementos_clave', []))}")
        
        return "\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error formateando datos perceptuales: {e}")
        return "Error formateando datos perceptuales para análisis."

# === CORRECCIÓN DE FORMATEO - REEMPLAZAR EN main_bot.py ===

def format_analysis_for_telegram(analysis_text: str) -> str:
    """Convierte el análisis en formato limpio para Telegram usando HTML con validación."""
    lines = analysis_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Convertir headers de markdown a HTML
        if line.startswith('# '):
            formatted_lines.append(f"\n🔹 <b>{line[2:].strip()}</b>\n")
        elif line.startswith('## '):
            formatted_lines.append(f"\n📋 <b>{line[3:].strip()}</b>")
        elif line.startswith('### '):
            formatted_lines.append(f"\n• <b>{line[4:].strip()}</b>")
        # Convertir elementos de lista
        elif line.startswith('- ') or line.startswith('• '):
            formatted_lines.append(f"  • {line[2:].strip()}")
        # Convertir texto en negrita (pero validar que esté balanceado)
        elif '**' in line:
            # Contar asteriscos para asegurar balance
            asterisk_count = line.count('**')
            if asterisk_count % 2 == 0:  # Número par = balanceado
                line = line.replace('**', '')
                formatted_lines.append(f"<b>{line}</b>")
            else:
                # Si no está balanceado, remover asteriscos sin formatear
                line = line.replace('**', '')
                formatted_lines.append(line)
        # Texto normal
        else:
            formatted_lines.append(line)
    
    # Unir y limpiar
    result = '\n'.join(formatted_lines)
    
    # Limpiar caracteres problemáticos más agresivamente
    result = result.replace('*', '')
    result = result.replace('#', '')
    result = result.replace('`', '')
    result = result.replace('[', '')
    result = result.replace(']', '')
    result = result.replace('(', '')
    result = result.replace(')', '')
    
    # VALIDACIÓN CRÍTICA: Asegurar que todas las etiquetas HTML estén balanceadas
    result = validate_and_fix_html_tags(result)
    
    return result

def validate_and_fix_html_tags(text: str) -> str:
    """Valida y corrige etiquetas HTML mal balanceadas."""
    import re
    
    # Contar etiquetas de apertura y cierre
    open_b_tags = len(re.findall(r'<b>', text))
    close_b_tags = len(re.findall(r'</b>', text))
    
    open_i_tags = len(re.findall(r'<i>', text))
    close_i_tags = len(re.findall(r'</i>', text))
    
    # Corregir etiquetas <b> desbalanceadas
    if open_b_tags > close_b_tags:
        # Agregar etiquetas de cierre faltantes
        text += '</b>' * (open_b_tags - close_b_tags)
    elif close_b_tags > open_b_tags:
        # Remover etiquetas de cierre excesivas
        excess_closes = close_b_tags - open_b_tags
        for _ in range(excess_closes):
            text = text.replace('</b>', '', 1)
    
    # Corregir etiquetas <i> desbalanceadas
    if open_i_tags > close_i_tags:
        text += '</i>' * (open_i_tags - close_i_tags)
    elif close_i_tags > open_i_tags:
        excess_closes = close_i_tags - open_i_tags
        for _ in range(excess_closes):
            text = text.replace('</i>', '', 1)
    
    return text

async def send_analysis_safely(bot, chat_id: int, analysis_text: str, title: str = "ANÁLISIS PERCEPTUAL AVANZADO"):
    """Envía análisis con manejo seguro de longitud y formato."""
    try:
        formatted_analysis = format_analysis_for_telegram(analysis_text)
        
        # LÍMITE ESTRICTO: 3500 caracteres por mensaje para evitar errores
        max_length = 3500
        
        if len(formatted_analysis) > max_length:
            # Dividir en partes más pequeñas
            parts = []
            current_part = ""
            
            lines = formatted_analysis.split('\n')
            for line in lines:
                if len(current_part + line + '\n') > max_length:
                    if current_part.strip():
                        parts.append(current_part.strip())
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'
            
            # Agregar la última parte
            if current_part.strip():
                parts.append(current_part.strip())
            
            # Enviar cada parte
            for i, part in enumerate(parts):
                header = f"📊 <b>{title} - Parte {i+1}/{len(parts)}</b>\n\n" if i == 0 else ""
                
                # Validar cada parte antes de enviar
                safe_part = validate_and_fix_html_tags(header + part)
                
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=safe_part,
                        parse_mode='HTML'
                    )
                    await asyncio.sleep(0.5)  # Pausa para evitar flood
                except Exception as e:
                    logger.error(f"Error enviando parte {i+1}: {e}")
                    # Fallback: enviar sin HTML
                    clean_text = safe_part.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
                    await bot.send_message(
                        chat_id=chat_id,
                        text=f"📊 {title} - Parte {i+1}/{len(parts)}\n\n{clean_text}"
                    )
                    await asyncio.sleep(0.5)
        else:
            # Enviar como mensaje único
            safe_text = validate_and_fix_html_tags(f"📊 <b>{title}</b>\n\n{formatted_analysis}")
            
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=safe_text,
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"Error enviando análisis completo: {e}")
                # Fallback: enviar sin HTML
                clean_text = safe_text.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>', '')
                await bot.send_message(
                    chat_id=chat_id,
                    text=clean_text
                )
        
        return True
        
    except Exception as e:
        logger.error(f"Error crítico enviando análisis: {e}")
        # Último fallback: mensaje de error
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=f"📊 <b>{title}</b>\n\n⚠️ Error procesando análisis detallado. El sistema de análisis perceptual funcionó correctamente, pero hubo un problema de formato en la respuesta."
            )
        except:
            pass
        return False

def extract_score_from_analysis(analysis_text: str) -> float:
    """Extrae la puntuación numérica del análisis."""
    try:
        score_patterns = [
            r'(\d+\.?\d*)/10',
            r'Puntuación.*?(\d+\.?\d*)',
            r'Score.*?(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*/\s*10'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, analysis_text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                return min(score, 10.0)
        
        return 5.0
        
    except:
        return 5.0

# --- 9. MODELOS PYDANTIC ---
class DrawingSubmission(BaseModel):
    imageData: str
    userId: int
    targetCoordinates: Optional[str] = None

class APIResponse(BaseModel):
    status: str
    message: Optional[str] = None

# --- 10. SERVIDOR API (FastAPI) ---
app_fastapi = FastAPI(title="Protocolo Nautilus API", version="3.2.0")
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
        
        user_sessions[user_id]["session_data"]["fase3_boceto_bytes"] = image_data
        
        logger.info(f"Iniciando análisis inmediato del boceto para usuario {user_id}")
        sketch_desc = await describe_sketch_with_mistral(image_data)
        user_sessions[user_id]["session_data"]["sketch_description"] = sketch_desc
        logger.info(f"Análisis del boceto completado para usuario {user_id}")
        
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

# --- 11. FUNCIONES DE IA ESPECIALIZADAS ---
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
Usa terminología apropiada del campo y mantén un tono alentador pero riguroso.
IMPORTANTE: Debes incluir una puntuación numérica final en formato "X.X/10.0" al final del análisis."""
    
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

# --- 12. HANDLERS DE TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    user_id = user.id
    
    session_start_times[user_id] = datetime.now()
    user_sessions[user_id] = {"chat_id": update.effective_chat.id, "session_data": {}}
    
    protocol_message = f"""🧠 <b>PROTOCOLO NAUTILUS v3.2</b>
<i>Sistema de Percepción Remota Controlada con Ranking</i>

Hola {user.mention_html()}, bienvenido al sistema más avanzado de percepción remota.

<b>🔬 ¿QUÉ HAREMOS?</b>
Participarás en una sesión científica de 4 fases donde percibirás información sobre un objetivo remoto usando solo sus coordenadas numéricas.

<b>🛡️ MEDIDAS DE SEGURIDAD</b>
• Solo objetivos éticos: arquitectura histórica, paisajes naturales, arte
• Sin contenido traumático, violento o controvertido
• Búsqueda automática con filtros de seguridad
• Protección de identidad con pseudónimos

<b>⚗️ CÓMO FUNCIONA</b>
• <b>IA de Búsqueda:</b> Selecciona objetivos seguros en tiempo real
• <b>IA de Análisis:</b> Evalúa tu boceto objetivamente  
• <b>IA de Correlación:</b> Compara tus datos con el objetivo real

<b>🏆 SISTEMA DE RANKING</b>
• Puntos por precisión, detalles, tiempo y calidad
• Compite con otros perceptores en el ranking global
• Seguimiento de tu progreso personal

<b>📊 INVESTIGACIÓN</b>
Tus datos (anónimos) contribuyen a la investigación sobre percepción remota. Tu pseudónimo: <code>{get_user_pseudonym(user.id)}</code>

<b>⏳ PREPARANDO OBJETIVO...</b>
<i>Buscando objetivo ético usando DuckDuckGo...</i>"""

    await update.message.reply_html(protocol_message)
    
    try:
        target_ref = generate_professional_coordinates()
        selected_target = await select_ethical_target_dynamic()
        
        user_sessions[user.id]["session_data"]["target"] = selected_target
        user_sessions[user.id]["session_data"]["target_ref"] = target_ref
        
        logger.info(f"Usuario {user.id} ({user.first_name}) inició sesión. Objetivo: {selected_target['name']}, Coordenadas: {target_ref}")
        
        success_message = f"""✅ <b>OBJETIVO SELECCIONADO</b>

<b>Coordenadas asignadas:</b> <code>{target_ref}</code>

<b>FASE 1: IMPRESIONES GESTALT</b>
Describe tus <b>primeras impresiones</b> sobre el objetivo:

• Sensaciones táctiles (rugoso, suave, frío, cálido)
• Impresiones dimensionales (grande, pequeño, alto, ancho)  
• Datos primitivos de forma o estructura

<i>Las impresiones son sutiles como \"recuerdos descoloridos\". Confía en tus primeras intuiciones.</i>"""
        
        await update.message.reply_html(success_message)
        
    except Exception as e:
        logger.error(f"Error en búsqueda de objetivo: {e}")
        target_ref = generate_professional_coordinates()
        selected_target = get_fallback_target()
        user_sessions[user.id]["session_data"]["target"] = selected_target
        user_sessions[user.id]["session_data"]["target_ref"] = target_ref
        
        fallback_message = f"""✅ <b>OBJETIVO SELECCIONADO</b> (Fallback)

<b>Coordenadas asignadas:</b> <code>{target_ref}</code>

<b>FASE 1: IMPRESIONES GESTALT</b>
Describe tus <b>primeras impresiones</b> sobre el objetivo:

• Sensaciones táctiles
• Impresiones dimensionales  
• Datos primitivos de forma

<i>Confía en tus primeras intuiciones.</i>"""
        
        await update.message.reply_html(fallback_message)
    
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

# REEMPLAZA ESTA FUNCIÓN COMPLETA

async def fase_2_sensorial(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    target_ref = user_sessions[user_id]["session_data"]["target_ref"]
    
    user_sessions[user_id]["session_data"]["fase2"] = update.message.text
    
    # --- CAMBIO IMPORTANTE: Construimos la URL pasando la del backend ---
    # La URL de ngrok (PUBLIC_BOT_URL) se añade como un parámetro 'backend_url'
    webapp_url = f"{CANVAS_URL}?target={target_ref}&backend_url={PUBLIC_BOT_URL}"
    
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

async def safe_send_photo(bot, chat_id: int, photo_url: str, caption: str, parse_mode: str = 'HTML'):
    """Envía una foto de manera segura con fallback en caso de error."""
    try:
        await bot.send_photo(
            chat_id=chat_id,
            photo=photo_url,
            caption=caption,
            parse_mode=parse_mode
        )
        return True
    except Exception as e:
        logger.error(f"Error enviando foto desde URL {photo_url}: {e}")
        
        # Fallback: intentar descargar y enviar como bytes
        try:
            response = requests.get(photo_url, timeout=10)
            response.raise_for_status()
            
            await bot.send_photo(
                chat_id=chat_id,
                photo=BytesIO(response.content),
                caption=caption,
                parse_mode=parse_mode
            )
            logger.info("✅ Imagen enviada usando fallback con bytes")
            return True
        except Exception as e2:
            logger.error(f"Error en fallback de imagen: {e2}")
            
            # Último fallback: enviar solo el texto sin imagen
            await bot.send_message(
                chat_id=chat_id,
                text=f"🎯 <b>REVELACIÓN DEL OBJETIVO</b>\n\n{caption}\n\n<i>⚠️ La imagen no pudo ser cargada. URL: {photo_url}</i>",
                parse_mode=parse_mode
            )
            return False

# === FUNCIÓN finalizar() MEJORADA ===
async def finalizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id not in user_sessions:
        await update.message.reply_text("❌ No se encontró una sesión activa. Por favor, /start.")
        return ConversationHandler.END
    
    await update.message.reply_text("⏳ <b>Procesando análisis perceptual avanzado...</b>\n<i>🧠 IA realizando protocolo completo del objetivo...\n👁️ Comparando capacidades perceptuales...</i>", parse_mode='HTML')
    
    session_data = user_sessions[user_id].get("session_data", {})
    target_info = session_data.get("target")
    target_ref = session_data.get("target_ref", "????-????")
    
    if not target_info:
        await update.message.reply_text("❌ Error al recuperar objetivo. Por favor, /start.")
        return ConversationHandler.END

    logger.info(f"Usuario {user_id} finalizó. Objetivo: {target_info['name']}, Coordenadas: {target_ref}")
    
    user_transcript = (
        f"FASE 1 - Impresiones Gestalt:\n{session_data.get('fase1', 'N/A')}\n\n"
        f"FASE 2 - Datos Sensoriales:\n{session_data.get('fase2', 'N/A')}\n\n"
        f"FASE 4 - Datos Conceptuales:\n{session_data.get('fase4', 'N/A')}"
    )
    
    # === NUEVO SISTEMA DE ANÁLISIS PERCEPTUAL ===
    
    # 1. Descargar imagen del objetivo
    target_image_bytes = None
    try:
        response = requests.get(target_info["url"], timeout=15)
        response.raise_for_status()
        target_image_bytes = response.content
        logger.info("✅ Imagen del objetivo descargada correctamente")
    except Exception as e:
        logger.error(f"No se pudo descargar imagen del objetivo: {e}")
    
    # 2. Generar análisis perceptual completo del objetivo
    target_perceptual_data = {}
    if target_image_bytes:
        try:
            await update.message.reply_text("🧠 <b>IA analizando objetivo...</b>\n<i>Generando datos perceptuales esperables...</i>", parse_mode='HTML')
            target_perceptual_data = await analyze_target_with_mistral(target_image_bytes)
            
            if "error" not in target_perceptual_data:
                logger.info("✅ Análisis perceptual del objetivo completado")
            else:
                logger.warning(f"Error en análisis perceptual: {target_perceptual_data['error']}")
                
        except Exception as e:
            logger.error(f"Error en análisis perceptual del objetivo: {e}")
            target_perceptual_data = {"error": str(e)}
    
    # 3. Obtener descripción del boceto (mantenemos BLIP para descripción visual básica)
    sketch_desc = session_data.get("sketch_description", "El perceptor no proporcionó un boceto.")
    if sketch_desc == "El perceptor no proporcionó un boceto.":
        user_drawing_bytes = session_data.get("fase3_boceto_bytes")
        if user_drawing_bytes:
            sketch_desc = await describe_sketch_with_mistral(user_drawing_bytes)

    # 4. Generar análisis profesional mejorado
    await update.message.reply_text("📊 <b>Generando análisis comparativo...</b>\n<i>Evaluando capacidades perceptuales...</i>", parse_mode='HTML')
    
    session_analysis = await get_enhanced_professional_analysis(
        user_transcript, target_perceptual_data, sketch_desc, target_info['name'], target_ref
    )
    
    score = extract_score_from_analysis(session_analysis)
    
    # === SISTEMA DE RANKING (MANTENER IGUAL) ===
    user_pseudonym = get_user_pseudonym(user_id)
    total_points = 0
    user_position = "?"
    is_new_record = False
    
    try:
        global nautilus_db
        if nautilus_db:
            session_start_time = session_start_times.get(user_id)
            
            total_points = await nautilus_db.save_session_to_db(
                user_id, user_pseudonym, session_data, score, session_start_time
            )
            
            user_position = nautilus_db.get_user_ranking_position(user_pseudonym)
            
            previous_best = nautilus_db.get_user_best_score(user_pseudonym)
            if previous_best and len(previous_best) > 0:
                is_new_record = total_points > previous_best[0]
            else:
                is_new_record = True
            
            logger.info(f"Usuario {user_pseudonym}: {total_points} puntos, posición #{user_position}")
            
    except Exception as e:
        logger.error(f"Error en sistema de ranking: {e}")
        total_points = int(score * 100)
    
    # Limpiar tiempo de sesión
    if user_id in session_start_times:
        del session_start_times[user_id]
    
    # Guardar datos para investigación
    save_session_data(user_id, session_data, score)
    
    # === REVELACIÓN DEL OBJETIVO CON DATOS PERCEPTUALES ===
    
    # Crear mensaje de revelación mejorado
    revelation_caption = f"<b>Coordenadas:</b> <code>{target_ref}</code>\n<b>Objetivo Real:</b> {target_info['name']}\n\n<i>{target_info.get('description', 'Objetivo de percepción remota controlada')}</i>"
    
    # Agregar datos perceptuales esperables si están disponibles
    if target_perceptual_data and "error" not in target_perceptual_data:
        revelation_caption += "\n\n<b>🧠 DATOS PERCEPTUALES ESPERABLES:</b>"
        
        if "gestalt" in target_perceptual_data:
            gestalt = target_perceptual_data["gestalt"]
            if gestalt.get("tactiles"):
                revelation_caption += f"\n• <i>Táctiles:</i> {', '.join(gestalt['tactiles'][:3])}"
            if gestalt.get("dimensionales"):
                revelation_caption += f"\n• <i>Dimensionales:</i> {', '.join(gestalt['dimensionales'][:2])}"
        
        if "sensoriales" in target_perceptual_data:
            sensoriales = target_perceptual_data["sensoriales"]
            if sensoriales.get("termicas"):
                revelation_caption += f"\n• <i>Térmicas:</i> {', '.join(sensoriales['termicas'][:2])}"
        
        revelation_caption += "\n\n<i>¿Captaste alguna de estas sensaciones?</i>"
    
    # Enviar revelación del objetivo con manejo seguro de errores
    await safe_send_photo(
        context.bot,
        user_id,
        target_info["url"],
        revelation_caption
    )
    
    # === ENVIAR ANÁLISIS PROFESIONAL MEJORADO ===
    if "Error:" in session_analysis:
        await context.bot.send_message(
            chat_id=user_id, 
            text=f"⚠️ <b>No se pudo generar el análisis perceptual avanzado</b>\n\n{session_analysis}", 
            parse_mode='HTML'
        )
    else:
        # Usar la nueva función de envío seguro
        success = await send_analysis_safely(
            context.bot, 
            user_id, 
            session_analysis, 
            "ANÁLISIS PERCEPTUAL AVANZADO"
        )
        
        if not success:
            # Si todo falla, enviar resumen básico
            await context.bot.send_message(
                chat_id=user_id,
                text=f"📊 <b>ANÁLISIS COMPLETADO</b>\n\n✅ Tu sesión fue analizada con el nuevo sistema perceptual\n🎯 Puntuación obtenida: {int(score * 100)}/1000\n\n<i>El análisis detallado está siendo procesado...</i>",
                parse_mode='HTML'
            )
    
    # === MOSTRAR PUNTUACIÓN Y RANKING (MANTENER IGUAL) ===
    try:
        if nautilus_db:
            user_best = nautilus_db.get_user_best_score(user_pseudonym)
            if user_best:
                detail_bonus = user_best[2]
                time_bonus = user_best[3]
                quality_bonus = user_best[4]
            else:
                detail_bonus = total_points - int(score * 100) - 125
                time_bonus = 50
                quality_bonus = 75
            
            points_message = f"""🎯 <b>TU PUNTUACIÓN TOTAL</b>

🔮 <b>{total_points}</b> puntos obtenidos
📍 <b>Posición #{user_position}</b> en el ranking global

<b>📊 Desglose de Puntos:</b>
• Score Perceptual: <b>{int(score * 100)}</b> pts
• Bonus Detalles: <b>📡 {detail_bonus}</b> pts  
• Bonus Tiempo: <b>⏱️ {time_bonus}</b> pts
• Bonus Calidad: <b>🎯 {quality_bonus}</b> pts

{'🏆 <b>¡NUEVO RÉCORD PERSONAL!</b>' if is_new_record else '📈 Sigue entrenando para mejorar tu capacidad perceptual'}

<b>🧠 NUEVA CARACTERÍSTICA:</b>
<i>Ahora el análisis evalúa capacidades perceptuales reales basadas en datos esperables del entorno objetivo.</i>"""

            await context.bot.send_message(
                chat_id=user_id,
                text=points_message,
                parse_mode='HTML'
            )
            
            # Mostrar ranking actual (top 5)
            current_rankings = nautilus_db.get_global_ranking(5)
            if current_rankings:
                ranking_message = nautilus_db.format_ranking_message(current_rankings, user_pseudonym)
                
                await context.bot.send_message(
                    chat_id=user_id,
                    text=ranking_message,
                    parse_mode='HTML'
                )
            
            # Estadísticas generales
            stats = nautilus_db.get_ranking_stats()
            stats_message = f"""📊 <b>ESTADÍSTICAS DEL SISTEMA</b>

🎯 Total de sesiones: <b>{stats['total_sessions']}</b>
👥 Perceptores únicos: <b>{stats['unique_users']}</b>  
📈 Promedio de puntos: <b>{stats['average_points']}</b>
🏆 Récord absoluto: <b>{stats['highest_score']}</b> pts

<b>🆕 ANÁLISIS PERCEPTUAL v3.2:</b>
✅ Evaluación de capacidades sensoriales reales
✅ Comparación con datos esperables del entorno
✅ Entrenamiento basado en percepción contextual

<i>Cada sesión mejora la comprensión científica de la percepción remota.</i>"""
            
            await context.bot.send_message(
                chat_id=user_id,
                text=stats_message,
                parse_mode='HTML'
            )
            
    except Exception as e:
        logger.error(f"Error mostrando ranking: {e}")
    
    # === MENSAJE DE CIERRE MEJORADO ===
    pseudonym = get_user_pseudonym(user_id)
    await update.message.reply_html(
        f"🙏 <b>Sesión de Entrenamiento Perceptual Completada</b>\n\n"
        f"Gracias por participar, <b>{pseudonym}</b>!\n"
        f"Tu puntuación perceptual: <b>{total_points}</b> puntos\n"
        f"Posición actual: <b>#{user_position}</b>\n\n"
        f"<b>🧠 Sistema de Entrenamiento Avanzado:</b>\n"
        f"• Evaluación de capacidades perceptuales reales\n"
        f"• Comparación con datos esperables del entorno\n"
        f"• Desarrollo de habilidades sensoriales específicas\n"
        f"• Competencia basada en precisión perceptual\n\n"
        f"<i>¡El nuevo sistema evalúa tu capacidad real de percepción remota!</i>\n\n"
        f"Para una nueva sesión de entrenamiento, envía /start\n"
        f"Ver ranking perceptual: /ranking"
    )
    
    del user_sessions[user_id]
    return ConversationHandler.END

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id in user_sessions: 
        del user_sessions[user_id]
    await update.message.reply_text("❌ Sesión cancelada. Envía /start para comenzar una nueva sesión de percepción remota.")
    return ConversationHandler.END

# --- 13. COMANDOS ADICIONALES ---
async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Proporciona información sobre el protocolo de percepción remota."""
    info_text = """🧠 <b>PROTOCOLO NAUTILUS - Información Detallada</b>

<b>¿Qué es la Percepción Remota?</b>
Es la capacidad de obtener información sobre un objetivo distante usando medios extrasensoriales. No se trata de "ver" el objetivo, sino de percibir datos sutiles.

<b>🔬 Metodología Científica:</b>
• Coordenadas aleatorias generadas automáticamente
• Búsqueda de objetivos éticos en tiempo real
• Análisis objetivo mediante IA especializada
• Datos anónimos para investigación

<b>🛡️ Seguridad Garantizada:</b>
• Solo lugares históricos, arquitectura, paisajes
• Filtros automáticos contra contenido traumático
• Protección de identidad con pseudónimos

<b>🏆 Sistema de Puntuación:</b>
• Score base del análisis LLM (100-1000 pts)
• Bonus por riqueza de detalles (0-200 pts)
• Bonus por eficiencia temporal (0-100 pts)
• Bonus por calidad descriptiva (0-100 pts)

<b>📊 Sistema de Research:</b>
• Cada sesión contribuye a la base de datos científica
• Objetivos exitosos se identifican automáticamente
• Análisis de patrones de percepción remota"""
    await update.message.reply_html(info_text)

async def ranking(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra el ranking global de mejores puntuaciones."""
    try:
        if not nautilus_db:
            await update.message.reply_text("❌ Sistema de ranking no disponible.")
            return
        
        user_pseudonym = get_user_pseudonym(update.effective_user.id)
        rankings = nautilus_db.get_global_ranking(10)
        
        if not rankings:
            await update.message.reply_html("🔮 <b>RANKING NAUTILUS</b>\n\nAún no hay datos de ranking disponibles.\n¡Sé el primero en completar una sesión!")
            return
        
        ranking_message = nautilus_db.format_ranking_message(rankings, user_pseudonym)
        
        # Agregar información personal del usuario
        user_best = nautilus_db.get_user_best_score(user_pseudonym)
        user_position = nautilus_db.get_user_ranking_position(user_pseudonym)
        
        if user_best:
            personal_info = f"\n\n🎯 <b>TU MEJOR PUNTUACIÓN</b>\n📍 Posición: #{user_position}\n🔮 Puntos: {user_best[0]}\n🎯 Objetivo: {user_best[5]} - {user_best[6]}"
            ranking_message += personal_info
        
        await update.message.reply_html(ranking_message)
        
    except Exception as e:
        logger.error(f"Error en comando ranking: {e}")
        await update.message.reply_text("❌ Error al obtener el ranking.")

async def mi_ranking(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra el historial personal del usuario."""
    try:
        if not nautilus_db:
            await update.message.reply_text("❌ Sistema de ranking no disponible.")
            return
        
        user_pseudonym = get_user_pseudonym(update.effective_user.id)
        user_best = nautilus_db.get_user_best_score(user_pseudonym)
        user_position = nautilus_db.get_user_ranking_position(user_pseudonym)
        
        if not user_best:
            await update.message.reply_html(
                f"👤 <b>TU PERFIL - {user_pseudonym}</b>\n\n"
                f"🎯 Aún no has completado ninguna sesión.\n"
                f"¡Envía /start para tu primera aventura de percepción remota!"
            )
            return
        
        profile_message = f"""👤 <b>TU PERFIL - {user_pseudonym}</b>

🏆 <b>TU MEJOR PUNTUACIÓN</b>
🔮 <b>{user_best[0]}</b> puntos totales
📍 Posición global: <b>#{user_position}</b>

<b>📊 Desglose de tu mejor sesión:</b>
• Score LLM: <b>{int(user_best[1] * 100)}</b> pts
• Bonus Detalles: <b>📡 {user_best[2]}</b> pts  
• Bonus Tiempo: <b>⏱️ {user_best[3]}</b> pts
• Bonus Calidad: <b>🎯 {user_best[4]}</b> pts

🎯 <b>Objetivo:</b> {user_best[5]} - {user_best[6]}
📅 <b>Fecha:</b> {user_best[7]}

<i>¡Sigue entrenando para superar tu récord!</i>"""

        await update.message.reply_html(profile_message)
        
    except Exception as e:
        logger.error(f"Error en comando mi_ranking: {e}")
        await update.message.reply_text("❌ Error al obtener tu perfil.")

async def estadisticas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra estadísticas del sistema y del usuario."""
    try:
        active_sessions = len(user_sessions)
        user_pseudonym = get_user_pseudonym(update.effective_user.id)
        
        ranking_stats = ""
        if nautilus_db:
            stats = nautilus_db.get_ranking_stats()
            user_best = nautilus_db.get_user_best_score(user_pseudonym)
            user_position = nautilus_db.get_user_ranking_position(user_pseudonym)
            
            if user_best:
                ranking_stats = f"""
<b>🏆 Tu Rendimiento:</b>
• Mejor puntuación: {user_best[0]} pts
• Posición actual: #{user_position}
• Último objetivo: {user_best[6]}

<b>📊 Estadísticas Globales:</b>
• Total sesiones: {stats['total_sessions']}
• Perceptores únicos: {stats['unique_users']}
• Promedio de puntos: {stats['average_points']}
• Récord absoluto: {stats['highest_score']} pts"""
            else:
                ranking_stats = f"""
<b>📊 Estadísticas Globales:</b>
• Total sesiones: {stats['total_sessions']}
• Perceptores únicos: {stats['unique_users']}
• Promedio de puntos: {stats['average_points']}
• Récord absoluto: {stats['highest_score']} pts
• Tu estado: Sin sesiones completadas"""
        
        stats_text = f"""📊 <b>ESTADÍSTICAS DEL SISTEMA</b>

<b>🤖 Estado del Sistema:</b>
• Sesiones activas: {active_sessions}
• Versión: 3.2 (Sistema de Ranking)
• IA Local: {'🟢 Activa' if blip_model else '🔴 Inactiva'}
• IA en la Nube: {'🟢 Activa' if openrouter_client else '🔴 Inactiva'}
• Sistema Ranking: {'🟢 Activo' if nautilus_db else '🔴 Inactivo'}

<b>👤 Tu Perfil:</b>
• Pseudónimo: <code>{user_pseudonym}</code>{ranking_stats}

<b>🔬 Características v3.2:</b>
✅ Sistema de puntuación extendida
✅ Rankings competitivos en tiempo real
✅ Bonificaciones por calidad y detalles
✅ Seguimiento de progreso personal
✅ Búsqueda dinámica de objetivos éticos
✅ Manejo seguro de imágenes con fallbacks"""
        
        await update.message.reply_html(stats_text)
        
    except Exception as e:
        logger.error(f"Error en estadísticas: {e}")
        await update.message.reply_text("❌ Error al generar estadísticas.")

# --- 14. CONFIGURACIÓN DE LA APLICACIÓN ---
def setup_telegram_application() -> Application:
    global telegram_app, nautilus_db
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Inicializar base de datos de rankings
    try:
        nautilus_db = NautilusDB(openrouter_client=openrouter_client)
        logger.info("✅ Sistema de ranking inicializado")
    except Exception as e:
        logger.error(f"❌ Error inicializando sistema de ranking: {e}")
        nautilus_db = None
    
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
    
    # Agregar handlers
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("info", info))
    app.add_handler(CommandHandler("estadisticas", estadisticas))
    app.add_handler(CommandHandler("stats", estadisticas))
    app.add_handler(CommandHandler("ranking", ranking))
    app.add_handler(CommandHandler("mi_ranking", mi_ranking))
    app.add_handler(CommandHandler("perfil", mi_ranking))
    
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
        logger.info("🤖 Protocolo Nautilus v3.2 funcionando...")
        logger.info("🔍 Sistema DuckDuckGo de búsqueda dinámica activo")
        logger.info("📊 Sistema de investigación y datos habilitado")
        logger.info("🏆 Sistema de ranking competitivo activo")
        await server.serve()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

def main():
    """Función principal de ejecución."""
    logger.info("🚀 Iniciando Protocolo Nautilus v3.2 - Sistema de Ranking Competitivo")
    logger.info("🔬 Sistema de percepción remota con búsqueda dinámica")
    logger.info("🛡️ Protocolos de seguridad ética implementados")
    logger.info("📊 Sistema de investigación y pseudónimos activado")
    logger.info("🏆 Sistema de puntuación extendida y rankings")
    
    try:
        asyncio.run(run_services())
    except KeyboardInterrupt:
        logger.info("👋 Protocolo Nautilus cerrado por el usuario.")
    except Exception as e:
        logger.error(f"💥 Error fatal en main: {e}", exc_info=True)

if __name__ == "__main__":
    main()