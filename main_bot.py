# main_bot.py

# ==============================================================================
#                      PROTOCOLO NAUTILUS - TELEGRAM BOT v3.1
#                    (DuckDuckGo Real + UX Optimizada)
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

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

if not TELEGRAM_TOKEN: raise ValueError("TELEGRAM_TOKEN no encontrado.")

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
    # Usar hash del user_id para generar pseudónimo consistente
    hash_input = f"nautilus_{user_id}_salt_2025"
    hash_digest = hashlib.md5(hash_input.encode()).hexdigest()
    
    # Listas de nombres científicos/místicos
    prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Theta", "Lambda", "Sigma", "Omega", 
                "Nova", "Quasar", "Pulsar", "Nebula", "Cosmos", "Astral", "Stellar", "Lunar", "Solar", "Vortex"]
    suffixes = ["Explorer", "Seeker", "Voyager", "Navigator", "Observer", "Perceiver", "Sensor", "Scanner", 
                "Detector", "Finder", "Hunter", "Tracker", "Reader", "Viewer", "Seer", "Oracle", "Mystic"]
    
    # Usar el hash para seleccionar consistentemente
    prefix_idx = int(hash_digest[:2], 16) % len(prefixes)
    suffix_idx = int(hash_digest[2:4], 16) % len(suffixes)
    number = int(hash_digest[4:6], 16) % 1000
    
    return f"{prefixes[prefix_idx]}{suffixes[suffix_idx]}{number:03d}"

def save_session_data(user_id: int, session_data: dict, score: float):
    """Guarda los datos de la sesión para research futuro."""
    try:
        # Cargar datos existentes
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"sessions": [], "high_score_targets": []}
        
        # Agregar nueva sesión
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
        
        # Si el puntaje es alto (>7.0), agregar al pool de objetivos exitosos
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
                # Actualizar promedio
                for target in data["high_score_targets"]:
                    if target["url"] == session_record["target_url"]:
                        target["success_count"] += 1
                        target["average_score"] = (target["average_score"] + score) / 2
                        break
        
        # Guardar datos
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Datos de sesión guardados para {get_user_pseudonym(user_id)}")
        
    except Exception as e:
        logger.error(f"Error guardando datos de sesión: {e}")

# --- 5. GENERACIÓN INTELIGENTE DE TÉRMINOS CON LLM ---

async def generate_search_term_with_llm() -> str:
    """
    Usa Mistral para generar términos de búsqueda éticos y seguros de manera inteligente.
    """
    if not openrouter_client:
        # Fallback si no hay cliente disponible
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

**EJEMPLOS DE BUENOS TÉRMINOS:**
- "ancient stone lighthouse peaceful coast"
- "serene mountain temple garden"
- "historic marble fountain courtyard"
- "tranquil forest waterfall scene"
- "classical observatory dome architecture"

**FORMATO DE RESPUESTA:**
Responde ÚNICAMENTE con el término de búsqueda, sin explicaciones adicionales.

Genera ahora un término único y seguro:"""

        response = await asyncio.to_thread(
            openrouter_client.chat.completions.create,
            model=MISTRAL_CLOUD_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,  # Alta creatividad para variedad
            max_tokens=50
        )
        
        search_term = response.choices[0].message.content.strip()
        
        # Validación básica del término generado
        forbidden_words = [
            'person', 'people', 'human', 'man', 'woman', 'child', 'face', 'body',
            'war', 'battle', 'death', 'grave', 'cemetery', 'disaster', 'accident',
            'violence', 'blood', 'weapon', 'gun', 'bomb', 'fire', 'destruction'
        ]
        
        # Verificar que no contenga palabras prohibidas
        search_term_lower = search_term.lower()
        if any(word in search_term_lower for word in forbidden_words):
            logger.warning(f"Término generado contiene palabras prohibidas: {search_term}")
            # Usar fallback seguro
            return "peaceful ancient stone temple"
        
        # Verificar longitud razonable
        if len(search_term.split()) > 6:
            logger.warning(f"Término generado muy largo: {search_term}")
            search_term = " ".join(search_term.split()[:5])
        
        logger.info(f"Término de búsqueda generado: '{search_term}'")
        return search_term
        
    except Exception as e:
        logger.error(f"Error generando término con LLM: {e}")
        # Fallback seguro
        return "ancient peaceful stone temple"

async def generate_search_term_with_llm() -> str:
    """
    Usa Mistral para generar términos de búsqueda éticos y seguros de manera inteligente.
    """
    if not openrouter_client:
        # Fallback si no hay cliente disponible
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
            temperature=0.8,  # Alta creatividad para variedad
            max_tokens=50
        )
        
        search_term = response.choices[0].message.content.strip()
        
        # Limpiar comillas y caracteres especiales
        search_term = search_term.replace('"', '').replace("'", '').replace('`', '')
        search_term = search_term.strip()
        
        # Validación básica del término generado
        forbidden_words = [
            'person', 'people', 'human', 'man', 'woman', 'child', 'face', 'body',
            'war', 'battle', 'death', 'grave', 'cemetery', 'disaster', 'accident',
            'violence', 'blood', 'weapon', 'gun', 'bomb', 'fire', 'destruction'
        ]
        
        # Verificar que no contenga palabras prohibidas
        search_term_lower = search_term.lower()
        if any(word in search_term_lower for word in forbidden_words):
            logger.warning(f"Término generado contiene palabras prohibidas: {search_term}")
            # Usar fallback seguro
            return "peaceful ancient stone temple"
        
        # Verificar longitud razonable
        if len(search_term.split()) > 6:
            logger.warning(f"Término generado muy largo: {search_term}")
            search_term = " ".join(search_term.split()[:5])
        
        logger.info(f"Término de búsqueda generado: '{search_term}'")
        return search_term
        
    except Exception as e:
        logger.error(f"Error generando término con LLM: {e}")
        # Fallback seguro
        return "ancient peaceful stone temple"

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
                    # Configuración corregida para la API actual
                    search_params = {
                        "keywords": query,
                        "region": "us-en",  # Región en inglés
                        "safesearch": "On",  # "On" en lugar de "Strict"
                        "size": "Medium",
                        "type_image": None,  # Cualquier tipo
                        "layout": None,      # Cualquier layout
                        "license_image": None,  # Cualquier licencia
                        "max_results": max_results * 3  # Buscar más para filtrar
                    }
                    
                    # Ejecutar búsqueda de imágenes
                    for r in ddgs.images(**search_params):
                        try:
                            image_url = r.get("image")
                            if image_url and validate_image_url_basic(image_url):
                                # Verificar que la URL no contenga términos problemáticos
                                url_lower = image_url.lower()
                                forbidden_url_parts = ['person', 'people', 'face', 'human', 'man', 'woman']
                                if not any(part in url_lower for part in forbidden_url_parts):
                                    results.append(image_url)
                                    logger.debug(f"URL válida encontrada: {image_url[:100]}...")
                                    
                                    if len(results) >= max_results:
                                        break
                            
                            # Pequeña pausa para no sobrecargar
                            time.sleep(0.1)
                            
                        except Exception as e:
                            logger.debug(f"Error procesando resultado individual: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error en búsqueda DDGS detallado: {e}")
                
            return results
        
        # Ejecutar búsqueda en thread separado
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
        
        # Headers más completos
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
        
        # URL con parámetros específicos para imágenes
        search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&t=h_&iax=images&ia=images&safe=strict"
        
        response = await asyncio.to_thread(requests.get, search_url, headers=headers, timeout=20)
        response.raise_for_status()
        
        logger.debug(f"Respuesta HTTP: {response.status_code}, Tamaño: {len(response.content)} bytes")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        image_urls = []
        
        # Estrategia 1: Buscar imágenes directas
        img_tags = soup.find_all('img', src=True)
        for img in img_tags:
            src = img.get('src')
            if src and validate_image_url_basic(src) and 'http' in src:
                if len(image_urls) < max_results:
                    image_urls.append(src)
                    logger.debug(f"Imagen encontrada (directa): {src[:100]}...")
        
        # Estrategia 2: Buscar en atributos data
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
        
        # Estrategia 3: Buscar en JSON embebido
        if len(image_urls) < max_results:
            json_urls = extract_images_from_scripts(soup)
            for url in json_urls:
                if validate_image_url_basic(url):
                    if len(image_urls) < max_results:
                        image_urls.append(url)
                        logger.debug(f"Imagen encontrada (JSON): {url[:100]}...")
        
        # Estrategia 4: Usar términos alternativos si no encontramos nada
        if not image_urls:
            logger.info("No se encontraron imágenes, probando términos alternativos")
            # Simplificar el término de búsqueda
            simple_terms = query.split()[:2]  # Solo las primeras 2 palabras
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
                # Patrones más específicos para encontrar URLs de imágenes
                patterns = [
                    r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif|svg)(?:\?[^\s"\'<>]*)?',
                    r'"image":\s*"(https?://[^\s"\'<>]+)"',
                    r'"url":\s*"(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif))"',
                    r'"src":\s*"(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif))"'
                ]
                
                for pattern in patterns:
                    import re
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
        
        # Mapeo de términos a URLs de Unsplash verificadas
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
        
        # Buscar coincidencias en el query
        query_lower = query.lower()
        for key, url in unsplash_alternatives.items():
            if key in query_lower:
                logger.info(f"Encontrada alternativa de Unsplash para '{key}': {url}")
                return [url]
        
        # Si no hay coincidencia específica, usar una imagen genérica apropiada
        default_urls = list(unsplash_alternatives.values())
        return [random.choice(default_urls)]
        
    except Exception as e:
        logger.error(f"Error en búsqueda alternativa de Unsplash: {e}")
        return []

# Función principal que coordina todas las estrategias de búsqueda
async def search_duckduckgo_images(query: str, max_results: int = 5) -> List[str]:
    """
    Función principal que coordina todas las estrategias de búsqueda de imágenes.
    Intenta múltiples métodos en orden de preferencia.
    """
    logger.info(f"Iniciando búsqueda de imágenes para: '{query}'")
    
    # Estrategia 1: DDGS (método principal)
    results = await search_duckduckgo_images_real(query, max_results)
    if results:
        logger.info(f"✅ DDGS exitoso: {len(results)} imágenes")
        return results
    
    # Estrategia 2: Web scraping (primer fallback)
    logger.info("DDGS falló, intentando web scraping...")
    results = await search_duckduckgo_images_fallback(query, max_results)
    if results:
        logger.info(f"✅ Web scraping exitoso: {len(results)} imágenes")
        return results
    
    # Estrategia 3: Términos simplificados (segundo fallback)
    if len(query.split()) > 2:
        simple_query = " ".join(query.split()[:2])
        logger.info(f"Probando términos simplificados: '{simple_query}'")
        
        results = await search_duckduckgo_images_real(simple_query, max_results)
        if results:
            logger.info(f"✅ Términos simplificados exitoso: {len(results)} imágenes")
            return results
    
    # Estrategia 4: Unsplash alternativo (último recurso)
    logger.info("Todas las búsquedas fallaron, usando Unsplash alternativo...")
    results = await search_unsplash_alternative(query, max_results)
    if results:
        logger.info(f"✅ Unsplash alternativo exitoso: {len(results)} imágenes")
        return results
    
    # Si todo falla, retornar lista vacía (el sistema usará fallback de emergencia)
    logger.warning("Todas las estrategias de búsqueda fallaron")
    return []
    """Validación básica mejorada de URL de imagen."""
    if not url or not isinstance(url, str):
        return False
    
    # Debe empezar con http/https
    if not url.startswith(('http://', 'https://')):
        return False
    
    # Verificar extensiones de imagen o dominios confiables
    url_lower = url.lower()
    
    # Extensiones de imagen válidas
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.svg']
    has_image_ext = any(ext in url_lower for ext in image_extensions)
    
    # Dominios confiables para imágenes
    trusted_domains = [
        'unsplash.com', 'pixabay.com', 'pexels.com', 'wikimedia.org',
        'flickr.com', 'imgur.com', 'cloudinary.com', 'amazonaws.com'
    ]
    is_trusted_domain = any(domain in url_lower for domain in trusted_domains)
    
    # URLs que NO queremos (contenido inapropiado)
    forbidden_patterns = [
        'profile', 'avatar', 'user', 'people', 'person', 'face',
        'social', 'facebook', 'instagram', 'twitter', 'selfie'
    ]
    has_forbidden = any(pattern in url_lower for pattern in forbidden_patterns)
    
    return (has_image_ext or is_trusted_domain) and not has_forbidden

async def validate_image_content_with_llm(image_url: str) -> bool:
    """
    Valida que el contenido de una imagen sea apropiado usando Mistral Vision.
    """
    if not openrouter_client:
        return True  # Si no hay LLM, asumir que es válida
    
    try:
        # Descargar imagen para análisis
        response = await asyncio.to_thread(requests.get, image_url, timeout=10)
        if response.status_code != 200:
            return False
            
        # Convertir a base64
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
        return False  # En caso de error, rechazar por seguridad

async def select_ethical_target_dynamic() -> Dict[str, str]:
    """
    Selecciona un objetivo ético usando LLM para generar términos de búsqueda.
    """
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            logger.info(f"Intento {attempt + 1} de selección de objetivo")
            
            # 1. Generar término de búsqueda con LLM
            search_term = await generate_search_term_with_llm()
            
            # 2. Buscar imágenes con DuckDuckGo
            image_urls = await search_duckduckgo_images(search_term)
            
            # 3. Validar contenido de imágenes encontradas
            for url in image_urls:
                if await validate_image_content_with_llm(url):
                    # Crear nombre descriptivo del objetivo
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
    
    # Si todos los intentos fallan, usar fallback garantizado
    logger.warning("Todos los intentos fallaron, usando objetivo de emergencia")
    return get_fallback_target()

def get_fallback_target() -> Dict[str, str]:
    """Objetivos de emergencia completamente seguros."""
    emergency_targets = [
        {
            "name": "Ancient Stone Temple",
            "url": "https://images.unsplash.com/photo-1520637836862-4d197d17c93a?w=800&h=600&fit=crop",
            "description": "Templo de piedra ancestral - objetivo de emergencia"
        },
        {
            "name": "Peaceful Mountain Lake",
            "url": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop", 
            "description": "Lago de montaña sereno - objetivo de emergencia"
        },
        {
            "name": "Historic Lighthouse Coast",
            "url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&h=600&fit=crop",
            "description": "Faro histórico costero - objetivo de emergencia"
        }
    ]
    return random.choice(emergency_targets)

# --- 6. FUNCIONES DE FORMATEO ---
def format_analysis_for_telegram(analysis_text: str) -> str:
    """Convierte el análisis en formato limpio para Telegram usando HTML."""
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
        # Convertir texto en negrita
        elif '**' in line:
            line = line.replace('**', '')
            formatted_lines.append(f"<b>{line}</b>")
        # Texto normal
        else:
            formatted_lines.append(line)
    
    # Unir y limpiar
    result = '\n'.join(formatted_lines)
    
    # Limpiar caracteres problemáticos
    result = result.replace('*', '')
    result = result.replace('#', '')
    result = result.replace('`', '')
    result = result.replace('[', '')
    result = result.replace(']', '')
    result = result.replace('(', '')
    result = result.replace(')', '')
    
    return result

def extract_score_from_analysis(analysis_text: str) -> float:
    """Extrae la puntuación numérica del análisis."""
    try:
        # Buscar patrones como "8.5/10", "Puntuación: 7.2", etc.
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
                return min(score, 10.0)  # Máximo 10.0
        
        # Si no se encuentra, devolver puntaje neutro
        return 5.0
        
    except:
        return 5.0

# Modelos Pydantic
class DrawingSubmission(BaseModel):
    imageData: str
    userId: int
    targetCoordinates: Optional[str] = None

class APIResponse(BaseModel):
    status: str
    message: Optional[str] = None

# --- 7. SERVIDOR API (FastAPI) ---
app_fastapi = FastAPI(title="Protocolo Nautilus API", version="3.1.0")
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

# --- 8. FUNCIONES DE IA ESPECIALIZADAS ---
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

# --- 9. HANDLERS DE TELEGRAM ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    user_sessions[user.id] = {"chat_id": update.effective_chat.id, "session_data": {}}
    
    # Mensaje inicial del protocolo (usuario lee mientras buscamos objetivo)
    protocol_message = f"""🧠 <b>PROTOCOLO NAUTILUS v3.1</b>
<i>Sistema de Percepción Remota Controlada</i>

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

<b>📊 INVESTIGACIÓN</b>
Tus datos (anónimos) contribuyen a la investigación sobre percepción remota. Tu pseudónimo: <code>{get_user_pseudonym(user.id)}</code>

<b>⏳ PREPARANDO OBJETIVO...</b>
<i>Buscando objetivo ético usando DuckDuckGo...</i>"""

    # Enviar mensaje del protocolo
    sent_message = await update.message.reply_html(protocol_message)
    
    # Buscar objetivo en paralelo mientras el usuario lee
    try:
        # Generar coordenadas profesionales
        target_ref = generate_professional_coordinates()
        
        # Buscar objetivo ético dinámicamente
        selected_target = await select_ethical_target_dynamic()
        
        # Guardar datos de la sesión
        user_sessions[user.id]["session_data"]["target"] = selected_target
        user_sessions[user.id]["session_data"]["target_ref"] = target_ref
        
        logger.info(f"Usuario {user.id} ({user.first_name}) inició sesión. Objetivo: {selected_target['name']}, Coordenadas: {target_ref}")
        
        # Crear nuevo mensaje con las coordenadas (en lugar de editar)
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
        # Fallback si falla la búsqueda
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
    
    # Extraer puntuación para research
    score = extract_score_from_analysis(session_analysis)
    
    # Guardar datos de la sesión para investigación
    save_session_data(user_id, session_data, score)
    
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
    
    # Enviar análisis profesional con formato limpio
    if "Error:" in session_analysis:
        await context.bot.send_message(
            chat_id=user_id, 
            text=f"⚠️ <b>No se pudo generar el análisis profesional</b>\n\n{session_analysis}", 
            parse_mode='HTML'
        )
    else:
        try:
            # Formatear el análisis para Telegram
            formatted_analysis = format_analysis_for_telegram(session_analysis)
            
            # Dividir en partes si es necesario
            if len(formatted_analysis) > 4000:
                parts = [formatted_analysis[i:i+4000] for i in range(0, len(formatted_analysis), 4000)]
                for i, part in enumerate(parts):
                    header = f"📊 <b>ANÁLISIS PROFESIONAL - Parte {i+1}/{len(parts)}</b>\n\n" if i == 0 else ""
                    await context.bot.send_message(
                        chat_id=user_id, 
                        text=header + part, 
                        parse_mode='HTML'
                    )
                    await asyncio.sleep(1)
            else:
                await context.bot.send_message(
                    chat_id=user_id, 
                    text=f"📊 <b>ANÁLISIS PROFESIONAL</b>\n\n{formatted_analysis}", 
                    parse_mode='HTML'
                )
        except Exception as e:
            logger.error(f"Error enviando análisis formateado: {e}")
            # Enviar versión básica si falla el formateo
            basic_text = session_analysis.replace('#', '').replace('*', '').replace('`', '')
            await context.bot.send_message(
                chat_id=user_id, 
                text=f"📊 <b>ANÁLISIS PROFESIONAL</b>\n\n{basic_text}", 
                parse_mode='HTML'
            )
    
    # Mensaje de cierre con información de investigación
    pseudonym = get_user_pseudonym(user_id)
    await update.message.reply_html(
        f"🙏 <b>Sesión Completada</b>\n\n"
        f"Gracias por participar, <b>{pseudonym}</b>!\n"
        f"Tu puntuación: <b>{score:.1f}/10.0</b>\n\n"
        f"<b>📊 Contribución a la Investigación:</b>\n"
        f"• Datos guardados de forma anónima\n"
        f"• Ayudas a identificar patrones en percepción remota\n"
        f"• Objetivos exitosos se agregan al pool de investigación\n\n"
        f"<i>Cada sesión nos acerca más a entender este fenómeno.</i>\n\n"
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

# --- 10. COMANDOS ADICIONALES ---
async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Proporciona información sobre el protocolo de percepción remota."""
    info_text = """
🧠 <b>PROTOCOLO NAUTILUS - Información Detallada</b>

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

<b>📊 Sistema de Research:</b>
• Cada sesión contribuye a la base de datos científica
• Objetivos exitosos se identifican automáticamente
• Análisis de patrones de percepción remota
"""
    await update.message.reply_html(info_text)

async def estadisticas(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Muestra estadísticas del sistema y del usuario."""
    try:
        # Estadísticas generales
        active_sessions = len(user_sessions)
        user_pseudonym = get_user_pseudonym(update.effective_user.id)
        
        # Cargar datos de investigación si existen
        user_stats = "Sin sesiones previas"
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Estadísticas del usuario
            user_sessions_data = [s for s in data["sessions"] if s["user_pseudonym"] == user_pseudonym]
            if user_sessions_data:
                avg_score = sum(s["score"] for s in user_sessions_data) / len(user_sessions_data)
                best_score = max(s["score"] for s in user_sessions_data)
                total_sessions = len(user_sessions_data)
                user_stats = f"Sesiones: {total_sessions} | Promedio: {avg_score:.1f} | Mejor: {best_score:.1f}"
        
        stats_text = f"""
📊 <b>ESTADÍSTICAS DEL SISTEMA</b>

<b>🤖 Estado del Sistema:</b>
• Sesiones activas: {active_sessions}
• Versión: 3.1 (DuckDuckGo Real)
• IA Local: {'🟢 Activa' if blip_model else '🔴 Inactiva'}
• IA en la Nube: {'🟢 Activa' if openrouter_client else '🔴 Inactiva'}

<b>👤 Tu Perfil:</b>
• Pseudónimo: <code>{user_pseudonym}</code>
• Estadísticas: {user_stats}

<b>🔬 Características:</b>
✅ Búsqueda dinámica de objetivos
✅ Análisis inmediato de bocetos  
✅ Coordenadas profesionales aleatorias
✅ Sistema de investigación integrado
✅ Protocolos de seguridad psicológica
"""
        await update.message.reply_html(stats_text)
        
    except Exception as e:
        logger.error(f"Error en estadísticas: {e}")
        await update.message.reply_text("❌ Error al generar estadísticas.")

# --- 11. CONFIGURACIÓN FINAL DE LA APLICACIÓN ---
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
    app.add_handler(CommandHandler("stats", estadisticas))
    
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
        logger.info("🤖 Protocolo Nautilus v3.1 funcionando...")
        logger.info("🔍 Sistema DuckDuckGo de búsqueda dinámica activo")
        logger.info("📊 Sistema de investigación y datos habilitado")
        await server.serve()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

def main():
    """Función principal de ejecución."""
    logger.info("🚀 Iniciando Protocolo Nautilus v3.1 - DuckDuckGo Real + Research")
    logger.info("🔬 Sistema de percepción remota con búsqueda dinámica")
    logger.info("🛡️ Protocolos de seguridad ética implementados")
    logger.info("📊 Sistema de investigación y pseudónimos activado")
    
    try:
        asyncio.run(run_services())
    except KeyboardInterrupt:
        logger.info("👋 Protocolo Nautilus cerrado por el usuario.")
    except Exception as e:
        logger.error(f"💥 Error fatal en main: {e}", exc_info=True)

if __name__ == "__main__":
    main()