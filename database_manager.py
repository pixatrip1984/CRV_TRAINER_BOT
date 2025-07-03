# database_manager.py - VERSIÓN CORREGIDA

import sqlite3
import json
import os
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio
from openai import OpenAI

logger = logging.getLogger(__name__)

class NautilusDB:
    def __init__(self, db_path="nautilus_rankings.db", openrouter_client=None):
        self.db_path = db_path
        self.openrouter_client = openrouter_client
        self.init_database()
        self.migrate_json_data()
    
    def init_database(self):
        """Inicializa la base de datos y crea las tablas necesarias."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Crear tabla principal
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_rankings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_pseudonym TEXT NOT NULL,
                        user_id INTEGER NOT NULL,
                        
                        -- Puntuaciones
                        llm_score REAL NOT NULL,
                        detail_bonus INTEGER NOT NULL DEFAULT 0,
                        time_bonus INTEGER NOT NULL DEFAULT 0,
                        quality_bonus INTEGER NOT NULL DEFAULT 0,
                        total_points INTEGER NOT NULL,
                        
                        -- Información de sesión
                        coordinates TEXT NOT NULL,
                        target_name TEXT NOT NULL,
                        target_url TEXT NOT NULL,
                        sketch_summary TEXT NOT NULL,
                        
                        -- Datos temporales
                        session_duration_minutes INTEGER DEFAULT 0,
                        timestamp_utc TEXT NOT NULL,
                        session_date TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # CORREGIR: Crear índices correctamente
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_total_points ON user_rankings(total_points DESC)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_pseudonym ON user_rankings(user_pseudonym)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_date ON user_rankings(session_date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON user_rankings(timestamp_utc)')
                
                conn.commit()
                logger.info("✅ Base de datos inicializada correctamente")
                
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
    
    def migrate_json_data(self):
        """Migra datos existentes del archivo JSON a la base de datos."""
        json_file = "nautilus_research_data.json"
        if not os.path.exists(json_file):
            return
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Verificar si ya se migraron los datos
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM user_rankings')
                existing_count = cursor.fetchone()[0]
                
                if existing_count > 0:
                    logger.info("Datos ya migrados previamente")
                    return
                
                # Migrar sesiones
                for session in data.get("sessions", []):
                    # Calcular puntos básicos para datos migrados
                    llm_score = session.get("score", 5.0)
                    detail_bonus = self.calculate_detail_bonus_legacy(session.get("phases", {}))
                    time_bonus = 50  # Valor por defecto para datos migrados
                    quality_bonus = 25  # Valor por defecto para datos migrados
                    total_points = int((llm_score * 100) + detail_bonus + time_bonus + quality_bonus)
                    
                    cursor.execute('''
                        INSERT INTO user_rankings (
                            user_pseudonym, user_id, llm_score, detail_bonus, 
                            time_bonus, quality_bonus, total_points,
                            coordinates, target_name, target_url, sketch_summary,
                            session_duration_minutes, timestamp_utc, session_date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session.get("user_pseudonym", "Unknown"),
                        0,  # user_id no disponible en datos antiguos
                        llm_score,
                        detail_bonus,
                        time_bonus, 
                        quality_bonus,
                        total_points,
                        session.get("coordinates", ""),
                        session.get("target_name", ""),
                        session.get("target_url", ""),
                        "Datos migrados del sistema anterior",
                        15,  # Duración estimada
                        session.get("timestamp", datetime.now().isoformat()),
                        session.get("timestamp", datetime.now().isoformat())[:10]
                    ))
                
                conn.commit()
                logger.info(f"✅ Migrados {len(data.get('sessions', []))} registros desde JSON")
                
        except Exception as e:
            logger.error(f"Error migrando datos JSON: {e}")
    
    def calculate_detail_bonus_legacy(self, phases: Dict) -> int:
        """Calcula bonus de detalles para datos migrados."""
        try:
            total_words = 0
            for phase_text in phases.values():
                if isinstance(phase_text, str):
                    total_words += len(phase_text.split())
            return min(200, total_words * 3)
        except:
            return 50
    
    def calculate_detail_bonus(self, fase1: str, fase2: str, fase4: str) -> int:
        """
        Calcula bonus por riqueza de detalles (0-200 puntos).
        Premia descripciones específicas y detalladas.
        """
        try:
            # Contar palabras en cada fase
            words_fase1 = len(fase1.split()) if fase1 else 0
            words_fase2 = len(fase2.split()) if fase2 else 0
            words_fase4 = len(fase4.split()) if fase4 else 0
            
            total_words = words_fase1 + words_fase2 + words_fase4
            
            # Bonus base por cantidad de palabras
            word_bonus = min(150, total_words * 2)
            
            # Bonus extra por especificidad (buscar términos técnicos/específicos)
            specificity_terms = [
                'rugoso', 'suave', 'áspero', 'metálico', 'piedra', 'madera', 'cristal',
                'vertical', 'horizontal', 'circular', 'angular', 'curvo', 'recto',
                'grande', 'pequeño', 'masivo', 'delicado', 'alto', 'bajo', 'ancho',
                'frío', 'cálido', 'húmedo', 'seco', 'denso', 'ligero', 'sólido',
                'antiguo', 'moderno', 'histórico', 'natural', 'artificial', 'ornamental'
            ]
            
            full_text = f"{fase1} {fase2} {fase4}".lower()
            specificity_count = sum(1 for term in specificity_terms if term in full_text)
            specificity_bonus = min(50, specificity_count * 5)
            
            total_bonus = word_bonus + specificity_bonus
            return min(200, total_bonus)
            
        except Exception as e:
            logger.error(f"Error calculando detail bonus: {e}")
            return 50
    
    def calculate_time_bonus(self, session_duration_minutes: int) -> int:
        """
        Calcula bonus por eficiencia temporal (0-100 puntos).
        Premia el tiempo óptimo de concentración.
        """
        try:
            if session_duration_minutes <= 0:
                return 50  # Default para datos sin tiempo
            
            # Tiempo óptimo: 8-20 minutos
            if 8 <= session_duration_minutes <= 20:
                return 100
            # Muy rápido (menos de 8 min): penalización por prisa
            elif session_duration_minutes < 8:
                return max(20, 100 - (8 - session_duration_minutes) * 10)
            # Lento pero aceptable (20-40 min)
            elif 20 < session_duration_minutes <= 40:
                return max(50, 100 - (session_duration_minutes - 20) * 2)
            # Muy lento (más de 40 min): penalización mayor
            else:
                return max(10, 50 - (session_duration_minutes - 40))
                
        except Exception as e:
            logger.error(f"Error calculando time bonus: {e}")
            return 50
    
    async def calculate_quality_bonus_with_llm(self, user_transcript: str) -> int:
        """
        Calcula bonus por calidad usando LLM (0-100 puntos).
        Evalúa la riqueza descriptiva y especificidad.
        """
        if not self.openrouter_client or not user_transcript.strip():
            return 50  # Default si no hay LLM o transcript
        
        try:
            logger.info("Calculando quality bonus con Mistral...")
            
            system_prompt = """Eres un evaluador experto en percepción remota. Evalúa la CALIDAD DESCRIPTIVA del siguiente transcript de sesión."""
            
            user_prompt = f"""Evalúa la calidad descriptiva de este transcript de percepción remota:

---
{user_transcript}
---

CRITERIOS DE EVALUACIÓN:
• **Especificidad:** ¿Las descripciones son específicas o genéricas?
• **Riqueza sensorial:** ¿Incluye múltiples tipos de percepciones (táctil, visual, etc.)?
• **Originalidad:** ¿Hay elementos únicos/inusuales descritos?
• **Coherencia:** ¿Las descripciones son consistentes entre fases?
• **Detalle técnico:** ¿Incluye aspectos estructurales/dimensionales específicos?

ESCALA DE PUNTUACIÓN:
- 90-100: Excepcional riqueza descriptiva, muy específico, múltiples detalles únicos
- 70-89: Buena calidad, específico, varios detalles interesantes  
- 50-69: Calidad promedio, algunos detalles específicos
- 30-49: Básico, descripciones genéricas, pocos detalles
- 10-29: Muy básico, descripciones vagas o mínimas

Responde ÚNICAMENTE con un número del 10 al 100 que represente la puntuación de calidad."""

            response = await asyncio.to_thread(
                self.openrouter_client.chat.completions.create,
                model="mistralai/mistral-small-3.2-24b-instruct:free",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            quality_text = response.choices[0].message.content.strip()
            
            # Extraer número de la respuesta
            quality_score = int(re.search(r'\d+', quality_text).group())
            quality_score = max(10, min(100, quality_score))  # Limitar entre 10-100
            
            logger.info(f"Quality bonus calculado: {quality_score}")
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculando quality bonus con LLM: {e}")
            return 50
    
    async def save_session_to_db(self, user_id: int, user_pseudonym: str, session_data: Dict, 
                                llm_score: float, session_start_time: datetime = None) -> int:
        """
        Guarda una sesión completa en la base de datos con todos los cálculos.
        Retorna los puntos totales obtenidos.
        """
        try:
            # Calcular duración de la sesión
            session_duration = 15  # Default
            if session_start_time:
                duration_delta = datetime.now() - session_start_time
                session_duration = max(1, int(duration_delta.total_seconds() / 60))
            
            # Obtener datos de las fases
            fase1 = session_data.get("fase1", "")
            fase2 = session_data.get("fase2", "")
            fase4 = session_data.get("fase4", "")
            
            # Crear transcript completo para evaluación de calidad
            user_transcript = (
                f"FASE 1 - Impresiones Gestalt:\n{fase1}\n\n"
                f"FASE 2 - Datos Sensoriales:\n{fase2}\n\n"
                f"FASE 4 - Datos Conceptuales:\n{fase4}"
            )
            
            # Calcular todos los bonos
            detail_bonus = self.calculate_detail_bonus(fase1, fase2, fase4)
            time_bonus = self.calculate_time_bonus(session_duration)
            quality_bonus = await self.calculate_quality_bonus_with_llm(user_transcript)
            
            # Calcular puntos totales
            total_points = int((llm_score * 100) + detail_bonus + time_bonus + quality_bonus)
            
            # Obtener información del objetivo
            target_info = session_data.get("target", {})
            target_ref = session_data.get("target_ref", "Unknown")
            sketch_summary = session_data.get("sketch_description", "Sin boceto proporcionado")
            
            # Resumir descripción del boceto si es muy larga
            if len(sketch_summary) > 200:
                sketch_summary = sketch_summary[:197] + "..."
            
            # Preparar datos para inserción
            timestamp_utc = datetime.utcnow().isoformat()
            session_date = timestamp_utc[:10]
            
            # Insertar en la base de datos
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_rankings (
                        user_pseudonym, user_id, llm_score, detail_bonus,
                        time_bonus, quality_bonus, total_points,
                        coordinates, target_name, target_url, sketch_summary,
                        session_duration_minutes, timestamp_utc, session_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_pseudonym,
                    user_id,
                    llm_score,
                    detail_bonus,
                    time_bonus,
                    quality_bonus,
                    total_points,
                    target_ref,
                    target_info.get("name", "Objetivo Desconocido"),
                    target_info.get("url", ""),
                    sketch_summary,
                    session_duration,
                    timestamp_utc,
                    session_date
                ))
                conn.commit()
            
            logger.info(f"✅ Sesión guardada: {user_pseudonym} - {total_points} puntos")
            logger.info(f"Desglose: LLM({llm_score*100}) + Detalles({detail_bonus}) + Tiempo({time_bonus}) + Calidad({quality_bonus})")
            
            return total_points
            
        except Exception as e:
            logger.error(f"Error guardando sesión en DB: {e}")
            return int(llm_score * 100)  # Fallback básico
    
    def get_global_ranking(self, limit: int = 10) -> List[Tuple]:
        """Obtiene el ranking global de mejores puntuaciones."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        user_pseudonym,
                        total_points,
                        llm_score,
                        detail_bonus,
                        time_bonus,
                        quality_bonus,
                        coordinates,
                        target_name,
                        session_date,
                        timestamp_utc
                    FROM user_rankings 
                    ORDER BY total_points DESC, timestamp_utc ASC
                    LIMIT ?
                ''', (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error obteniendo ranking global: {e}")
            return []
    
    def get_user_ranking_position(self, user_pseudonym: str) -> int:
        """Obtiene la posición específica de un usuario en el ranking global."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) + 1 as position
                    FROM user_rankings r1
                    WHERE r1.total_points > (
                        SELECT MAX(r2.total_points) 
                        FROM user_rankings r2 
                        WHERE r2.user_pseudonym = ?
                    )
                ''', (user_pseudonym,))
                result = cursor.fetchone()
                return result[0] if result else 999
        except Exception as e:
            logger.error(f"Error obteniendo posición de usuario: {e}")
            return 999
    
    def get_user_best_score(self, user_pseudonym: str) -> Optional[Tuple]:
        """Obtiene el mejor puntaje de un usuario específico."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        total_points,
                        llm_score,
                        detail_bonus,
                        time_bonus, 
                        quality_bonus,
                        coordinates,
                        target_name,
                        session_date
                    FROM user_rankings 
                    WHERE user_pseudonym = ?
                    ORDER BY total_points DESC
                    LIMIT 1
                ''', (user_pseudonym,))
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error obteniendo mejor puntaje de usuario: {e}")
            return None
    
    def format_ranking_message(self, rankings: List[Tuple], highlight_user: str = None) -> str:
        """
        Formatea el mensaje del ranking con emojis psíquicos.
        """
        if not rankings:
            return "🔮 No hay datos de ranking disponibles."
        
        # Emojis por posición
        position_emojis = {
            1: "🔮",  # Bola de cristal - máximo poder
            2: "👁️",  # Tercer ojo - visión superior  
            3: "🧿",  # Ojo turco - intuición
        }
        default_emoji = "🌟"  # Estrella - energía psíquica
        
        message_lines = ["🔮 <b>RANKING GLOBAL NAUTILUS</b>\n"]
        
        current_points = None
        current_position = 0
        display_position = 0
        
        for i, rank_data in enumerate(rankings):
            (user_pseudonym, total_points, llm_score, detail_bonus, 
             time_bonus, quality_bonus, coordinates, target_name, 
             session_date, timestamp_utc) = rank_data
            
            # Manejar empates
            if total_points != current_points:
                current_position = i + 1
                display_position = current_position
                current_points = total_points
            
            # Seleccionar emoji
            emoji = position_emojis.get(display_position, default_emoji)
            
            # Formatear línea del ranking
            line = f"{emoji} <b>{display_position}.</b> {user_pseudonym} - <b>{total_points}</b> pts"
            
            # Desglose de puntos (solo para top 3)
            if display_position <= 3:
                line += f"\n    📊{int(llm_score*100)} + 📡{detail_bonus} + ⏱️{time_bonus} + 🎯{quality_bonus}"
                line += f"\n    🎯 <i>{coordinates} - {target_name[:30]}{'...' if len(target_name) > 30 else ''}</i>"
            
            # Destacar usuario actual
            if highlight_user and user_pseudonym == highlight_user:
                line += " ⭐ <b>¡TÚ!</b>"
            
            # Indicar empates
            if i > 0 and total_points == rankings[i-1][1]:
                line += " ⚡"
            
            message_lines.append(line)
            message_lines.append("")  # Línea en blanco
        
        # Agregar leyenda
        message_lines.extend([
            "\n📊 <i>Score LLM</i> • 📡 <i>Detalles</i> • ⏱️ <i>Tiempo</i> • 🎯 <i>Calidad</i>",
            "⚡ <i>Empate</i> • ⭐ <i>Tu posición</i>"
        ])
        
        return "\n".join(message_lines)
    
    def get_ranking_stats(self) -> Dict:
        """Obtiene estadísticas generales del ranking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Estadísticas básicas
                cursor.execute('SELECT COUNT(*) FROM user_rankings')
                total_sessions = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT user_pseudonym) FROM user_rankings')
                unique_users = cursor.fetchone()[0]
                
                cursor.execute('SELECT AVG(total_points) FROM user_rankings')
                avg_points = cursor.fetchone()[0] or 0
                
                cursor.execute('SELECT MAX(total_points) FROM user_rankings')
                max_points = cursor.fetchone()[0] or 0
                
                return {
                    "total_sessions": total_sessions,
                    "unique_users": unique_users,
                    "average_points": round(avg_points, 1),
                    "highest_score": max_points
                }
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {
                "total_sessions": 0,
                "unique_users": 0,
                "average_points": 0,
                "highest_score": 0
            }