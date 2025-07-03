document.addEventListener("DOMContentLoaded", () => {
    // ⚠️ IMPORTANTE: Reemplaza esta URL con tu URL de ngrok
    const NGROK_URL = "https://29cb-2a09-bac5-4bbd-2632-00-3ce-8.ngrok-free.app";
    
    // Primero, verificamos que estamos dentro de un entorno de Telegram
    if (window.Telegram && window.Telegram.WebApp) {
        
        const tg = window.Telegram.WebApp;
        
        // --- INICIALIZACIÓN DE LA MINI APP ---
        
        // Le decimos a Telegram que nuestro script está listo. Es importante.
        tg.ready();
        
        // Le pedimos a Telegram que expanda la ventana para tener más espacio.
        tg.expand();

        // --- CONFIGURACIÓN DEL BOTÓN PRINCIPAL OFICIAL DE TELEGRAM ---
        // Le damos un texto, lo activamos y lo mostramos.
        tg.MainButton.setText("Enviar Dibujo ✅");
        tg.MainButton.enable();
        tg.MainButton.show();

        // --- CONFIGURACIÓN DEL LIENZO (CANVAS) ---

        const canvas = document.getElementById('drawing-canvas');
        const ctx = canvas.getContext('2d');
        const clearButton = document.getElementById('clear-button');

        // Función para ajustar el tamaño del lienzo al de la ventana
        function resizeCanvas() {
            const container = document.getElementById('canvas-container');
            canvas.width = container.clientWidth * 0.95;
            canvas.height = container.clientHeight * 0.95;
            ctx.strokeStyle = "#000"; // Color del trazo: negro
            ctx.lineWidth = 3;        // Grosor del trazo
            ctx.lineCap = "round";    // Terminaciones de línea redondeadas
        }

        // Redimensionamos al cargar y si la ventana cambia de tamaño
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        let drawing = false;

        // --- LÓGICA DE DIBUJO (para ratón y pantallas táctiles) ---

        function startDrawing(e) { 
            drawing = true; 
            draw(e); 
        }

        function stopDrawing() { 
            drawing = false; 
            ctx.beginPath(); // Levantamos el "lápiz"
        }

        function draw(e) {
            if (!drawing) return; 
            e.preventDefault(); // Evita que la página haga scroll mientras se dibuja en el móvil
            const rect = canvas.getBoundingClientRect();
            // Calcula las coordenadas correctas para ratón (clientX/Y) y para táctil (touches[0].clientX/Y)
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const y = (e.clientY || e.touches[0].clientY) - rect.top;
            
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        // --- ASIGNACIÓN DE EVENTOS ---

        // Eventos para el ratón
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mousemove', draw);

        // Eventos para pantallas táctiles
        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchend', stopDrawing);
        canvas.addEventListener('touchmove', draw);
        
        // Evento para el botón de borrar
        clearButton.addEventListener('click', () => { 
            ctx.clearRect(0, 0, canvas.width, canvas.height); 
        });

        // --- NUEVO MÉTODO DE ENVÍO DIRECTO VIA HTTP ---

        // Función para obtener el User ID de forma segura
        function getUserId() {
            try {
                // Intentamos obtener el ID del usuario desde initDataUnsafe
                if (tg.initDataUnsafe && tg.initDataUnsafe.user && tg.initDataUnsafe.user.id) {
                    return tg.initDataUnsafe.user.id;
                }
                // Fallback: intentamos desde initData si está disponible
                if (tg.initData) {
                    const urlParams = new URLSearchParams(tg.initData);
                    const userParam = urlParams.get('user');
                    if (userParam) {
                        const userData = JSON.parse(decodeURIComponent(userParam));
                        return userData.id;
                    }
                }
                return null;
            } catch (error) {
                console.error('Error obteniendo User ID:', error);
                return null;
            }
        }

        // Función para enviar el dibujo directamente al servidor
        async function sendDrawingToServer() {
            try {
                // Validamos que ngrok URL esté configurada
                if (NGROK_URL === "REEMPLAZAR_CON_TU_URL_DE_NGROK") {
                    tg.showAlert("❌ Error: URL de ngrok no configurada. Consulta el README.md");
                    return;
                }

                // Convertimos el contenido del lienzo a Data URL
                const imageData = canvas.toDataURL('image/png', 0.8);
                
                // Obtenemos el ID del usuario
                const userId = getUserId();
                if (!userId) {
                    tg.showAlert("❌ Error: No se pudo obtener el ID de usuario");
                    return;
                }

                // Preparamos los datos para enviar
                const payload = {
                    imageData: imageData,
                    userId: userId,
                    timestamp: new Date().toISOString(),
                    canvasSize: {
                        width: canvas.width,
                        height: canvas.height
                    }
                };

                // Mostramos el botón como "enviando"
                tg.MainButton.setText("Enviando... ⏳");
                tg.MainButton.disable();

                console.log('Enviando dibujo al servidor...', NGROK_URL);

                // Realizamos la petición HTTP POST al servidor
                const response = await fetch(`${NGROK_URL}/submit_drawing`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const result = await response.json();
                console.log('Respuesta del servidor:', result);

                // Si todo salió bien, mostramos confirmación y cerramos
                if (result.status === 'ok') {
                    tg.showAlert("✅ Dibujo enviado correctamente", () => {
                        tg.close();
                    });
                } else {
                    throw new Error(result.message || 'Respuesta inesperada del servidor');
                }

            } catch (error) {
                console.error('Error enviando dibujo:', error);
                
                // Restauramos el botón
                tg.MainButton.setText("Enviar Dibujo ✅");
                tg.MainButton.enable();

                // Mostramos error específico dependiendo del tipo
                let errorMessage = "❌ Error enviando el dibujo. ";
                
                if (error.name === 'TypeError' && error.message.includes('fetch')) {
                    errorMessage += "Verifica que ngrok esté funcionando y la URL sea correcta.";
                } else if (error.message.includes('HTTP')) {
                    errorMessage += `Servidor respondió: ${error.message}`;
                } else {
                    errorMessage += `Detalles: ${error.message}`;
                }
                
                tg.showAlert(errorMessage);
            }
        }

        // --- MANEJADOR DEL BOTÓN PRINCIPAL ---
        
        // Este es el manejador oficial para el botón principal de Telegram
        tg.onEvent('mainButtonClicked', sendDrawingToServer);

    } else {
        // Mensaje de error si la página se abre fuera de Telegram
        console.error("API de Telegram WebApp no encontrada. Esta página debe abrirse dentro de Telegram.");
        document.body.innerHTML = "<h1>Error</h1><p>Esta aplicación solo funciona dentro de Telegram.</p>";
    }
});