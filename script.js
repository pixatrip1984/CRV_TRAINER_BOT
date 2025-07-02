document.addEventListener("DOMContentLoaded", () => {
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

        // --- MÉTODO DE ENVÍO DE DATOS ---

        // Este es el manejador oficial para el botón principal de Telegram
        tg.onEvent('mainButtonClicked', function() {
            try {
                // Convertimos el contenido del lienzo a una cadena de texto en formato Data URL (base64)
                const dataUrl = canvas.toDataURL();
                // Usamos la función de la API para enviar los datos de vuelta a nuestro bot
                tg.sendData(dataUrl);
            } catch (error) {
                // Si algo falla, usamos la alerta nativa de Telegram para informar al usuario
                tg.showAlert(`Error al enviar el dibujo: ${error.message}`);
            }
        });

    } else {
        // Mensaje de error si la página se abre fuera de Telegram
        console.error("API de Telegram WebApp no encontrada. Esta página debe abrirse dentro de Telegram.");
        document.body.innerHTML = "<h1>Error</h1><p>Esta aplicación solo funciona dentro de Telegram.</p>";
    }
});