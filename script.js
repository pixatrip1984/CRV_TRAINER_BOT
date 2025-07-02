document.addEventListener("DOMContentLoaded", () => {
    // 1. VERIFICAR SI EL OBJETO TELEGRAM EXISTE
    if (window.Telegram && window.Telegram.WebApp) {
        console.log("Objeto Telegram.WebApp encontrado. Inicializando...");
        
        const tg = window.Telegram.WebApp;

        // 2. LA SOLUCIÓN CLAVE: ESPERAR A QUE TELEGRAM ESTÉ LISTO
        tg.ready();
        
        // Expande la Mini App para usar más pantalla
        tg.expand();

        const canvas = document.getElementById('drawing-canvas');
        const ctx = canvas.getContext('2d');
        const sendButton = document.getElementById('send-button');
        const clearButton = document.getElementById('clear-button');

        function resizeCanvas() {
            const container = document.getElementById('canvas-container');
            canvas.width = container.clientWidth * 0.95;
            canvas.height = container.clientHeight * 0.95;
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 3;
            ctx.lineCap = "round";
            console.log("Lienzo redimensionado:", canvas.width, canvas.height);
        }

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        let drawing = false;

        function startDrawing(e) { drawing = true; draw(e); }
        function stopDrawing() { drawing = false; ctx.beginPath(); }
        function draw(e) {
            if (!drawing) return;
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const y = (e.clientY || e.touches[0].clientY) - rect.top;
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchend', stopDrawing);
        canvas.addEventListener('touchmove', draw);

        // --- MANEJADOR DEL BOTÓN DE ENVIAR (CON DEPURACIÓN) ---
        sendButton.addEventListener('click', () => {
            console.log("Botón 'Enviar Dibujo' presionado.");
            try {
                // Comprobamos de nuevo si tg está disponible
                if (tg) {
                    const dataUrl = canvas.toDataURL(); // Imagen en formato base64
                    console.log("Imagen convertida a Data URL (los primeros 100 caracteres):", dataUrl.substring(0, 100));
                    
                    // 3. ENVIAR LOS DATOS
                    tg.sendData(dataUrl); 
                    console.log("tg.sendData() ha sido llamado. La Mini App debería cerrarse si todo fue bien.");
                    
                    // Nota: tg.close() se llama automáticamente después de sendData si la comunicación es exitosa.
                } else {
                    console.error("El objeto tg no estaba disponible al hacer clic en enviar.");
                    alert("Error: No se pudo comunicar con Telegram.");
                }
            } catch (error) {
                console.error("Ocurrió un error al enviar los datos:", error);
                alert(`Error: ${error.message}`);
            }
        });

        clearButton.addEventListener('click', () => {
            console.log("Botón 'Borrar Todo' presionado.");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        });

    } else {
        // Mensaje para cuando se abre fuera de Telegram (útil para depurar)
        console.error("Telegram WebApp API no encontrada. Asegúrate de abrir esto dentro de Telegram.");
        document.body.innerHTML = "<h1>Error</h1><p>Esta página debe ser abierta como una Mini App dentro de Telegram.</p>";
    }
});
