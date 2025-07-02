document.addEventListener("DOMContentLoaded", () => {
    if (window.Telegram && window.Telegram.WebApp) {
        const tg = window.Telegram.WebApp;
        
        tg.ready();
        tg.expand();
        tg.MainButton.setText("Enviar Dibujo ✅");
        tg.MainButton.enable();
        tg.MainButton.show();

        const canvas = document.getElementById('drawing-canvas');
        const ctx = canvas.getContext('2d');
        const clearButton = document.getElementById('clear-button');

        function resizeCanvas() {
            const container = document.getElementById('canvas-container');
            canvas.width = container.clientWidth * 0.95;
            canvas.height = container.clientHeight * 0.95;
            ctx.strokeStyle = "#000";
            ctx.lineWidth = 3;
            ctx.lineCap = "round";
        }

        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        let drawing = false;

        function startDrawing(e) { 
            drawing = true; 
            draw(e); 
        }

        function stopDrawing() { 
            drawing = false; 
            ctx.beginPath();
        }

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
        
        clearButton.addEventListener('click', () => { 
            ctx.clearRect(0, 0, canvas.width, canvas.height); 
        });

        function getUserId() {
            try {
                if (tg.initDataUnsafe && tg.initDataUnsafe.user && tg.initDataUnsafe.user.id) {
                    return tg.initDataUnsafe.user.id;
                }
                return null;
            } catch (error) {
                console.error('Error obteniendo User ID:', error);
                return null;
            }
        }

        // 🔒 VERSIÓN SEGURA: Usa sendData() pero con protocolo mejorado
        async function sendDrawingSecure() {
            try {
                const userId = getUserId();
                if (!userId) {
                    tg.showAlert("❌ Error: No se pudo obtener el ID de usuario");
                    return;
                }

                tg.MainButton.setText("Enviando... ⏳");
                tg.MainButton.disable();

                // Convertir canvas a base64 comprimido
                const dataUrl = canvas.toDataURL('image/jpeg', 0.7); // JPEG con 70% calidad
                
                // Crear objeto de datos más pequeño
                const payload = {
                    type: 'nautilus_drawing',
                    userId: userId,
                    imageData: dataUrl,
                    timestamp: Date.now()
                };

                // Enviar via sendData() con protocolo optimizado
                tg.sendData(JSON.stringify(payload));
                
            } catch (error) {
                console.error('Error:', error);
                tg.MainButton.setText("Enviar Dibujo ✅");
                tg.MainButton.enable();
                tg.showAlert(`❌ Error: ${error.message}`);
            }
        }
        
        tg.onEvent('mainButtonClicked', sendDrawingSecure);

    } else {
        console.error("API de Telegram WebApp no encontrada.");
        document.body.innerHTML = "<h1>Error</h1><p>Esta aplicación solo funciona dentro de Telegram.</p>";
    }
});