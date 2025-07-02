document.addEventListener("DOMContentLoaded", () => {
    const BOT_TOKEN = "7944810548:AAHiwicHirgwdD0Cm0QmPEAOVB7VGo1A3H0"; // ← Pon tu token nuevo aquí
    
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

        async function sendDrawingViaTelegram() {
            try {
                const userId = getUserId();
                if (!userId) {
                    tg.showAlert("❌ Error: No se pudo obtener el ID de usuario");
                    return;
                }

                tg.MainButton.setText("Enviando... ⏳");
                tg.MainButton.disable();

                canvas.toBlob(async (blob) => {
                    try {
                        const formData = new FormData();
                        formData.append('photo', blob, `nautilus_drawing_${userId}_${Date.now()}.png`);
                        formData.append('chat_id', userId);
                        formData.append('caption', `🎨 Dibujo Fase 3 - Usuario ${userId}\n#NautilusDrawing`);

                        const response = await fetch(`https://api.telegram.org/bot${BOT_TOKEN}/sendPhoto`, {
                            method: 'POST',
                            body: formData
                        });

                        const result = await response.json();

                        if (result.ok) {
                            tg.showAlert("✅ Dibujo enviado correctamente", () => {
                                tg.close();
                            });
                        } else {
                            throw new Error(result.description || 'Error de Telegram API');
                        }
                    } catch (error) {
                        console.error('Error:', error);
                        tg.MainButton.setText("Enviar Dibujo ✅");
                        tg.MainButton.enable();
                        tg.showAlert(`❌ Error: ${error.message}`);
                    }
                }, 'image/png', 0.9);

            } catch (error) {
                console.error('Error:', error);
                tg.MainButton.setText("Enviar Dibujo ✅");
                tg.MainButton.enable();
                tg.showAlert(`❌ Error: ${error.message}`);
            }
        }
        
        tg.onEvent('mainButtonClicked', sendDrawingViaTelegram);

    } else {
        console.error("API de Telegram WebApp no encontrada.");
        document.body.innerHTML = "<h1>Error</h1><p>Esta aplicación solo funciona dentro de Telegram.</p>";
    }
});