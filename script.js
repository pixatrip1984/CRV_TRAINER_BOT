document.addEventListener("DOMContentLoaded", () => {
    if (window.Telegram && window.Telegram.WebApp) {
        const tg = window.Telegram.WebApp;
        
        tg.ready();
        tg.expand();

        // --- CONFIGURACIÓN DEL BOTÓN PRINCIPAL OFICIAL DE TELEGRAM ---
        tg.MainButton.setText("Enviar Dibujo ✅");
        tg.MainButton.enable();
        tg.MainButton.show();
        // -------------------------------------------------------------

        const canvas = document.getElementById('drawing-canvas');
        const ctx = canvas.getContext('2d');
        const clearButton = document.getElementById('clear-button');

        // --- CÓDIGO DE DIBUJO (sin cambios) ---
        function resizeCanvas() {
            const container = document.getElementById('canvas-container');
            canvas.width = container.clientWidth * 0.95;
            canvas.height = container.clientHeight * 0.95;
            ctx.strokeStyle = "#000"; ctx.lineWidth = 3; ctx.lineCap = "round";
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
        let drawing = false;
        function startDrawing(e) { drawing = true; draw(e); }
        function stopDrawing() { drawing = false; ctx.beginPath(); }
        function draw(e) {
            if (!drawing) return; e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX || e.touches[0].clientX) - rect.left;
            const y = (e.clientY || e.touches[0].clientY) - rect.top;
            ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y);
        }
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchend', stopDrawing);
        canvas.addEventListener('touchmove', draw);
        clearButton.addEventListener('click', () => { ctx.clearRect(0, 0, canvas.width, canvas.height); });

        // --- NUEVO MANEJADOR DE EVENTO PARA EL BOTÓN PRINCIPAL ---
        tg.onEvent('mainButtonClicked', function() {
            try {
                const dataUrl = canvas.toDataURL();
                tg.sendData(dataUrl);
            } catch (error) {
                tg.showAlert(`Error al enviar: ${error.message}`);
            }
        });

    } else {
        document.body.innerHTML = "<h1>Error</h1><p>Esta página debe ser abierta como una Mini App dentro de Telegram.</p>";
    }
});