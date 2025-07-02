document.addEventListener("DOMContentLoaded", () => {
    // --- Referencias a los elementos del DOM ---
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const downloadButton = document.getElementById('download-button');
    const clearButton = document.getElementById('clear-button');
    const controlsDiv = document.getElementById('controls');
    
    // --- Código de dibujo (sin cambios) ---
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

    // --- LA NUEVA LÓGICA DE DESCARGA ---
    downloadButton.addEventListener('click', () => {
        // 1. Obtiene la imagen del lienzo como un Data URL
        const dataUrl = canvas.toDataURL('image/png');
        
        // 2. Oculta el lienzo y los controles originales
        canvas.style.display = 'none';
        controlsDiv.innerHTML = ''; // Borra los botones
        
        // 3. Muestra la imagen final al usuario y las instrucciones
        const finalImage = new Image();
        finalImage.src = dataUrl;
        finalImage.style.maxWidth = '100%';
        finalImage.style.border = '2px solid #0088cc';
        
        const instructions = document.createElement('p');
        instructions.innerHTML = "<b>Paso 1:</b> Guarda esta imagen en tu dispositivo (clic derecho y 'Guardar imagen como...' o mantenla presionada en el móvil).<br><br><b>Paso 2:</b> Cierra esta ventana y adjunta la imagen que guardaste en el chat del bot.";
        instructions.style.textAlign = 'center';
        instructions.style.padding = '20px';
        
        const container = document.getElementById('canvas-container');
        container.innerHTML = ''; // Limpia el contenedor
        container.appendChild(instructions);

        // 4. (Opcional, mejora la UX) Crea un enlace de descarga
        const downloadLink = document.createElement('a');
        downloadLink.href = dataUrl;
        downloadLink.download = 'boceto-nautilus.png';
        downloadLink.appendChild(finalImage);
        
        container.appendChild(downloadLink);
    });
});