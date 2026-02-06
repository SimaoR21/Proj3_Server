# server.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
# os teus ficheiros color.py e util.py devem estar na mesma pasta
from color import get_color_and_shade, get_coloradd_symbol

app = Flask(__name__)

model = YOLO("yolov8m.pt")          # ou yolov8n.pt se quiseres mais rápido

@app.route('/processar', methods=['POST'])
def processar():
    if 'imagem' not in request.files:
        return jsonify({"erro": "sem imagem"}), 400
    
    file = request.files['imagem']
    file_bytes = file.read()
    
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"erro": "imagem inválida"}), 400
    
    # detecção YOLO
    results = model(img, conf=0.5, verbose=False)[0]
    
    dets = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cls = int(box.cls)
        nome = model.names[cls]
        
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        cor, conf_cor = get_color_and_shade(roi)
        if conf_cor < 0.20:
            continue
            
        simbolo = get_coloradd_symbol(cor)
        texto = f"{nome} {int(conf*100)}%  {cor} {simbolo}"
        
        dets.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "texto": texto
        })
    
    return jsonify({
        "status": "ok",
        "deteccoes": dets
    })


if __name__ == '__main__':
    print("Servidor a correr em http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
