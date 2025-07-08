import os
import time
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capturas')
def capturas():
    pasta = os.path.join("static", "capturas")
    videos = [f for f in os.listdir(pasta) if f.endswith(".webm")]
    videos.sort(reverse=True)
    return render_template("capturas.html", imagens=videos)

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_video(filename):
    caminho = os.path.join("static", "capturas", filename)
    try:
        if os.path.exists(caminho):
            os.remove(caminho)
            return jsonify({'success': True}), 200
        else:
            return jsonify({'error': 'Ficheiro não encontrado'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def gen_frames():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] A usar dispositivo: {device.upper()}")
    model = YOLO("yolo11n.pt").to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # WebM
    out = None
    gravando = False
    tempo_sem_objetos = 0
    tempo_maximo_sem_obj = 2  # segundos
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    previous_boxes_by_type = {}

    # Variáveis para controle de alertas
    last_alert_time = time.time()  # Tempo da última vez que o alerta foi enviado
    alert_interval = 3  # Intervalo de 3 segundos entre os alertas
    alert_sent = set()  # Usando um conjunto para armazenar os tipos de objetos já enviados

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Processamento com a resolução original para manter a velocidade do vídeo
        frame = cv2.resize(frame, (640, 360))  # Reduz a resolução para melhorar a performance sem alterar a taxa de FPS

        # Deteção com GPU se disponível
        results = model.predict(frame, device=device, stream=False, verbose=False)[0]
        current_time = time.time()
        detected = False
        detected_types = set()

        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            cls = int(cls_id)
            confidence = float(conf) * 100
            x1, y1, x2, y2 = map(int, box)
            nova_caixa = np.array([x1, y1, x2, y2])

            tipo, color = None, (0, 255, 0)
            if cls == 0:
                tipo = "Pessoa"; color = (0, 255, 0)
            elif cls == 2:
                tipo = "Carro"; color = (255, 0, 0)

            if tipo:
                label = f"{tipo} {confidence:.0f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                caixas_anteriores = previous_boxes_by_type.get(tipo, [])
                novo_objeto = all(np.linalg.norm(nova_caixa - np.array(c)) > 60 for c in caixas_anteriores)

                if novo_objeto:
                    detected_types.add(tipo)
                    previous_boxes_by_type.setdefault(tipo, []).append([x1, y1, x2, y2])

                detected = True

        if detected:
            if not gravando:
                nome_video = f"captura_{int(current_time)}.webm"
                caminho_video = os.path.join("static", "capturas", nome_video)
                os.makedirs("static/capturas", exist_ok=True)
                out = cv2.VideoWriter(caminho_video, fourcc, fps, (frame_largura, frame_altura))
                print(f"[INFO] Iniciada gravação: {nome_video}")
                gravando = True
            tempo_sem_objetos = 0

            # Enviar alerta a cada intervalo de tempo (3 segundos)
            for tipo in detected_types:
                if tipo not in alert_sent or current_time - last_alert_time >= alert_interval:
                    socketio.emit('alert', {'message': f"{tipo} detetado!"})
                    alert_sent.add(tipo)  # Marca esse tipo como já enviado
                    last_alert_time = current_time  # Atualiza o tempo do último alerta
        else:
            if gravando:
                tempo_sem_objetos += 1
                if tempo_sem_objetos >= fps * tempo_maximo_sem_obj:
                    out.release()
                    print("[INFO] Gravação terminada.")
                    gravando = False
                    tempo_sem_objetos = 0

            # Resetando os alertas quando não há objetos detectados
            alert_sent.clear()

        if gravando and out:
            out.write(frame)

        # Compressão leve para envio rápido
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # Aumentar um pouco a qualidade para melhorar a imagem
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')