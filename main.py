from ultralytics import YOLO
import cv2
import numpy as np

# Implementamos el modelo YOLOv8 para detección de objetos

model = YOLO("yolov8n.pt")


# Definimos una variable que guardara la URL de la camara virtual.
camera_URL = "http://192.168.1.160:8080/video"
# Se usa la APP IP WEBCAM, cada vez que se utilice este codigo debe de actualizarse a la ip actual

#Empleamos la camara que utilizaremos, en caso de que estuviera integrada al sistema aqui se cambiara
cap = cv2.VideoCapture(camera_URL)

# En caso de haber error
if not cap.isOpened():
    print("No se pudo conectar con la cámara del teléfono")
    exit()

print("VITABOT AI iniciado")
print("Presiona Q para salir")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error al recibir imagen")
        break

        # Funcionamiento de YOLO para la deteccion

    results = model(frame)

    annotated_frame = frame.copy()

    frame_height, frame_width = frame.shape[:2]

    for r in results:

        boxes = r.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])

            label = model.names[cls]

            confidence = float(box.conf[0])


            # Estimacion de profundidad basada en el tamaño del objeto detectado
         

            object_width = x2 - x1
            object_height = y2 - y1

            object_area = object_width * object_height

            # Relación aproximada
            proximity_score = object_area / (frame_width * frame_height)

            # Clasificación de distancia
            if proximity_score > 0.25:
                distance_label = "MUY CERCA"
            elif proximity_score > 0.10:
                distance_label = "CERCA"
            elif proximity_score > 0.03:
                distance_label = "MEDIA DISTANCIA"
            else:
                distance_label = "LEJOS"

            # Color segun distancia
            if distance_label == "MUY CERCA":
                color = (0, 0, 255)

            elif distance_label == "CERCA":
                color = (0, 165, 255)

            elif distance_label == "MEDIA DISTANCIA":
                color = (0, 255, 255)

            else:
                color = (0, 255, 0)

            # Dibujar rectangulo
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Texto
            text = f"{label} | {distance_label}"

            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


    # Resultados en pantalla
   

    cv2.imshow("VITABOT AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()