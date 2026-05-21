from ultralytics import YOLO
import cv2

# Cargar modelo YOLO
model = YOLO("yolov8n.pt")

# Abrir cámara
cap = cv2.VideoCapture("http://192.168.1.161:4747/video")

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Cámara iniciada correctamente")
print("Presiona Q para salir")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al recibir frame")
        break

    # Detectar objetos
    results = model(frame)

    # Dibujar detecciones
    annotated_frame = results[0].plot()

    # Mostrar ventana
    cv2.imshow("Reconocimiento de Objetos", annotated_frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()