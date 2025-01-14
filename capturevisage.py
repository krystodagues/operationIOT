import cv2
import os
import time
from mtcnn import MTCNN

# Dossier pour enregistrer les images de visage
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)  # Crée le dossier si nécessaire

# Initialiser la webcam
cap = cv2.VideoCapture(0)

# Vérifier si la webcam est disponible
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
    exit()

# Initialiser le détecteur MTCNN
detector = MTCNN()

# Compte à rebours avant la capture
for i in range(3, 0, -1):
    print(f"Capture dans {i}...")
    time.sleep(1)

print("Début de la capture des visages !")

# Capturer 100 images de visage
num_images = 100
captured_faces = 0

while captured_faces < num_images:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la capture de l'image.")
        break

    # Convertir l'image en RGB (nécessaire pour MTCNN)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des visages avec MTCNN
    results = detector.detect_faces(rgb_frame)

    for result in results:
        # Récupérer les coordonnées du visage détecté
        x, y, w, h = result['box']

        # Assurer que les dimensions sont valides
        x, y = max(0, x), max(0, y)
        face = frame[y:y+h, x:x+w]

        # Sauvegarder l'image du visage dans le dossier dataset
        image_path = os.path.join(output_dir, f"face_{captured_faces + 1:03d}.jpg")
        cv2.imwrite(image_path, face)  # Enregistrer le visage
        captured_faces += 1

        # Afficher l'image du visage capturé
        cv2.imshow("Face Capture", face)

        # Quitter la boucle si suffisamment de visages ont été capturés
        if captured_faces >= num_images:
            break

    # Afficher le flux vidéo avec des rectangles autour des visages détectés
    for result in results:
        x, y, w, h = result['box']
        x, y = max(0, x), max(0, y)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Webcam", frame)

    # Attendre 100 ms et vérifier si l'utilisateur appuie sur 'q'
    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("Capture interrompue par l'utilisateur.")
        break

print(f"Capture terminée. Les images sont sauvegardées dans le dossier '{output_dir}'.")

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
