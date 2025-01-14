import cv2
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from torchvision import transforms

# Chargement du modèle ONNX
onnx_model_path = 'mobilefacenet.onnx'
session = ort.InferenceSession(onnx_model_path)

# Chargement du classificateur de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuration des transformations pour l'image
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Fonction pour vérifier si un visage est détecté
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Fonction pour prédire si c'est un visage connu ou inconnu
def predict_face(frame):
    faces = detect_face(frame)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_image = frame[y:y+h, x:x+w]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = transform(face_image).unsqueeze(0).numpy()

        # Utilisation du modèle ONNX
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: face_image})[0]

        # 0 - inconnu, 1 - connu
        predicted_class = np.argmax(output)
        return predicted_class  # Retourner la classe détectée (0 ou 1)
    
    return -1  # Aucun visage détecté

# Fonction pour mettre à jour la fenêtre tkinter avec la webcam
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Convertir la couleur de l'image BGR en RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Détecter si un visage est détecté et effectuer une prédiction
    class_id = predict_face(frame)
    if class_id == 1:  # Visage connu
        button.config(bg="green")
    elif class_id == 0:  # Visage inconnu
        button.config(bg="red")
    else:
        button.config(bg="yellow")  # Aucun visage détecté

    # Mettre à jour l'image dans le label tkinter
    label.img_tk = img_tk
    label.config(image=img_tk)
    root.after(10, update_frame)

# Configuration de la fenêtre Tkinter
root = tk.Tk()
root.title("Détection de Visage")

# Label pour afficher l'image de la webcam
label = tk.Label(root)
label.pack()

# Bouton pour afficher si un visage est détecté
button = tk.Button(root, text="Visage Détecté", bg="yellow", font=("Arial", 14), width=20, height=2)
button.pack()

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)

# Appeler la fonction pour mettre à jour la fenêtre
update_frame()

# Démarrer la fenêtre tkinter
root.mainloop()

# Libérer la webcam à la fin
cap.release()
cv2.destroyAllWindows()
