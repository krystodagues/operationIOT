import cv2
import torch
import numpy as np
from torchvision import transforms
import tkinter as tk
from PIL import Image, ImageTk

# Définition du modèle MobileFaceNet
class MobileFaceNet(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(MobileFaceNet, self).__init__()
        # Couches de convolution
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
        )
        
        # Couches fully connected
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

# Chargement du modèle MobileFaceNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileFaceNet().to(device)

# Charger les poids avec strict=False pour ignorer les clés inattendues
try:
    model.load_state_dict(torch.load('mobilefacenet3.pth'), strict=False)
    print("Poids chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement des poids : {e}")

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
    face = detect_face(frame)
    if len(face) > 0:
        x, y, w, h = face[0]
        face_image = frame[y:y+h, x:x+w]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
        face_image = transform(face_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(face_image)
        
        # 0 - inconnu, 1 - connu
        _, predicted = output.max(1)
        return predicted.item()  # Retourner la classe détectée (0 ou 1)
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
