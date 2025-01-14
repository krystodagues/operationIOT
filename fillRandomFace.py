import os
import requests
import random
import cv2
from mtcnn import MTCNN
from PIL import Image
import numpy as np

# Clé API Pexels
PEXELS_API_KEY = "G0SUXeOLJYT6ogvCqh7PgC8idt8pklHLcmDzPXC8IBKHYt350Oxzbm3H"

# Dossier où les images seront enregistrées
output_folder = "dataset/inconnu"
os.makedirs(output_folder, exist_ok=True)

# Initialiser le détecteur MTCNN
detector = MTCNN()

# Fonction pour télécharger une image à partir de Pexels
def download_image(image_url, image_path):
    try:
        # Récupérer l'image à partir de l'URL
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Image téléchargée: {image_path}")
        else:
            print(f"Erreur lors du téléchargement de l'image: {image_url}")
    except Exception as e:
        print(f"Erreur: {e}")

# Fonction pour extraire et enregistrer un visage détecté
def extract_and_save_face(image_path, save_path):
    # Charger l'image
    image = Image.open(image_path)
    image_np = np.array(image)

    # Détecter les visages avec MTCNN
    faces = detector.detect_faces(image_np)

    if faces:
        for i, face in enumerate(faces):
            # Extraire la boîte délimitant le visage
            x, y, w, h = face['box']
            face_image = image_np[y:y+h, x:x+w]

            # Convertir le visage en image PIL pour la sauvegarder
            face_pil = Image.fromarray(face_image)

            # Sauvegarder le visage dans le dossier
            face_pil.save(save_path)
            print(f"Visage extrait et enregistré: {save_path}")
            return True
    else:
        print(f"Aucun visage détecté dans l'image: {image_path}")
        return False

# Fonction pour récupérer des images de visages aléatoires
def get_random_faces_from_pexels(num_images=100):
    url = "https://api.pexels.com/v1/search"
    headers = {
        "Authorization": PEXELS_API_KEY
    }
    
    # Télécharger num_images images de visages
    for i in range(num_images):
        # Rechercher des visages
        params = {
            "query": "face",
            "per_page": 1,
            "page": random.randint(1, 100)  # Choisir une page aléatoire pour les résultats
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["photos"]:
                # Extraire l'URL de la première image
                image_url = data["photos"][0]["src"]["original"]
                image_path = os.path.join(output_folder, f"image_{i + 1}.jpg")
                download_image(image_url, image_path)

                # Extraire et sauvegarder le visage
                face_save_path = os.path.join(output_folder, f"face_{i + 1}.jpg")
                if extract_and_save_face(image_path, face_save_path):
                    os.remove(image_path)  # Supprimer l'image originale si un visage a été extrait
            else:
                print("Aucune image trouvée pour cette page.")
        else:
            print(f"Erreur API Pexels: {response.status_code}")

# Remplir le dossier 'inconnu' avec des images de visages extraites
get_random_faces_from_pexels(num_images=100)
