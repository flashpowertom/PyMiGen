"""Générateur d'image Mistral : script Python principal."""

import sys
import yaml
import requests
import os
from pathlib import Path

# 1. Lire le fichier YAML
def load_yaml_config(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

# 2. Générer une image via une API (exemple avec une API générique)
def generate_image(api_url: str, api_key: str, prompt: str) -> bytes:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "n": 1,
        "size": "1024x1024",
        "response_format": "b64_json"  # ou "url" selon l'API
    }
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()

    # Exemple de traitement pour une réponse en base64
    if "b64_json" in response.json():
        import base64
        image_data = base64.b64decode(response.json()["b64_json"][0])
    else:
        # Si l'API retourne une URL, télécharge l'image
        image_url = response.json()["data"][0]["url"]
        image_data = requests.get(image_url).content
    return image_data

# 3. Sauvegarder l'image sur le disque
def save_image(image_data: bytes, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as file:
        file.write(image_data)

def main() -> int:
    print("Bonjour depuis le Générateur d'image Mistral !")
    print("Arguments reçus :", sys.argv[1:])
    # Chemin vers le fichier YAML
    yaml_file = "config.yml"
    config = load_yaml_config(yaml_file)

    # Récupérer les mots-clés et le chemin de sortie
    keywords = config.get("keywords", "")
    output_path = config.get("output_path", "output.png")

    # Remplacer par les infos de ton API (ex: Mistral, DALL·E, etc.)
    api_url = "https://api.mistral.ai/v1/images/generations"  # À adapter
    api_key = os.getenv("MISTRAL_API_KEY")  # ou une clé en dur (non recommandé)

    # Générer et sauvegarder l'image
    try:
        image_data = generate_image(api_url, api_key, keywords)
        save_image(image_data, output_path)
        print(f"Image générée et sauvegardée sous : {output_path}")
    except Exception as e:
        print(f"Erreur : {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
