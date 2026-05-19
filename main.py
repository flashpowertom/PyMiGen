"""Générateur d'image avec Hugging Face InferenceClient."""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Charger les variables d'environnement depuis .env
load_dotenv()

# 1. Lire le fichier YAML
def load_yaml_config(file_path: str) -> dict:
    """Charge la configuration depuis un fichier YAML."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        if not config:
            raise ValueError(f"Le fichier {file_path} est vide ou invalide.")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Fichier {file_path} introuvable. Vérifiez le chemin.")
    except yaml.YAMLError as e:
        raise ValueError(f"Erreur de parsing YAML dans {file_path}: {e}")

# 2. Générer une image via Hugging Face InferenceClient
def generate_image_hf(api_key: str, prompt: str, model: str = "black-forest-labs/FLUX.1-dev", provider: str = "wavespeed") -> bytes:
    """
    Génère une image via Hugging Face InferenceClient.
    """
    if not api_key:
        raise ValueError("La clé API Hugging Face est manquante. Définissez HUGGINGFACE_API_KEY dans .env.")

    try:
        # Initialiser le client avec la clé API et le fournisseur
        client = InferenceClient(
            provider=provider,
            api_key=api_key
        )

        # Générer l'image (retourne un objet PIL.Image)
        print("⏳ Génération de l'image en cours (peut prendre 10-60 secondes)...")
        image = client.text_to_image(
            prompt=prompt,
            model=model
        )

        # Convertir l'image PIL en bytes (PNG)
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    except Exception as e:
        raise ValueError(f"Erreur lors de la génération de l'image: {e}")

# 3. Sauvegarder l'image sur le disque
def save_image(image_data: bytes, output_path: str) -> str:
    """
    Sauvegarde les données binaires d'une image dans un fichier.
    """
    path = Path(output_path)
    if path.is_absolute():
        path = path.resolve()
    else:
        path = Path.cwd() / path

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as file:
        file.write(image_data)

    return str(path)

def main() -> int:
    """Point d'entrée principal du script."""
    print("🚀 Générateur d'image Hugging Face (InferenceClient) - Démarrage...")

    # Chemin vers le fichier YAML (relatif au script)
    script_dir = Path(__file__).parent
    yaml_file = script_dir / "config.yml"

    try:
        # Charger la configuration
        config = load_yaml_config(str(yaml_file))
        print(f"✅ Configuration chargée depuis {yaml_file}")

        # Récupérer les paramètres
        prompt = config.get("keywords", "").strip()
        model = config.get("model", "black-forest-labs/FLUX.1-dev")
        provider = config.get("provider", "wavespeed")
        output_path = config.get("output_path", "output/output.png")

        # Vérifier que le prompt n'est pas vide
        if not prompt:
            raise ValueError("Aucun prompt (keywords) défini dans config.yml.")

        # Récupérer la clé API depuis les variables d'environnement
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError(
                "Clé API Hugging Face manquante. "
                "Définissez la variable d'environnement HUGGINGFACE_API_KEY. "
                "Obtenez une clé sur: https://huggingface.co/settings/tokens"
            )

        print(f"📝 Prompt: {prompt}")
        print(f"🖼️  Modèle: {model}")
        print(f"🌐 Fournisseur: {provider}")

        # Générer et sauvegarder l'image
        image_data = generate_image_hf(api_key, prompt, model, provider)
        saved_path = save_image(image_data, output_path)
        print(f"✨ Image générée et sauvegardée sous: {saved_path}")

    except Exception as e:
        print(f"❌ Erreur: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())