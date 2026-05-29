"""Generateur d'image avec Hugging Face et selection de prompts ponderes."""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
import io
from prompt_generator import (
    load_yaml_config,
    generate_random_prompts,
    select_prompts
)

# Charger les variables d'environnement depuis .env
load_dotenv()

# Parser les arguments en ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Generateur d'image avec Hugging Face.")
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Mode automatique : genere 3 images sans afficher la fenetre de selection."
    )
    return parser.parse_args()









# 5. Generer une image via Hugging Face InferenceClient
def generate_image_hf(api_key: str, prompt: str, model: str = "black-forest-labs/FLUX.1-dev", provider: str = "wavespeed") -> bytes:
    """
    Genere une image via Hugging Face InferenceClient.
    """
    if not api_key:
        raise ValueError("La cle API Hugging Face est manquante. Definissez HUGGINGFACE_API_KEY dans .env.")

    try:
        client = InferenceClient(provider=provider, api_key=api_key)
        print(f"  [WAIT] Generation de l'image pour : '{prompt}'...")
        image = client.text_to_image(prompt=prompt, model=model)

        # Convertir l'image PIL en bytes (PNG)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    except Exception as e:
        raise ValueError(f"Erreur lors de la generation de l'image: {e}")

# 6. Sauvegarder l'image sur le disque
def save_image(image_data: bytes, output_path: str, prompt: str) -> str:
    """
    Sauvegarde les donnees binaires d'une image dans un fichier.
    """
    # Nettoyer le prompt pour en faire un nom de fichier valide
    safe_prompt = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in prompt[:50])
    filename = f"{safe_prompt}.png"
    path = Path(output_path) / filename

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as file:
        file.write(image_data)

    return str(path)

def main() -> int:
    """Point d'entree principal du script."""
    # Parser les arguments en ligne de commande
    args = parse_args()

    print("[START] Generateur d'image avec selection de prompts ponderes - Demarrage...")

    # Chemin vers le fichier YAML (relatif au script)
    script_dir = Path(__file__).parent
    yaml_file = script_dir / "config.yml"

    try:
        # Charger la configuration
        config = load_yaml_config(str(yaml_file))
        print(f"[OK] Configuration chargee depuis {yaml_file}")

        # Recuperer les parametres
        keywords = config.get("keywords", [])
        num_prompts = config.get("num_prompts", 5)
        max_keywords_per_prompt = config.get("max_keywords_per_prompt", 3)
        output_path = config.get("output_path", "output/")
        model = config.get("model", "black-forest-labs/FLUX.1-dev")
        provider = config.get("provider", "wavespeed")

        # Verifier que les mots-cles sont presents
        if not keywords:
            raise ValueError("Aucun mot-cle defini dans config.yml. Ajoutez une liste 'keywords' avec 'text' et 'weight'.")

        # Recuperer les parametres Mistral (optionnels)
        prompt_context = config.get("prompt_contexte", "")
        mistral_model = config.get("mistral_model", "mistral-tiny")
        mistral_api_url = config.get("mistral_api_url", "https://api.mistral.ai/v1/chat/completions")
        mistral_api_key = os.getenv("MISTRAL_API_KEY")

        # Recuperer la cle API Hugging Face
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cle API Hugging Face manquante. "
                "Definissez la variable d'environnement HUGGINGFACE_API_KEY. "
                "Obtenez une cle sur: https://huggingface.co/settings/tokens"
            )

        print(f"[CONFIG] Mots-cles charges (ponderes): {[(kw['text'], kw['weight']) for kw in keywords]}")
        print(f"[RANDOM] Nombre de prompts a generer: {num_prompts}")
        print(f"[MAX] Mots-cles max par prompt: {max_keywords_per_prompt}")
        print(f"[MODEL] Modele: {model} | Fournisseur: {provider}")

        # Generer des prompts aleatoires ponderes
        prompts = generate_random_prompts(keywords, num_prompts, max_keywords_per_prompt)
        print(f"\n[RANDOM] Prompts generes: {prompts}")

        # Mode automatique (-y) : pas de fenetre, utilise les 3 premiers prompts
        if args.yes:
            print("\n[AUTO] Mode automatique active (-y) : generation de 3 images avec les 3 premiers prompts...")
            selected_prompts = prompts[:3]
            if len(prompts) < 3:
                print(f"[WARNING] Seulement {len(prompts)} prompt(s) disponible(s), utilisation de tous.")
                selected_prompts = prompts
        else:
            # Afficher la boite de dialogue pour selectionner les prompts
            print("\n[SELECT] Ouverture de la fenetre de selection des prompts...")
            
            # Passer les parametres Mistral pour le raffinement automatique
            selected_prompts = select_prompts(
                prompts,
                context=prompt_context,
                mistral_api_key=mistral_api_key,
                mistral_model=mistral_model,
                mistral_api_url=mistral_api_url
            )

            if not selected_prompts:
                print("[ERROR] Aucun prompt selectionne. Au revoir !")
                return 0

        print(f"[OK] Prompts selectionnes: {selected_prompts}")

        # Generer les images pour chaque prompt selectionne
        print(f"\n[GEN] Generation des images pour {len(selected_prompts)} prompt(s)...")
        for i, prompt in enumerate(selected_prompts, 1):
            print(f"\n[{i}/{len(selected_prompts)}]")
            try:
                image_data = generate_image_hf(api_key, prompt, model, provider)
                saved_path = save_image(image_data, output_path, prompt)
                print(f"  [SAVED] Image sauvegardee sous: {saved_path}")
            except Exception as e:
                print(f"  [ERROR] Erreur pour le prompt '{prompt}': {e}")

        print(f"\n[SUCCESS] Toutes les images ont ete generees dans le dossier '{output_path}' !")

    except Exception as e:
        print(f"[ERROR] Erreur: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())