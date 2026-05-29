"""Prompt Generator Module.

This module provides functionality for generating weighted random prompts
from a configuration file, and displaying them in a Tkinter window for
user selection.
"""

import os
import random
import json
import yaml
import requests
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime


def load_yaml_config(file_path: str) -> dict:
    """Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, invalid, or cannot be parsed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        if not config:
            raise ValueError(f"The file {file_path} is empty or invalid.")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found. Check the path.")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error in {file_path}: {e}")


def select_weighted_keywords(keywords: list, max_keywords: int) -> list:
    """Select keywords based on their weight (0-10).
    
    Args:
        keywords: List of dictionaries with 'text' and 'weight' keys.
        max_keywords: Maximum number of keywords to select.
        
    Returns:
        List of selected keyword texts (unique).
    """
    if not keywords:
        return []

    # Extract texts and weights
    texts = [kw["text"] for kw in keywords]
    weights = [kw["weight"] for kw in keywords]

    # Normalize weights to probabilities
    total_weight = sum(weights)
    if total_weight == 0:
        # If all weights are 0, use uniform random selection
        return random.sample(texts, min(max_keywords, len(texts)))

    probabilities = [w / total_weight for w in weights]

    # Select a random number of keywords (between 1 and max_keywords)
    num_to_select = random.randint(1, min(max_keywords, len(keywords)))

    # Select with replacement considering weights
    selected_indices = random.choices(
        range(len(keywords)),
        weights=probabilities,
        k=num_to_select
    )

    # Return unique texts (avoid duplicates)
    return list({texts[i] for i in selected_indices})


def generate_random_prompts(keywords: list, num_prompts: int, max_keywords_per_prompt: int) -> list:
    """Generate random syntactically correct prompts from weighted keywords.
    
    Args:
        keywords: List of dictionaries with 'text' and 'weight' keys.
        num_prompts: Number of prompts to generate.
        max_keywords_per_prompt: Maximum number of keywords per prompt.
        
    Returns:
        List of generated prompt strings.
    """
    prompts = []

    for _ in range(num_prompts):
        # Select weighted keywords
        selected_keywords = select_weighted_keywords(keywords, max_keywords_per_prompt)

        # Build a natural prompt
        if len(selected_keywords) == 1:
            prompt = selected_keywords[0]
        elif len(selected_keywords) == 2:
            # Example: "un chat noir sous une lune pleine"
            connector = random.choice(['sous', 'avec', 'et', 'devant', 'pres de', 'sur', 'dans'])
            prompt = f"{selected_keywords[0]} {connector} {selected_keywords[1]}"
        else:
            # Example: "un chat noir, lune pleine, style cyberpunk"
            prompt = ", ".join(selected_keywords)

        # Add a random style or atmosphere (optional)
        styles = ["", "style cyberpunk", "ambiance futuriste", "eclairage neon", "atmosphere mysterieuse", "rendu realiste"]
        if random.random() > 0.5 and not any(style in prompt for style in styles[1:]):
            prompt += f", {random.choice(styles[1:])}"

        prompts.append(prompt)

    return prompts


def refine_prompts_with_mistral(
    prompts: list,
    context: str,
    api_key: str,
    model: str = "mistral-tiny",
    api_url: str = "https://api.mistral.ai/v1/chat/completions",
    return_message: bool = False
) -> list:
    """Send prompts with context to Mistral API to get refined prompts for image generation.
    
    Args:
        prompts: List of prompt strings to refine.
        context: System context/prompt for Mistral.
        api_key: Mistral API key.
        model: Mistral model to use.
        api_url: Mistral API endpoint URL.
        return_message: If True, returns tuple of (refined_prompts, user_message).
        
    Returns:
        List of refined prompt strings from Mistral, or tuple if return_message=True.
        
    Raises:
        ValueError: If API request fails or response is invalid.
    """
    if not api_key:
        raise ValueError("Mistral API key is required.")
    
    if not prompts:
        return [] if not return_message else ([], "")
    
    # Combine all prompts into a single user message
    prompts_list = "\n".join([f"{i+1}. {p}" for i, p in enumerate(prompts)])
    user_message = f"""Voici EXACTEMENT {len(prompts)} mots-clés ou consignes à transformer. 
Tu DOIS retourner EXACTEMENT {len(prompts)} prompts optimisés, UN par consigne, dans l'ORDRE.

{prompts_list}

RÈGLES STRICTES :
1. Retourne UNIQUEMENT un objet JSON valide
2. Format : {{"prompts": ["prompt_1", "prompt_2", ... "prompt_{len(prompts)}"]}}
3. Chaque prompt doit être une chaîne de caractères longue et détaillée
4. Aucune introduction, aucun commentaire, aucun texte supplémentaire
5. Juste le JSON, rien d'autre."""
    
    # Store user_message for debugging if needed
    _user_message = user_message
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": context},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract the refined prompt from Mistral's response
        if "choices" in data and len(data["choices"]) > 0:
            refined_content = data["choices"][0]["message"]["content"]
            
            # Clean up the content: remove ALL code block markers
            refined_content = refined_content.strip()
            # Remove code blocks (```json ... ``` or ``` ... ```)
            while "```" in refined_content:
                parts = refined_content.split("```")
                refined_content = "".join(parts[1::2])  # Take every other part starting from index 1
            refined_content = refined_content.strip()
            
            # Try to parse as JSON first
            try:
                refined_data = json.loads(refined_content)
                
                # Handle different JSON structures
                if isinstance(refined_data, list):
                    # Direct list of prompts
                    refined_prompts = refined_data
                elif isinstance(refined_data, dict):
                    if "prompts" in refined_data:
                        # List under "prompts" key
                        prompts_list = refined_data["prompts"]
                        if isinstance(prompts_list, list):
                            refined_prompts = []
                            for item in prompts_list:
                                if isinstance(item, str):
                                    refined_prompts.append(item)
                                elif isinstance(item, dict):
                                    # Extract prompt from dict
                                    for v in item.values():
                                        if isinstance(v, str):
                                            refined_prompts.append(v)
                                            break
                    else:
                        # Extract all string values from dict
                        refined_prompts = [v for v in refined_data.values() if isinstance(v, str)]
                else:
                    refined_prompts = [refined_data] if isinstance(refined_data, str) else [str(refined_data)]
                
                # Clean up: remove quotes if present
                refined_prompts = [p.strip('"\'') for p in refined_prompts]
                
                # Filter to keep only valid-looking prompts
                # A valid prompt should be long enough or contain typical image generation parameters
                prompt_keywords = ['--ar', '--v', '--style', '--chaos', '8K', 'HD', 'resolution', 'photo', 'image', 'scène', 'scene', 'ultra', 'hyper', 'realistic', 'detailed']
                refined_prompts = [
                    p for p in refined_prompts 
                    if (len(p) >= 50 or any(kw.lower() in p.lower() for kw in prompt_keywords))
                ]
                
                # Keep only the same number of prompts as we sent
                if len(refined_prompts) > len(prompts):
                    refined_prompts = refined_prompts[:len(prompts)]
                elif len(refined_prompts) < len(prompts):
                    # If we lost some prompts due to filtering, duplicate the last one to match count
                    # (This shouldn't happen if Mistral returns proper JSON)
                    while len(refined_prompts) < len(prompts):
                        refined_prompts.append(refined_prompts[-1] if refined_prompts else "")
                    
            except json.JSONDecodeError:
                # Fallback: try to extract prompts from text with quotes
                import re
                # Try to find all text between quotes
                quotes_pattern = r'"(?:\\.|[^"\\])*"'
                quoted_strings = re.findall(quotes_pattern, refined_content)
                
                if quoted_strings:
                    # Filter out JSON keys and short strings (likely keys), and remove quotes
                    refined_prompts = [
                        (s[1:-1] if s.startswith('"') and s.endswith('"') else s).strip()
                        for s in quoted_strings 
                        if len(s) > 20 and not s.lower().strip('"') in ["prompts", "prompt", "json"]
                    ]
                
                if not refined_prompts:
                    # Last resort: split by lines and filter
                    lines = [l.strip() for l in refined_content.split("\n") if l.strip()]
                    # Filter out markdown, separators, etc.
                    refined_prompts = [
                        l for l in lines 
                        if l and not l.startswith("**") and not l.startswith("---") 
                        and not l.startswith("Prompt") and not l.startswith("```")
                        and l not in ["{", "}", "[", "]"] and ":" not in l
                    ]
                    
                    if not refined_prompts:
                        refined_prompts = [refined_content]
                
                # Clean up: remove quotes if present
                refined_prompts = [p.strip('"\'') for p in refined_prompts]
                
                # Filter to keep only valid-looking prompts
                prompt_keywords = ['--ar', '--v', '--style', '--chaos', '8K', 'HD', 'resolution', 'photo', 'image', 'scène', 'scene', 'ultra', 'hyper', 'realistic', 'detailed']
                refined_prompts = [
                    p for p in refined_prompts 
                    if (len(p) >= 50 or any(kw.lower() in p.lower() for kw in prompt_keywords))
                ]
                
                # Keep only the same number of prompts as we sent
                if len(refined_prompts) > len(prompts):
                    refined_prompts = refined_prompts[:len(prompts)]
                elif len(refined_prompts) < len(prompts):
                    # If we lost some prompts due to filtering, duplicate the last one to match count
                    while len(refined_prompts) < len(prompts):
                        refined_prompts.append(refined_prompts[-1] if refined_prompts else "")
        else:
            raise ValueError(f"Unexpected response format from Mistral API: {data}")
            
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error calling Mistral API: {e}")
    
    return refined_prompts


def select_prompts(
    prompts: list,
    context: str = None,
    mistral_api_key: str = None,
    mistral_model: str = None,
    mistral_api_url: str = None
) -> list:
    """Display a Tkinter window with checkboxes to select prompts.
    
    If Mistral parameters are provided, selected prompts are sent to Mistral
    with the context to get refined prompts optimized for image generation APIs.
    
    Args:
        prompts: List of prompt strings to display.
        context: System context for Mistral (optional).
        mistral_api_key: Mistral API key (optional).
        mistral_model: Mistral model name (optional).
        mistral_api_url: Mistral API endpoint URL (optional).
        
    Returns:
        List of selected prompt strings.
        
    Raises:
        ImportError: If tkinter is not available.
        ValueError: If Mistral API call fails.
    """
    # Check if Mistral integration is requested
    use_mistral = all([context, mistral_api_key, mistral_model, mistral_api_url])
    root = tk.Tk()
    root.title("Selection des prompts pour la generation d'images")
    root.geometry("600x400")

    # Variables to store selected and final prompts
    selected_prompts = []
    final_prompts = []

    # Main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Instruction label
    ttk.Label(
        main_frame, 
        text="Cochez les prompts que vous souhaitez utiliser pour generer des images :", 
        wraplength=580
    ).pack(pady=(0, 10))

    # Variables for checkboxes
    check_vars = []
    for i, prompt in enumerate(prompts):
        var = tk.BooleanVar()
        check_vars.append(var)
        # Use tk.Checkbutton (not ttk) to support wraplength
        # Use prompts[i] to avoid late binding issue with loop variable
        cb = tk.Checkbutton(
            main_frame, 
            text=prompts[i], 
            variable=var, 
            onvalue=True, 
            offvalue=False, 
            wraplength=550,
            anchor=tk.W
        )
        cb.pack(anchor=tk.W, pady=2)

    # Submit button
    def on_submit():
        nonlocal selected_prompts, final_prompts
        selected_prompts = [prompts[i] for i, var in enumerate(check_vars) if var.get()]

        if not selected_prompts:
            messagebox.showwarning("Aucune selection", "Veuillez cocher au moins un prompt.")
            return

        # Initialize JSON structure with selected prompts
        json_data = {
            "selected_prompts": selected_prompts,
            "count": len(selected_prompts),
            "timestamp": datetime.now().isoformat()
        }

        # Default to original prompts
        final_prompts = selected_prompts

        # Refine prompts with Mistral if all parameters are provided
        if use_mistral:
            try:
                print("[MISTRAL] Raffinement des prompts avec Mistral (un par un)...")
                
                # Process each prompt individually for detailed debugging
                prompts_debug_info = []
                refined_prompts = []
                
                for i, single_prompt in enumerate(selected_prompts, 1):
                    print(f"[MISTRAL] Traitement du prompt {i}/{len(selected_prompts)}...")
                    
                    # Create user message for this single prompt
                    user_msg_for_prompt = f"""Voici 1 mot-clé ou consigne à transformer en un prompt optimisé pour la génération d'images photo-réalistes.

1. {single_prompt}

RÈGLES STRICTES :
1. Retourne UNIQUEMENT un objet JSON valide
2. Format : {{"prompts": ["prompt_1"]}}
3. Le prompt doit être une chaîne de caractères longue et détaillée
4. Aucune introduction, aucun commentaire, aucun texte supplémentaire
5. Juste le JSON, rien d'autre."""
                    
                    # Send single prompt to Mistral
                    refined = refine_prompts_with_mistral(
                        [single_prompt],
                        context,
                        mistral_api_key,
                        mistral_model,
                        mistral_api_url
                    )
                    
                    # Store debug info
                    prompts_debug_info.append({
                        "original": single_prompt,
                        "mistral_request": user_msg_for_prompt.strip(),
                        "refined": refined[0] if refined else single_prompt
                    })
                    
                    refined_prompts.append(refined[0] if refined else single_prompt)
                    print(f"[MISTRAL] Prompt {i} raffiné reçu")
                
                # Add system context to each prompt debug info
                for info in prompts_debug_info:
                    info["mistral_system_context"] = context
                
                # Build detailed JSON structure
                json_data["prompts"] = prompts_debug_info
                json_data["refined_prompts"] = refined_prompts
                json_data["refined_count"] = len(refined_prompts)
                json_data["mistral_system_context"] = context  # Also at root level for reference
                print(f"[MISTRAL] Tous les prompts raffinés reçus: {len(refined_prompts)}")
                
                # Use refined prompts as final output
                final_prompts = refined_prompts
            except ValueError as e:
                print(f"[MISTRAL ERROR] {e}")
                json_data["mistral_error"] = str(e)
                # Keep original prompts if Mistral fails
                final_prompts = selected_prompts
        else:
            # If not using Mistral, just create basic debug info
            json_data["prompts"] = [{"original": p} for p in selected_prompts]

        # Display JSON structure on console
        print(json.dumps(json_data, indent=2, ensure_ascii=False))

        root.quit()
        root.destroy()

    ttk.Button(main_frame, text="Valider", command=on_submit).pack(pady=10)

    # Run main loop
    root.mainloop()

    return final_prompts
