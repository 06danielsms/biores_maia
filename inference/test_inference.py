#!/usr/bin/env python3
"""
Script para probar inferencias con los modelos Qwen2.5-3B
Soporta:
- Modelo Base (sin fine-tuning)
- Modelo LoRA (fine-tuned)
- Con parámetros fijos o con TD3 (si está disponible)
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

# Importación opcional de stable_baselines3 (solo necesario para TD3)
try:
    from stable_baselines3 import TD3
    TD3_AVAILABLE = True
except ImportError:
    TD3_AVAILABLE = False
    print("⚠️  Warning: stable_baselines3 no está instalado. TD3 no estará disponible.")


def build_prompt(medical_text: str) -> str:
    """Construye el prompt para el modelo."""
    return (
        "You are a specialist in healthcare communication. "
        "Use the context to transform the following medical text into a clear, "
        "concise, and easy-to-understand summary for a patient and their family. "
        "Retain all relevant clinical data, but explain technical terms using simple "
        "language and short sentences.\n\n"
        "### Medical text:\n"
        f"{medical_text}\n\n"
        "### Simplified summary:\n"
    )


@torch.no_grad()
def generate_summary(
    model,
    tokenizer,
    medical_text: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> str:
    """Genera un resumen simplificado del texto médico."""
    prompt = build_prompt(medical_text)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extraer solo la parte después de "### Simplified summary:"
    if "### Simplified summary:" in full_text:
        return full_text.split("### Simplified summary:")[-1].strip()
    
    return full_text.strip()


def load_base_model(model_id: str, device: str = "auto"):
    """Carga el modelo base."""
    print(f"\n{'='*80}")
    print(f"CARGANDO MODELO BASE: {model_id}")
    print(f"{'='*80}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    print(f"✓ Modelo base cargado exitosamente")
    return model, tokenizer


def load_lora_model(base_model, adapter_dir: str):
    """Carga los adaptadores LoRA sobre el modelo base."""
    print(f"\n{'='*80}")
    print(f"CARGANDO ADAPTADORES LoRA: {adapter_dir}")
    print(f"{'='*80}")
    
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"No se encuentra el directorio de adaptadores: {adapter_dir}")
    
    model_lora = PeftModel.from_pretrained(base_model, adapter_dir)
    print(f"✓ Adaptadores LoRA cargados exitosamente")
    
    return model_lora


def load_td3_agent(agent_path: str):
    """Carga un agente TD3 entrenado."""
    if not TD3_AVAILABLE:
        raise ImportError(
            "stable_baselines3 no está instalado. "
            "Instálalo con: pip install stable-baselines3==2.3.0"
        )
    
    print(f"\n{'='*80}")
    print(f"CARGANDO AGENTE TD3: {agent_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"No se encuentra el agente TD3: {agent_path}")
    
    agent = TD3.load(agent_path)
    print(f"✓ Agente TD3 cargado exitosamente")
    
    return agent


def get_td3_params(agent, medical_text: str):
    """Obtiene parámetros de decodificación del agente TD3."""
    # Observación: longitud del texto normalizada
    n_chars = len(medical_text)
    obs = np.array([min(1.0, n_chars / 8000.0)], dtype=np.float32)
    
    # Predecir acción
    action, _ = agent.predict(obs, deterministic=True)
    temperature = float(np.clip(action[0], 0.1, 1.0))
    top_p = float(np.clip(action[1], 0.1, 1.0))
    
    return temperature, top_p


def main():
    parser = argparse.ArgumentParser(
        description="Prueba de inferencia con modelos Qwen2.5-3B para simplificación médica"
    )
    
    # Argumentos del modelo
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="ID del modelo base en HuggingFace"
    )
    
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Ruta a los adaptadores LoRA (si se usa fine-tuning)"
    )
    
    parser.add_argument(
        "--td3-agent",
        type=str,
        default=None,
        help="Ruta al agente TD3 (para optimizar temperatura y top_p)"
    )
    
    # Parámetros de generación (si no se usa TD3)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperatura para la generación (0.1-1.0)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus sampling) para la generación (0.1-1.0)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Número máximo de tokens a generar"
    )
    
    # Texto de entrada
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Texto médico a simplificar"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Archivo con texto médico a simplificar"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo para la inferencia (auto, cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    # Validar entrada
    if not args.text and not args.file:
        print("\n❌ Error: Debes proporcionar un texto (--text) o un archivo (--file)")
        parser.print_help()
        return
    
    # Obtener texto médico
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            medical_text = f.read().strip()
    else:
        medical_text = args.text
    
    print(f"\n{'='*80}")
    print("TEXTO MÉDICO DE ENTRADA")
    print(f"{'='*80}")
    print(medical_text[:500] + ("..." if len(medical_text) > 500 else ""))
    print(f"\nLongitud: {len(medical_text)} caracteres")
    
    # Cargar modelo base
    base_model, tokenizer = load_base_model(args.base_model, args.device)
    
    # Cargar LoRA si se especifica
    if args.lora_adapter:
        model = load_lora_model(base_model, args.lora_adapter)
        model_name = "LoRA Fine-tuned"
    else:
        model = base_model
        model_name = "Base"
    
    # Cargar TD3 si se especifica
    td3_agent = None
    if args.td3_agent:
        if not TD3_AVAILABLE:
            print("\n❌ Error: stable_baselines3 no está instalado.")
            print("   Instálalo con: pip install stable-baselines3==2.3.0")
            return
        td3_agent = load_td3_agent(args.td3_agent)
        model_name += " + TD3"
    
    # Determinar parámetros de generación
    if td3_agent:
        temperature, top_p = get_td3_params(td3_agent, medical_text)
        print(f"\n📊 Parámetros optimizados por TD3:")
        print(f"   Temperature: {temperature:.3f}")
        print(f"   Top-p: {top_p:.3f}")
    else:
        temperature = args.temperature
        top_p = args.top_p
        print(f"\n📊 Parámetros fijos:")
        print(f"   Temperature: {temperature}")
        print(f"   Top-p: {top_p}")
    
    # Generar resumen
    print(f"\n{'='*80}")
    print(f"GENERANDO RESUMEN CON {model_name.upper()}")
    print(f"{'='*80}")
    
    summary = generate_summary(
        model=model,
        tokenizer=tokenizer,
        medical_text=medical_text,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=args.max_tokens,
    )
    
    print(f"\n{'='*80}")
    print("RESUMEN SIMPLIFICADO")
    print(f"{'='*80}")
    print(summary)
    print(f"\n{'='*80}")
    
    # Calcular métricas básicas de legibilidad
    try:
        import textstat
        flesch = textstat.flesch_reading_ease(summary)
        print(f"\n📈 Flesch Reading Ease: {flesch:.2f}")
        print(f"   (Mayor es más fácil de leer. >60 = estándar, >80 = fácil)")
    except ImportError:
        print("\n💡 Tip: Instala 'textstat' para ver métricas de legibilidad")
    
    print(f"\n✓ Inferencia completada con éxito")


if __name__ == "__main__":
    main()
