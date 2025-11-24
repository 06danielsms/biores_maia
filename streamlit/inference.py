"""
Módulo de inferencia para modelos Qwen2.5-3B
Permite realizar simplificación de texto médico usando:
- Modelo Base
- Modelo LoRA (fine-tuned)
- Con o sin TD3 (optimización de parámetros)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Importaciones opcionales
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from stable_baselines3 import TD3
    TD3_AVAILABLE = True
except ImportError:
    TD3_AVAILABLE = False


@dataclass
class InferenceConfig:
    """Configuración para la inferencia."""
    base_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_adapter_path: Optional[str] = None
    td3_agent_path: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    device: str = "auto"


@dataclass
class InferenceResult:
    """Resultado de una inferencia."""
    input_text: str
    simplified_text: str
    temperature: float
    top_p: float
    model_config: str
    input_length: int
    output_length: int
    flesch_score: Optional[float] = None


class ModelInference:
    """Clase para manejar inferencia con modelos Qwen."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.td3_agent = None
        self._model_loaded = False
    
    def load_model(self) -> None:
        """Carga el modelo base y componentes opcionales."""
        if self._model_loaded:
            return
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar modelo base
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_id,
            device_map=self.config.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Cargar LoRA si está configurado
        if self.config.lora_adapter_path:
            if not PEFT_AVAILABLE:
                raise ImportError("peft no está instalado. Instálalo con: pip install peft")
            
            if not os.path.exists(self.config.lora_adapter_path):
                raise FileNotFoundError(f"Adaptadores LoRA no encontrados: {self.config.lora_adapter_path}")
            
            self.model = PeftModel.from_pretrained(base_model, self.config.lora_adapter_path)
        else:
            self.model = base_model
        
        # Cargar TD3 si está configurado
        if self.config.td3_agent_path:
            if not TD3_AVAILABLE:
                raise ImportError("stable_baselines3 no está instalado. Instálalo con: pip install stable-baselines3")
            
            agent_path = Path(self.config.td3_agent_path)
            if not agent_path.exists():
                raise FileNotFoundError(f"Agente TD3 no encontrado: {agent_path}")
            
            # TD3 espera un directorio, cargar desde el path directo
            self.td3_agent = TD3.load(str(agent_path))
        
        self._model_loaded = True
    
    def _build_prompt(self, medical_text: str) -> str:
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
    
    def _get_td3_params(self, medical_text: str) -> Tuple[float, float]:
        """Obtiene parámetros de decodificación del agente TD3."""
        if self.td3_agent is None:
            return self.config.temperature, self.config.top_p
        
        # Observación: longitud del texto normalizada
        n_chars = len(medical_text)
        obs = np.array([min(1.0, n_chars / 8000.0)], dtype=np.float32)
        
        # Predecir acción
        action, _ = self.td3_agent.predict(obs, deterministic=True)
        temperature = float(np.clip(action[0], 0.1, 1.0))
        top_p = float(np.clip(action[1], 0.1, 1.0))
        
        return temperature, top_p
    
    @torch.no_grad()
    def generate(self, medical_text: str) -> InferenceResult:
        """Genera un resumen simplificado del texto médico."""
        if not self._model_loaded:
            self.load_model()
        
        # Determinar parámetros de generación
        temperature, top_p = self._get_td3_params(medical_text)
        
        # Construir prompt
        prompt = self._build_prompt(medical_text)
        
        # Tokenizar
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # Generar
        output_ids = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decodificar
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extraer solo la parte simplificada
        if "### Simplified summary:" in full_text:
            simplified_text = full_text.split("### Simplified summary:")[-1].strip()
        else:
            simplified_text = full_text.strip()
        
        # Calcular Flesch Reading Ease si textstat está disponible
        flesch_score = None
        try:
            import textstat
            flesch_score = textstat.flesch_reading_ease(simplified_text)
        except ImportError:
            pass
        
        # Determinar configuración del modelo
        model_config = "Base"
        if self.config.lora_adapter_path:
            model_config = "LoRA"
        if self.td3_agent:
            model_config += " + TD3"
        
        return InferenceResult(
            input_text=medical_text,
            simplified_text=simplified_text,
            temperature=temperature,
            top_p=top_p,
            model_config=model_config,
            input_length=len(medical_text),
            output_length=len(simplified_text),
            flesch_score=flesch_score,
        )
    
    def unload_model(self) -> None:
        """Libera la memoria del modelo."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.td3_agent is not None:
            del self.td3_agent
            self.td3_agent = None
        self._model_loaded = False
        
        # Limpiar cache de CUDA si está disponible
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_available_models(base_path: str = "../inference") -> dict:
    """
    Detecta los modelos y agentes disponibles en el directorio base.
    
    Returns:
        dict con claves 'lora' y 'td3_agents' conteniendo paths disponibles
    """
    base = Path(base_path)
    available = {
        "lora": [],
        "td3_agents": [],
    }
    
    # Buscar adaptadores LoRA
    lora_candidates = [
        base / "qwen2.5-3b-pls",
        base / "qwen2.5-3b-pls/qwen2.5-3b-pls",
    ]
    
    for path in lora_candidates:
        if path.exists() and (path / "adapter_config.json").exists():
            available["lora"].append(str(path))
    
    # Buscar agentes TD3 (directorios o archivos .zip)
    td3_candidates = [
        base / "td3_base_agent",
        base / "td3_base_agent.zip",
        base / "td3_lora_agent",
        base / "td3_lora_agent.zip",
    ]
    
    for path in td3_candidates:
        # Aceptar directorios con policy.pth o archivos .zip
        if path.exists():
            if path.is_dir() and (path / "policy.pth").exists():
                available["td3_agents"].append({
                    "name": path.name,
                    "path": str(path),
                })
            elif path.suffix == ".zip":
                available["td3_agents"].append({
                    "name": path.stem,
                    "path": str(path),
                })
    
    return available


__all__ = [
    "InferenceConfig",
    "InferenceResult",
    "ModelInference",
    "get_available_models",
    "PEFT_AVAILABLE",
    "TD3_AVAILABLE",
]
