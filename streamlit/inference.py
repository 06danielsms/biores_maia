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
            "Using the following abstract of a biomedical study as input, generate a Plain Language Summary (PLS) "
            "understandable by any patient, regardless of their health literacy. Ensure that the generated text adheres to the "
            "following instructions which should be followed step-by-step:\n"
            "a. Specific Structure: The generated PPLS should be presented in a logical order, using the following headings:\n"
            "1. Plain Protocol Title\n"
            "2. Rationale\n"
            "3. Objectives\n"
            "4. Trial Design\n"
            "5. Trial Population\n"
            "6. Interventions\n"
            "b. Sections should be authored following these parameters:\n"
            "1. Plain Protocol Title: Simplified protocol title understandable to a layperson but including specific indication "
            "for which the study is meant.\n"
            "2. Rationale: Include the phrase 'Researchers are looking for a better way to treat [condition]; background or "
            "study rationale describing the condition: what it is, what it may cause, and why it is a burden for the patients; "
            "the reason and main hypothesis for the study; and why the study is needed, and the study medication has the "
            "potential to treat the condition.\n"
            "3. Objectives: Answer 'What are the goals of the study?' Specify the main and secondary objectives of the trial "
            "and how they will be measured (e.g., the main trial endpoint is the percent change in the number of events "
            "from baseline to a specified time or the total number of adverse reactions at a particular time after baseline).\n"
            "4. Trial Design: Answer 'How is this study designed?' Include the description of the design and the expected "
            "amount of time a person will be in the study.\n"
            "5. Trial Population: Answer 'Who will participate in this study?' Include a description of the study and patient "
            "population (age, health condition, gender), and the key inclusion and exclusion criteria.\n"
            "6. Interventions: Answer 'What treatments are being given during the study?' Include a description of the "
            "medication, vaccine, or treatment(s) being studied, the route of administration, the duration of treatment, and "
            "any study-related diagnostic and monitoring procedures used. Include justification if a placebo is used.\n"
            "c. Consistency and Replicability: The generated PPLS should be consistent regardless of the order of sentences or "
            "the specific phrasing used in the input protocol text.\n"
            "d. Compliance with Plain Language Guidelines: The generated PPLS must follow these plain language guidelines:\n"
            "• Have readability grade level of 6 or below.\n"
            "• Do not have jargon. All technical or medical words or terms should be defined or broken down into simple "
            "and logical explanations.\n"
            "• Active voice, not passive.\n"
            "• Mostly one or two-syllable words.\n"
            "• Sentences of 15 words or less.\n"
            "• Short paragraphs of 3-5 sentences.\n"
            "• Simple numbers (e.g., ratios, no percentages).\n"
            "e. No Extra Content: The AI model should not invent information or add content that is not present in the input "
            "protocol. The PPLS should only present information from the original protocol in a simplified and understandable "
            "manner.\n"
            "f. Aim for an approximate PPLS length of 700-900 words.\n\n"
            "### Biomedical study abstract:\n"
            f"{medical_text}\n\n"
            "### Plain Language Summary:\n"
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
