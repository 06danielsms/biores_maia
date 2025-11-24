#!/bin/bash
# Script para probar diferentes configuraciones del modelo

echo "=========================================="
echo "EJEMPLOS DE USO - test_inference.py"
echo "=========================================="

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurar rutas (ajustar según tu instalación)
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
LORA_ADAPTER="./qwen2.5-3b-pls"  # Ajustar ruta
TD3_BASE_AGENT="./td3_base_agent.zip"  # Ajustar ruta
TD3_LORA_AGENT="./td3_lora_agent.zip"  # Ajustar ruta

# ==========================================
# EJEMPLO 1: Modelo Base (sin fine-tuning)
# ==========================================
echo -e "\n${BLUE}EJEMPLO 1: Modelo Base${NC}"
echo "Usando el modelo base sin modificaciones"
echo "=========================================="
python test_inference.py \
  --base-model "$BASE_MODEL" \
  --text "This randomized controlled trial evaluated the efficacy of angiotensin-converting enzyme inhibitors in patients with hypertension. The study showed a significant reduction in systolic blood pressure (p<0.001) compared to placebo." \
  --temperature 0.7 \
  --top-p 0.9

# ==========================================
# EJEMPLO 2: Modelo LoRA (fine-tuned)
# ==========================================
echo -e "\n${BLUE}EJEMPLO 2: Modelo LoRA (Fine-tuned)${NC}"
echo "Usando el modelo con adaptadores LoRA"
echo "=========================================="
python test_inference.py \
  --base-model "$BASE_MODEL" \
  --lora-adapter "$LORA_ADAPTER" \
  --file ejemplos_medicos.txt \
  --temperature 0.7 \
  --top-p 0.9

# ==========================================
# EJEMPLO 3: Modelo Base + TD3
# ==========================================
echo -e "\n${BLUE}EJEMPLO 3: Modelo Base + TD3${NC}"
echo "Usando TD3 para optimizar temperatura y top_p"
echo "=========================================="
python test_inference.py \
  --base-model "$BASE_MODEL" \
  --td3-agent "$TD3_BASE_AGENT" \
  --file ejemplos_medicos.txt

# ==========================================
# EJEMPLO 4: Modelo LoRA + TD3
# ==========================================
echo -e "\n${BLUE}EJEMPLO 4: Modelo LoRA + TD3 (Mejor configuración)${NC}"
echo "Combinando fine-tuning con optimización TD3"
echo "=========================================="
python test_inference.py \
  --base-model "$BASE_MODEL" \
  --lora-adapter "$LORA_ADAPTER" \
  --td3-agent "$TD3_LORA_AGENT" \
  --file ejemplos_medicos.txt

# ==========================================
# EJEMPLO 5: Texto desde línea de comandos
# ==========================================
echo -e "\n${BLUE}EJEMPLO 5: Texto directo desde línea de comandos${NC}"
echo "=========================================="
python test_inference.py \
  --base-model "$BASE_MODEL" \
  --lora-adapter "$LORA_ADAPTER" \
  --text "Myocardial infarction, commonly known as a heart attack, occurs when blood flow to the coronary arteries is blocked, typically by a thrombus. This results in ischemia and necrosis of cardiac myocytes. Patients may present with chest pain, dyspnea, and diaphoresis. Treatment includes antiplatelet therapy, anticoagulation, and potentially percutaneous coronary intervention." \
  --temperature 0.8 \
  --top-p 0.95 \
  --max-tokens 300

echo -e "\n${GREEN}✓ Ejemplos completados${NC}"
