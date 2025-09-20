

```mermaid
%%{init: {'themeVariables': { 'fontSize': '18px' }}}%%
flowchart TD
  subgraph FUENTES
    A["ClinicalTrials, Cochrane, Pfizer, etc."]
  end

  subgraph INGESTA
    B1["DVC + Git"]
    B2["S3 (raw storage)"]
  end

  subgraph ORQ
    C1["ZenML"]
    C2["Workers / Agents"]
  end

  subgraph PREP
    D1["regex, ftfy, unidecode"]
    D2["tokenizers"]
  end

  subgraph FEAT
    E1["sentence-transformers (roberta-base-bne, st-multilingual)"]
    E2["TF-IDF / BM25"]
  end

  subgraph CLS
    F1["Classic: SVM, LogReg, LightGBM"]
    F2["Transformer head"]
  end

  subgraph GEN
    G1["Ollama (serv.)"]
    G2["OpenAI, Deepseek-R1, Qwen3, phi3"]
    G3["PEFT / LoRA / QLoRA + bitsandbytes"]
  end

  subgraph EVAL
    H1["BERTScore"]
    H2["AlignScore"]
    H3["Readability (Flesch, Gunning, SMOG)"]
    H4["Opik (tracking)"]
  end

  subgraph SERV
    I1["FastAPI"]
    I2["Streamlit / Vue.js"]
    I3["MongoDB"]
    I4["Docker / Compose"]
  end

  A --> B1
  B1 --> B2
  B2 --> C1
  C1 --> C2
  C2 --> D1
  D1 --> D2
  D2 --> E1
  D2 --> E2

  E1 --> F2
  E2 --> F1
  F2 --> F1

  D2 --> G1
  G1 --> G2
  G2 --> G3

  G1 --> H1
  G1 --> H2
  G1 --> H3
  H1 --> H4
  H2 --> H4
  H3 --> H4

  F1 --> I1
  F2 --> I1
  H4 --> I1
  I1 --> I3
  I1 --> I2
  I1 --> I4
```

## Tecnologías por capa

- Fuentes: orígenes txt/CSV (ClinicalTrials, Cochrane, Pfizer, resúmenes).
- Ingesta y versionado: DVC + Git; almacenamiento bruto en S3 o S3-compatible.
- Orquestación: ZenML; workers/agents para ejecución distribuida.
- Preprocesado: regex, ftfy, unidecode; tokenizers de Hugging Face.
- Representaciones:
  - Contextuales: sentence-transformers (roberta-base-bne, st-multilingual).
  - Dispersas: TF-IDF o BM25 (scikit-learn / rank-bm25).
- Clasificación:
  - Clásicos: scikit-learn (SVM, LogisticRegression), LightGBM.
  - Deep: cabeza de clasificación sobre encoder (transformers / PyTorch).
- Generación (resúmenes): Ollama para servir modelos; Gemma2/Qwen2/Phi-2/TinyLlama; PEFT (LoRA/QLoRA) + bitsandbytes; checkpoints en S3.
- Evaluación y tracking: BERTScore, AlignScore; métricas de legibilidad; Opik para tracking de experimentos.
- Serving & UI: FastAPI (endpoints generate/score/classify), Streamlit o Vue.js, MongoDB para metadatos, Docker + Docker Compose.

---


