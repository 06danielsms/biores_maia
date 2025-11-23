from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from preprocessing import (
    DEFAULT_BATCH_DIR,
    DEFAULT_CONFIG,
    PreprocessConfig,
    apply_preprocessing,
    load_default_text,
    process_directory,
    persist_parquet_outputs,
)
from training import (
    DEFAULT_NOPLS_CHUNKS,
    DEFAULT_PLS_CHUNKS,
    TrainingConfig,
    TrainingResult,
    run_training_job,
)
from utils import load_repo_config, read_uploaded_text
from validation import evaluate_rows, load_metrics_dataset, plot_metric, score_pair
from metrics_computer import compute_metrics_batch
from metrics_visualizer import (
    plot_histograms_streamlit,
    plot_boxplots_streamlit,
    plot_correlation_heatmap,
    plot_medians_comparison,
    render_metrics_summary,
)

st.set_page_config(page_title="BioRes MAIA Studio", layout="wide")
st.title("BioRes MAIA · Laboratorio interactivo")
st.caption("Preprocesamiento ➝ Entrenamiento ➝ Validación, todo en una sola app Streamlit.")


def render_preprocessing() -> None:
    st.subheader("1 · Preprocesamiento y configuración")
    st.write(
        "Esta sección envuelve la lógica de `scripts/clean_en.py` para limpiar texto, "
        "normalizar datos sensibles y preparar chunks antes del entrenamiento."
    )

    uploaded = st.file_uploader("Carga un archivo (.txt, .md)", type=["txt", "md"])
    default_text = load_default_text()
    text_value = read_uploaded_text(uploaded) or default_text
    
    # Layout con columnas: texto entrada y texto procesado lado a lado
    input_col, output_col = st.columns(2)
    
    with input_col:
        text_input = st.text_area("Texto de entrada", value=text_value, height=220, key="input-text")

    col1, col2, col3 = st.columns(3)
    with col1:
        lowercase = st.toggle("Minúsculas", value=DEFAULT_CONFIG.lowercase)
        strip_html = st.toggle("Eliminar HTML", value=DEFAULT_CONFIG.strip_html)
        replace_urls = st.toggle("Reemplazar URLs", value=DEFAULT_CONFIG.replace_urls)
    with col2:
        remove_punct = st.toggle("Quitar puntuación", value=DEFAULT_CONFIG.remove_punctuation)
        normalize_unicode = st.toggle("Normalizar Unicode", value=DEFAULT_CONFIG.normalize_unicode)
        replace_numbers = st.selectbox("Números", options=["normalize", "mask", "keep"], index=0)
    with col3:
        deidentify = st.toggle("De-identificar PHI", value=DEFAULT_CONFIG.deidentify_phi)
        normalize_ws = st.toggle("Normalizar espacios", value=DEFAULT_CONFIG.normalize_whitespace)
        chunk_tokens = st.slider("Tokens por chunk", min_value=50, max_value=400, value=120, step=10)
        chunk_overlap = st.slider("Overlap", min_value=0, max_value=120, value=20, step=5)

    cfg = PreprocessConfig(
        lowercase=lowercase,
        remove_punctuation=remove_punct,
        normalize_unicode=normalize_unicode,
        strip_html=strip_html,
        replace_urls=replace_urls,
        replace_emails=True,
        deidentify_phi=deidentify,
        replace_numbers=replace_numbers,
        normalize_whitespace=normalize_ws,
    )

    if st.button("Aplicar preprocesamiento", type="primary"):
        if not text_input.strip():
            st.warning("Proporciona texto manualmente o sube un archivo antes de continuar.")
        else:
            result = apply_preprocessing(text_input, cfg, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap)
            st.session_state["preprocess_result"] = result
            st.session_state["process_counter"] = st.session_state.get("process_counter", 0) + 1
            st.rerun()
    
    # Mostrar resultado en la columna derecha si existe
    result = st.session_state.get("preprocess_result")
    if result:
        with output_col:
            st.text_area("Texto procesado", result["clean_text"], height=220, key=f"output-text-{st.session_state.get('process_counter', 0)}")
            st.download_button(
                "📥 Descargar",
                data=result["clean_text"].encode("utf-8"),
                file_name="texto_limpio.txt",
                mime="text/plain",
                key="download-clean"
            )
        
        # Métricas debajo de ambas columnas
        stats = result["stats"]
        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        met_col1.metric("Tokens original", stats["original"]["tokens"])
        met_col2.metric("Tokens limpio", stats["processed"]["tokens"], stats["delta_tokens"])
        met_col3.metric("Reducción %", f"{stats['token_reduction_pct']}%")
        met_col4.metric("Long. media palabra", stats["processed"]["avg_word_len"])

        st.markdown("#### Chunks sugeridos")
        preview = result["chunk_preview"]
        if preview:
            st.dataframe(pd.DataFrame(preview))
        else:
            st.info("No se generaron chunks (el texto limpio está vacío).")

    st.divider()
    
    st.markdown("#### 📈 Análisis de métricas de preprocesamiento")
    st.write(
        "Visualiza métricas textuales exhaustivas (legibilidad, diversidad léxica, "
        "distribuciones de caracteres) para analizar las diferencias entre PLS y NO_PLS."
    )
    
    # Auto-detect metrics files
    metrics_dir = Path("../metrics")
    available_metrics = []
    if metrics_dir.exists():
        available_metrics = [
            str(f) for f in metrics_dir.glob("*.parquet")
            if f.name.endswith("_metrics.parquet")
        ]
    
    if not available_metrics:
        st.warning("⚠️ No se encontraron archivos de métricas en la carpeta `metrics/`. "
                   "Ejecuta `./run_pipeline_local.sh` o calcula métricas manualmente.")
    else:
        st.info(f"📂 Encontrados {len(available_metrics)} archivos de métricas en `metrics/`")
        
        # Option to select which metrics file to load
        selected_file = st.selectbox(
            "Selecciona el archivo de métricas",
            options=available_metrics,
            format_func=lambda x: Path(x).name,
        )
        
        col_load1, col_load2 = st.columns([1, 3])
        with col_load1:
            load_btn = st.button("📊 Cargar y visualizar", type="primary", key="load-metrics-viz")
        with col_load2:
            if "metrics_df" in st.session_state:
                st.success(f"✅ {len(st.session_state['metrics_df']):,} chunks cargados")
        
        if load_btn:
            try:
                with st.spinner("Cargando métricas..."):
                    metrics_df = pd.read_parquet(selected_file)
                st.session_state["metrics_df"] = metrics_df
                st.success(f"✅ Métricas cargadas: {len(metrics_df):,} chunks, {len(metrics_df.columns)} columnas")
                
                # Show basic stats
                if "label" in metrics_df.columns:
                    label_counts = metrics_df["label"].value_counts()
                    st.write(f"**Distribución:** {label_counts.to_dict()}")
            except Exception as exc:
                st.error(f"Error al cargar métricas: {exc}")
    

    
    # Visualization section
    metrics_df = st.session_state.get("metrics_df")
    if metrics_df is not None and not metrics_df.empty:
        st.divider()
        st.markdown("### 📊 Visualizaciones de métricas")
        
        repo_cfg = load_repo_config()
        label_col = repo_cfg.get("io", {}).get("label_col", "label") if isinstance(repo_cfg, dict) else "label"
        
        # Summary metrics
        render_metrics_summary(metrics_df, label_col=label_col)
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Histogramas", "📦 Boxplots", "🔥 Correlación", "📈 Medianas"])
        
        with tab1:
            st.markdown("##### Distribuciones de densidad por métrica")
            st.write("Compara las distribuciones de métricas entre PLS y NO_PLS.")
            max_hist = st.slider("Número de histogramas", min_value=3, max_value=12, value=6, key="hist-count")
            plot_histograms_streamlit(metrics_df, label_col=label_col, bins=40, max_plots=max_hist)
        
        with tab2:
            st.markdown("##### Comparación de rangos por label")
            st.write("Los boxplots muestran mediana, cuartiles y rangos sin outliers extremos.")
            max_box = st.slider("Número de boxplots", min_value=3, max_value=10, value=6, key="box-count")
            plot_boxplots_streamlit(metrics_df, label_col=label_col, max_plots=max_box)
        
        with tab3:
            st.markdown("##### Correlaciones entre métricas")
            st.write("Identifica métricas redundantes o altamente correlacionadas.")
            plot_correlation_heatmap(metrics_df)
        
        with tab4:
            st.markdown("##### Valores medianos por label")
            st.write("Compara los valores centrales de cada métrica entre grupos.")
            plot_medians_comparison(metrics_df, label_col=label_col)
        
        # Download option
        st.divider()
        csv_buffer = io.StringIO()
        metrics_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "📥 Descargar métricas completas (CSV)",
            data=csv_buffer.getvalue(),
            file_name="metrics_preprocessing.csv",
            mime="text/csv",
        )


def plot_history(result: TrainingResult):
    import plotly.graph_objects as go  # type: ignore

    df = result.history_frame()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_acc"], mode="lines+markers", name="Acc Train"))
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_acc"], mode="lines+markers", name="Acc Val"))
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_f1"], mode="lines+markers", name="F1 Val", line=dict(dash="dash")))
    fig.update_layout(
        title="Progreso del entrenamiento",
        xaxis_title="Época",
        yaxis_title="Métrica",
        template="plotly_white",
    )
    return fig


def render_training() -> None:
    st.subheader("2 · Entrenamiento y monitoreo")
    st.write(
        "Se muestrean los parquet generados por la etapa de cleaning para entrenar un clasificador "
        "ligero (`SGDClassifier`). Esto permite visualizar las métricas clave como en los notebooks."
    )

    with st.expander("Ubicación de datos", expanded=False):
        pls_path = st.text_input("Chunks PLS", value=str(DEFAULT_PLS_CHUNKS))
        npls_path = st.text_input("Chunks NO_PLS", value=str(DEFAULT_NOPLS_CHUNKS))
        subset = st.slider("Muestras por label", min_value=100, max_value=1000, value=400, step=50)

    col1, col2, col3 = st.columns(3)
    with col1:
        learning_rate = st.number_input("Learning rate", min_value=1e-4, max_value=1e-1, value=0.001, step=1e-4, format="%.4f")
        epochs = st.slider("Épocas", min_value=2, max_value=15, value=5)
    with col2:
        batch_size = st.slider("Batch size", min_value=32, max_value=512, value=128, step=32)
        max_features = st.slider("Max features TF-IDF", min_value=1000, max_value=10000, value=6000, step=500)
    with col3:
        ngram_max = st.slider("n-gramas máximos", min_value=1, max_value=3, value=2)
        alpha = st.number_input("Alpha (L2)", min_value=1e-6, max_value=1e-2, value=1e-4, step=1e-6, format="%.6f")

    if st.button("Iniciar entrenamiento", type="primary"):
        cfg = TrainingConfig(
            pls_path=Path(pls_path),
            npls_path=Path(npls_path),
            subset_per_label=int(subset),
            learning_rate=float(learning_rate),
            epochs=int(epochs),
            batch_size=int(batch_size),
            max_features=int(max_features),
            ngram_max=int(ngram_max),
            alpha=float(alpha),
        )
        try:
            with st.spinner("Entrenando clasificador..."):
                result = run_training_job(cfg)
            st.session_state["training_result"] = result
            st.success("Entrenamiento completado")
        except Exception as exc:
            st.error(f"No fue posible entrenar el modelo: {exc}")

    result: TrainingResult | None = st.session_state.get("training_result")
    if result:
        latest = result.history[-1]
        met1, met2, met3, met4 = st.columns(4)
        met1.metric("Acc val", latest.val_acc)
        met2.metric("F1 val", latest.val_f1)
        met3.metric("Loss val", latest.val_loss)
        met4.metric("Características", result.feature_space)

        try:
            st.plotly_chart(plot_history(result), use_container_width=True)
        except Exception as exc:  # pragma: no cover - optional dependency guard
            st.warning(f"No se pudo renderizar la gráfica (instala plotly): {exc}")

        st.markdown("#### Distribución de clases muestreadas")
        st.dataframe(pd.DataFrame(
            [(label, count) for label, count in result.class_distribution.items()],
            columns=["Label", "Muestras"],
        ))

        st.markdown("#### Logs")
        st.code("\n".join(result.logs[-8:]) or "Sin logs", language="text")


def render_validation() -> None:
    st.subheader("3 · Validación y pruebas finales")
    st.write(
        "Analiza las métricas generadas a partir de `metrics.csv`, compara modelos y calcula ROUGE/BLEU "
        "para tus resúmenes generados."
    )

    metrics_df = load_metrics_dataset()
    if metrics_df is not None and not metrics_df.empty:
        metric_choice = st.selectbox(
            "Selecciona una métrica para visualizar",
            options=[
                "flesch_reading_ease",
                "ari",
                "n_words",
                "n_sents",
                "avg_word_len",
                "stopword_ratio",
            ],
        )
        try:
            chart = plot_metric(metrics_df, metric_choice)
            st.plotly_chart(chart, use_container_width=True)
        except Exception as exc:  # pragma: no cover
            st.warning(f"No se pudo renderizar la gráfica (instala plotly): {exc}")
        st.dataframe(metrics_df.head(20))
    else:
        st.info("metrics.csv no está disponible; omitiendo la visualización base.")

    st.markdown("#### Evalúa tus resúmenes")
    uploaded = st.file_uploader("Carga un CSV con columnas prediction,reference[,document_id]", type=["csv"], key="eval-uploader")
    if uploaded is not None:
        try:
            df_upload = pd.read_csv(uploaded)
            scored = evaluate_rows(df_upload)
            st.dataframe(scored)
            aggregate = scored[["rouge_f1", "bleu", "f1"]].mean().round(4)
            agg_col1, agg_col2, agg_col3 = st.columns(3)
            agg_col1.metric("ROUGE-1 F1", aggregate["rouge_f1"])
            agg_col2.metric("BLEU", aggregate["bleu"])
            agg_col3.metric("F1 tokens", aggregate["f1"])

            csv_buf = io.StringIO()
            scored.to_csv(csv_buf, index=False)
            st.download_button("Descargar métricas", data=csv_buf.getvalue(), file_name="validation_scores.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"No se pudo procesar el archivo: {exc}")

    st.markdown("#### Prueba rápida")
    col_ref, col_pred = st.columns(2)
    reference = col_ref.text_area("Referencia", height=140, key="ref")
    prediction = col_pred.text_area("Predicción", height=140, key="pred")
    if st.button("Calcular métricas", key="quick-eval"):
        if not reference.strip() or not prediction.strip():
            st.warning("Escribe tanto la referencia como la predicción.")
        else:
            scores = score_pair("quick", prediction, reference)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ROUGE-P", scores.rouge_precision)
            c2.metric("ROUGE-R", scores.rouge_recall)
            c3.metric("ROUGE-F1", scores.rouge_f1)
            c4.metric("BLEU", scores.bleu)


stage = st.sidebar.radio(
    "Etapas",
    options=["Preprocesamiento", "Entrenamiento", "Validación"],
    index=0,
)

if stage == "Preprocesamiento":
    render_preprocessing()
elif stage == "Entrenamiento":
    render_training()
else:
    render_validation()
