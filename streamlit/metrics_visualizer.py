"""Visualization utilities for preprocessing metrics."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["font.size"] = 9


def plot_histograms_streamlit(df: pd.DataFrame, label_col: str = "label", bins: int = 40, max_plots: int = 6) -> None:
    """Plot histogram distributions for numeric metrics."""
    # Exclude metadata columns
    exclude_cols = ["n_tokens", "document_id", "doc_id", "split"]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    if not num_cols:
        st.warning("No se encontraron columnas numéricas para visualizar.")
        return
    
    # Select most relevant metrics - updated with actual column names from compute_metrics.py
    priority_metrics = [
        "flesch_reading_ease", "automated_readability_index", "flesch_kincaid_grade", "gunning_fog",
        "n_words", "n_sents", "avg_word_len", "avg_sent_len_words",
        "prop_stop_tokens", "ttr", "lexical_density",
        "prop_long_words_7", "noun_ratio", "verb_ratio"
    ]
    selected_cols = [c for c in priority_metrics if c in num_cols][:max_plots]
    
    if not selected_cols:
        selected_cols = num_cols[:max_plots]
    
    # Create a grid layout
    ncols = 2
    nrows = (len(selected_cols) + 1) // 2
    
    if len(selected_cols) > 0:
        for idx, col in enumerate(selected_cols):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            
            if label_col in df.columns:
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
                labels_list = df[label_col].unique()
                
                for i, lab in enumerate(labels_list):
                    subset = df[df[label_col] == lab]
                    vals = subset[col].dropna().values
                    if len(vals) > 0:
                        color = colors[i % len(colors)]
                        ax.hist(vals, bins=bins, alpha=0.5, density=True, 
                               label=str(lab), edgecolor='black', linewidth=0.3,
                               color=color)
                ax.legend(fontsize=10, framealpha=0.9)
            else:
                vals = df[col].dropna().values
                if len(vals) > 0:
                    ax.hist(vals, bins=bins, alpha=0.7, density=True, 
                           edgecolor='black', linewidth=0.5, color='#4ECDC4')
            
            ax.set_title(f"Distribución: {col}", fontsize=12, weight='bold', pad=10)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel("Densidad", fontsize=10)
            ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def plot_boxplots_streamlit(df: pd.DataFrame, label_col: str = "label", max_plots: int = 6) -> None:
    """Plot boxplots comparing distributions by label."""
    exclude_cols = ["n_tokens", "document_id", "doc_id", "split"]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
    
    if label_col not in df.columns:
        st.info("No hay columna de label para comparar distribuciones.")
        return
    
    if not num_cols:
        st.warning("No se encontraron columnas numéricas para visualizar.")
        return
    
    # Select most relevant metrics - updated with actual column names
    priority_metrics = [
        "flesch_reading_ease", "automated_readability_index", "flesch_kincaid_grade",
        "n_words", "avg_word_len", "prop_stop_tokens", "ttr", "lexical_density",
        "prop_long_words_7", "noun_ratio"
    ]
    selected_cols = [c for c in priority_metrics if c in num_cols][:max_plots]
    
    if not selected_cols:
        selected_cols = num_cols[:max_plots]
    
    for col in selected_cols:
        labels_list = []
        vals_list = []
        for lab, g in df.groupby(label_col):
            v = g[col].dropna().values
            if len(v) > 0:
                labels_list.append(str(lab))
                vals_list.append(v)
        
        if vals_list:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bp = ax.boxplot(vals_list, labels=labels_list, showfliers=False, patch_artist=True,
                           widths=0.6)
            
            # Color boxes with improved palette
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
                median.set_color('darkred')
                median.set_linewidth(2)
            
            # Improve whiskers and caps
            for whisker in bp['whiskers']:
                whisker.set_linewidth(1.2)
                whisker.set_linestyle('--')
            for cap in bp['caps']:
                cap.set_linewidth(1.2)
            
            ax.set_title(f"Comparación: {col} por {label_col}", fontsize=12, weight='bold', pad=10)
            ax.set_ylabel(col, fontsize=10)
            ax.set_xlabel(label_col, fontsize=10)
            ax.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot correlation heatmap for numeric features."""
    num_df = df.select_dtypes(include=[np.number])
    
    # Remove non-feature columns
    cols_to_drop = ["n_tokens", "document_id", "doc_id", "split"]
    num_df = num_df.drop(columns=[c for c in cols_to_drop if c in num_df.columns], errors='ignore')
    
    if num_df.shape[1] < 3:
        st.info("No hay suficientes métricas numéricas para calcular correlaciones.")
        return
    
    # Limit to most important metrics to keep heatmap readable
    important_metrics = [
        "flesch_reading_ease", "automated_readability_index", "flesch_kincaid_grade", 
        "gunning_fog", "smog_index", "coleman_liau_index", "dale_chall",
        "n_words", "n_sents", "avg_word_len", "avg_sent_len_words",
        "ttr", "lexical_density", "prop_stop_tokens",
        "noun_ratio", "verb_ratio", "adj_ratio"
    ]
    
    available_metrics = [c for c in important_metrics if c in num_df.columns]
    if len(available_metrics) >= 3:
        num_df = num_df[available_metrics]
    
    corr = num_df.corr(numeric_only=True)
    
    fig, ax = plt.subplots(figsize=(max(12, 0.5 * len(corr.columns)), 9))
    cax = ax.imshow(corr, interpolation="nearest", cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8, ha='right')
    ax.set_yticklabels(corr.columns, fontsize=8)
    
    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlación', rotation=270, labelpad=20, fontsize=10)
    
    ax.set_title("Matriz de correlación de métricas", fontsize=14, weight='bold', pad=15)
    
    # Add gridlines
    ax.set_xticks(np.arange(len(corr.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr.columns)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_medians_comparison(df: pd.DataFrame, label_col: str = "label") -> None:
    """Plot median values comparison by label."""
    if label_col not in df.columns:
        st.info("No hay columna de label para comparar medianas.")
        return
    
    num_df = df.select_dtypes(include=[np.number])
    
    # Remove non-feature columns
    cols_to_drop = ["n_tokens", "document_id", "doc_id", "split"]
    num_df = num_df.drop(columns=[c for c in cols_to_drop if c in num_df.columns], errors='ignore')
    
    med = num_df.join(df[label_col]).groupby(label_col).median(numeric_only=True)
    
    if med.empty:
        st.warning("No se pudieron calcular medianas.")
        return
    
    # Select top metrics by variance across labels
    variances = med.var(axis=0).sort_values(ascending=False)
    top_metrics = variances.head(12).index.tolist()
    med_subset = med[top_metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot with improved styling
    med_subset.T.plot(kind="bar", ax=ax, alpha=0.8, edgecolor='black', 
                      linewidth=1.2, width=0.8)
    
    ax.set_title("Comparación de medianas por label (Top 12 métricas más variables)", 
                fontsize=13, weight='bold', pad=15)
    ax.set_xlabel("Métrica", fontsize=11, weight='bold')
    ax.set_ylabel("Valor mediano", fontsize=11, weight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(title=label_col, fontsize=10, framealpha=0.95, loc='best')
    ax.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_metrics_summary(df: pd.DataFrame, label_col: str = "label") -> None:
    """Display summary statistics in Streamlit metrics cards."""
    st.markdown("#### 📊 Resumen estadístico")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total chunks", f"{len(df):,}")
    
    with col2:
        if "n_words" in df.columns:
            avg_words = df['n_words'].mean()
            st.metric("Palabras promedio", f"{avg_words:.0f}")
        elif "avg_word_len" in df.columns:
            avg_len = df['avg_word_len'].mean()
            st.metric("Long. palabra media", f"{avg_len:.2f}")
    
    with col3:
        if "flesch_reading_ease" in df.columns:
            avg_flesch = df['flesch_reading_ease'].mean()
            st.metric("Flesch medio", f"{avg_flesch:.1f}")
        elif "automated_readability_index" in df.columns:
            avg_ari = df['automated_readability_index'].mean()
            st.metric("ARI medio", f"{avg_ari:.1f}")
    
    with col4:
        if label_col in df.columns:
            label_counts = df[label_col].value_counts()
            st.metric("Labels únicos", len(label_counts))
        else:
            if "ttr" in df.columns:
                avg_ttr = df['ttr'].mean()
                st.metric("TTR medio", f"{avg_ttr:.3f}")
    
    # Show label distribution if available
    if label_col in df.columns:
        st.markdown("##### 📋 Distribución por label")
        label_counts = df[label_col].value_counts()
        label_df = pd.DataFrame({
            "Label": label_counts.index,
            "Cantidad": label_counts.values,
            "Porcentaje": (label_counts.values / len(df) * 100).round(1)
        })
        
        # Add a simple bar chart
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            fig, ax = plt.subplots(figsize=(7, 3))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            bars = ax.bar(label_df["Label"], label_df["Cantidad"], 
                         color=[colors[i % len(colors)] for i in range(len(label_df))],
                         alpha=0.8, edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=9, weight='bold')
            
            ax.set_ylabel("Cantidad de chunks", fontsize=10, weight='bold')
            ax.set_xlabel("Label", fontsize=10, weight='bold')
            ax.set_title("Distribución de chunks por label", fontsize=11, weight='bold', pad=10)
            ax.grid(alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with col_table:
            st.dataframe(label_df, use_container_width=True, hide_index=True)
