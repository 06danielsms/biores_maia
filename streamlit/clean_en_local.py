"""Local text cleaning functions adapted from scripts/clean_en.py for Streamlit use."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# Compiled regex patterns for better performance (from scripts/clean_en.py)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
PHONE_RE = re.compile(r"\b\+?\d[\d\s\-\(\)]{6,}\d\b")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"\s+")
MULTI_DOT_RE = re.compile(r"\.{3,}")
MULTI_PUNCT_RE = re.compile(r"([!?]){2,}")


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to their canonical form.
    Uses NFKC normalization as in scripts/clean_en.py.
    """
    return unicodedata.normalize("NFKC", text)


def clean_text(text: str) -> str:
    """
    Clean text using the exact cleaning pipeline from scripts/clean_en.py.
    
    Steps (matching scripts/clean_en.py):
    1. Normalize Unicode (NFKC)
    2. Remove control characters
    3. Normalize line breaks (\r\n -> \n, \r -> \n)
    4. Collapse 3+ newlines to 2
    5. Join single line breaks between non-newline chars
    6. Remove HTML tags
    7. Replace URLs, emails, phone numbers, hashtags with space
    8. Remove leading list markers (>, *, -, •, ·, ◦)
    9. Clean up multiple dots (3+ -> ...)
    10. Clean up multiple punctuation (!!, ?? -> !, ?)
    11. Fix spacing before punctuation
    12. Normalize multiple spaces to single space
    13. Strip leading/trailing whitespace
    """
    # Step 1: Normalize Unicode
    text = normalize_unicode(text)
    
    # Step 2: Remove control characters
    text = CONTROL_CHARS_RE.sub(" ", text)
    
    # Step 3-5: Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([^\n])\n([^\n])", r"\1 \2", text)
    
    # Step 6: Remove HTML tags
    text = HTML_TAG_RE.sub(" ", text)
    
    # Step 7: Replace URLs, emails, phones, hashtags
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = HASHTAG_RE.sub(" ", text)
    
    # Step 8: Remove leading list markers
    text = re.sub(r"^[\s>*\-•·◦]+", "", text, flags=re.MULTILINE)
    
    # Step 9: Clean up multiple dots
    text = MULTI_DOT_RE.sub("...", text)
    
    # Step 10: Clean up multiple punctuation
    text = MULTI_PUNCT_RE.sub(r"\1", text)
    
    # Step 11: Fix spacing around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    
    # Step 12: Normalize multiple spaces
    text = MULTI_SPACE_RE.sub(" ", text)
    
    # Step 13: Final strip
    text = text.strip()
    
    return text


def text_to_chunks(text: str, max_words: int = 300) -> List[str]:
    """
    Split text into chunks of approximately max_words.
    Adapted from scripts/clean_en.py for local use.
    
    Args:
        text: Text to split
        max_words: Maximum words per chunk
        
    Returns:
        List of text chunks
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    for i in range(0, len(words), max_words):
        chunk_words = words[i:i + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    
    return chunks


def chunk_by_budget(
    texts: List[str], 
    max_tokens: int = 120, 
    overlap: int = 20
) -> List[Tuple[str, int]]:
    """
    Chunk texts with token budget and overlap.
    Compatible with the original interface but using improved chunking.
    
    Args:
        texts: List of texts to chunk
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of tuples (chunk_text, token_count)
    """
    all_chunks = []
    
    for text in texts:
        if not text.strip():
            continue
            
        words = text.split()
        if not words:
            continue
        
        # Simple chunking with overlap
        i = 0
        while i < len(words):
            chunk_words = words[i:i + max_tokens]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                all_chunks.append((chunk_text, len(chunk_words)))
            
            # Move forward, accounting for overlap
            i += max_tokens - overlap
            
            # Stop if we've consumed all words
            if i >= len(words):
                break
    
    return all_chunks


def normalize(text: str, config: dict) -> str:
    """
    Normalize text according to configuration.
    Maintains compatibility with original interface while using improved cleaning.
    
    Args:
        text: Text to normalize
        config: Configuration dict with cleaning options
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Apply base cleaning first (always)
    cleaned = clean_text(text)
    
    # Apply additional transformations based on config
    if config.get("lowercase", True):
        cleaned = cleaned.lower()
    
    if config.get("remove_punctuation", False):
        # Keep sentence-ending punctuation but remove others
        cleaned = re.sub(r"[^\w\s.!?]", " ", cleaned)
        cleaned = MULTI_SPACE_RE.sub(" ", cleaned)
    
    if config.get("normalize_whitespace", True):
        cleaned = MULTI_SPACE_RE.sub(" ", cleaned).strip()
    
    return cleaned


__all__ = [
    "normalize_unicode",
    "clean_text",
    "text_to_chunks",
    "chunk_by_budget",
    "normalize",
]
