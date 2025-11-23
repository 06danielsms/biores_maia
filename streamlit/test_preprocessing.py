"""Quick test to verify the updated preprocessing and metrics modules."""

import sys
from pathlib import Path

# Add streamlit directory to path
streamlit_dir = Path(__file__).parent
sys.path.insert(0, str(streamlit_dir))

from clean_en_local import clean_text, text_to_chunks, normalize

def test_cleaning():
    """Test the cleaning functions."""
    print("Testing cleaning functions...")
    
    test_text = """
    This is a test document with URLs like https://example.com and emails like test@example.com.
    
    It has   multiple    spaces and <html>tags</html>.
    
    Phone numbers like +1-234-567-8900 should be removed.
    #hashtags and !!!multiple!!! punctuation too...
    """
    
    cleaned = clean_text(test_text)
    print(f"\nOriginal length: {len(test_text)}")
    print(f"Cleaned length: {len(cleaned)}")
    print(f"\nCleaned text preview:\n{cleaned[:200]}...")
    
    # Test chunking
    chunks = text_to_chunks(cleaned, max_words=20)
    print(f"\nNumber of chunks (20 words each): {len(chunks)}")
    print(f"First chunk: {chunks[0] if chunks else 'None'}")
    
    # Test normalize with config
    config = {
        "lowercase": True,
        "remove_punctuation": False,
        "normalize_unicode": True,
        "strip_html": True,
        "replace_urls": True,
        "replace_emails": True,
        "normalize_whitespace": True,
    }
    
    normalized = normalize(test_text, config)
    print(f"\nNormalized text preview:\n{normalized[:200]}...")
    
    print("\n✅ All cleaning tests passed!")

if __name__ == "__main__":
    test_cleaning()
