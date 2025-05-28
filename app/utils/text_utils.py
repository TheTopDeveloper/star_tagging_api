import re
import string
from typing import List

def clean_text(text: str) -> str:
    """
    Clean and normalize text by:
    1. Removing extra whitespace
    2. Normalizing line endings
    3. Removing special characters
    4. Converting to lowercase
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove special characters but keep basic punctuation
    text = ''.join(char for char in text if char in string.printable)
    
    # Convert to lowercase
    text = text.lower()
    
    return text.strip()

def filter_keywords(keywords: List[str], max_keywords: int) -> List[str]:
    """
    Filter and clean keywords:
    1. Remove duplicates
    2. Remove very short keywords
    3. Remove common stop words
    4. Limit to max_keywords
    """
    if not keywords:
        return []
    
    # Common stop words to remove
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'shall', 'should', 'may', 'might', 'must', 'can', 'could'
    }
    
    # Clean and filter keywords
    filtered = []
    seen = set()
    
    for keyword in keywords:
        # Clean the keyword
        keyword = clean_text(keyword)
        
        # Skip if empty, too short, or a stop word
        if (not keyword or 
            len(keyword) < 3 or 
            keyword in stop_words or 
            keyword in seen):
            continue
        
        seen.add(keyword)
        filtered.append(keyword)
        
        # Stop if we have enough keywords
        if len(filtered) >= max_keywords:
            break
    
    return filtered

def is_valid_input(text: str, min_length: int = 50) -> bool:
    """
    Check if the input text is valid:
    1. Not empty
    2. Not too short
    3. Contains actual content
    """
    if not text:
        return False
    
    # Clean the text
    cleaned = clean_text(text)
    
    # Check length
    if len(cleaned) < min_length:
        return False
    
    # Check if it's just whitespace or special characters
    if not any(c.isalnum() for c in cleaned):
        return False
    
    return True 