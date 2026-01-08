"""
Query normalization and typo handling for robust query processing
"""
from typing import List, Set
import re
from difflib import SequenceMatcher


class QueryNormalizer:
    """
    Normalizes queries to handle typos, spelling mistakes, and terminology variations
    """
    
    # Common typos and corrections for technical terms
    COMMON_TYPOS = {
        # Technical terms
        'guidlines': 'guidelines',
        'guidline': 'guideline',
        'standards': 'standard',  # Keep plural forms
        'requirment': 'requirement',
        'requirments': 'requirements',
        'procedures': 'procedure',
        'procedur': 'procedure',
        'temprature': 'temperature',
        'tempature': 'temperature',
        'hygiene': 'hygiene',  # Common misspelling
        'hygene': 'hygiene',
        'certificates': 'certificate',
        'certificat': 'certificate',
        'certifcate': 'certificate',
        'descripton': 'description',
        'descriptin': 'description',
        'reports': 'report',
        'repots': 'reports',
        'storage': 'storage',
        'storag': 'storage',
        'monitoring': 'monitor',
        'monitering': 'monitoring',
        'maintainance': 'maintenance',
        'maintainence': 'maintenance',
        'sanitization': 'sanitize',
        'sanitisation': 'sanitization',
    }
    
    # Common word typos (single character errors)
    COMMON_WORD_TYPOS = {
        'teh': 'the',
        'adn': 'and',
        'taht': 'that',
        'whta': 'what',
        'shoudl': 'should',
        'shoul': 'should',
        'shold': 'should',
        'sholud': 'should',
        'woudl': 'would',
        'wolud': 'would',
        'wold': 'would',
    }
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalize query to handle typos and common mistakes
        
        Args:
            query: Original query text
            
        Returns:
            Normalized query text
        """
        # Convert to lowercase for processing
        query_lower = query.lower()
        
        # Split into words
        words = re.findall(r'\b[a-z0-9]+\b', query_lower)
        
        # Fix common typos
        corrected_words = []
        for word in words:
            # Check common typos first
            if word in QueryNormalizer.COMMON_TYPOS:
                corrected_words.append(QueryNormalizer.COMMON_TYPOS[word])
            elif word in QueryNormalizer.COMMON_WORD_TYPOS:
                corrected_words.append(QueryNormalizer.COMMON_WORD_TYPOS[word])
            else:
                # Try fuzzy matching for similar words (handle single character typos)
                corrected_word = QueryNormalizer._fuzzy_correct(word)
                corrected_words.append(corrected_word)
        
        # Reconstruct query preserving original structure
        normalized = query
        for i, word in enumerate(words):
            if word != corrected_words[i]:
                # Replace word in original query (case-insensitive)
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                normalized = pattern.sub(corrected_words[i], normalized, count=1)
        
        return normalized
    
    @staticmethod
    def _fuzzy_correct(word: str, threshold: float = 0.8) -> str:
        """
        Try to correct word using fuzzy matching against known terms
        
        Args:
            word: Word to correct
            threshold: Similarity threshold (0-1)
            
        Returns:
            Corrected word or original if no good match
        """
        # Combine all known terms
        known_terms = set(QueryNormalizer.COMMON_TYPOS.keys())
        known_terms.update(QueryNormalizer.COMMON_TYPOS.values())
        known_terms.update(QueryNormalizer.COMMON_WORD_TYPOS.keys())
        known_terms.update(QueryNormalizer.COMMON_WORD_TYPOS.values())
        
        # Add common technical terms
        known_terms.update([
            'guideline', 'guidelines', 'standard', 'standards',
            'requirement', 'requirements', 'procedure', 'procedures',
            'temperature', 'hygiene', 'certificate', 'certificates',
            'description', 'report', 'reports', 'storage', 'monitoring',
            'maintenance', 'sanitization', 'safety', 'food', 'product'
        ])
        
        # Find best match
        best_match = word
        best_score = 0.0
        
        for term in known_terms:
            # Skip if lengths are very different
            if abs(len(word) - len(term)) > 2:
                continue
            
            # Calculate similarity
            similarity = SequenceMatcher(None, word, term).ratio()
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = term
        
        return best_match
    
    @staticmethod
    def generate_fuzzy_keywords(query: str) -> List[str]:
        """
        Generate fuzzy keyword variations for better matching
        
        Args:
            query: Query text
            
        Returns:
            List of keyword variations including original
        """
        keywords = []
        query_lower = query.lower()
        words = re.findall(r'\b[a-z0-9]{3,}\b', query_lower)
        
        # Add original words
        keywords.extend(words)
        
        # Add normalized words
        normalized = QueryNormalizer.normalize_query(query)
        normalized_words = re.findall(r'\b[a-z0-9]{3,}\b', normalized.lower())
        keywords.extend([w for w in normalized_words if w not in keywords])
        
        # Generate variations with common typos (for matching)
        # This helps match documents even if they have typos
        for word in words:
            # Add common variations
            if len(word) > 4:
                # Try single character variations (for fuzzy matching)
                variations = []
                # Character swaps
                for i in range(len(word) - 1):
                    chars = list(word)
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                    variations.append(''.join(chars))
                # Character deletions
                for i in range(len(word)):
                    variations.append(word[:i] + word[i+1:])
                # Character insertions (common letters)
                for char in 'aeiou':
                    for i in range(len(word) + 1):
                        variations.append(word[:i] + char + word[i:])
                
                # Add variations that are close matches
                for var in variations[:3]:  # Limit to top 3
                    if var not in keywords and len(var) >= 3:
                        keywords.append(var)
        
        return list(set(keywords))  # Remove duplicates
    
    @staticmethod
    def expand_terminology(query: str) -> str:
        """
        Expand query with common terminology variations
        
        Args:
            query: Query text
            
        Returns:
            Expanded query with terminology variations
        """
        from app.core.rag.query_understanding import QueryUnderstanding
        
        # Normalize first
        normalized = QueryNormalizer.normalize_query(query)
        
        # Get synonyms
        query_lower = normalized.lower()
        words = re.findall(r'\b[a-z0-9]+\b', query_lower)
        
        expanded_terms = []
        for word in words:
            expanded_terms.append(word)
            # Add synonyms if available
            if word in QueryUnderstanding.SYNONYMS:
                expanded_terms.extend(QueryUnderstanding.SYNONYMS[word][:2])
        
        # Reconstruct query with expanded terms
        # For now, just return normalized - expansion happens in query variations
        return normalized

