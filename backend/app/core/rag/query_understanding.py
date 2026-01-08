"""
Query understanding and expansion for wider scope retrieval
"""
from typing import List, Set
import re


class QueryUnderstanding:
    """
    Service for understanding queries and generating variations
    to cast a wider net in retrieval
    """
    
    # Common synonyms and related terms for technical standards
    SYNONYMS = {
        'guideline': ['guideline', 'guidance', 'standard', 'procedure', 'protocol', 'rule', 'requirement'],
        'standard': ['standard', 'specification', 'requirement', 'criterion', 'benchmark'],
        'storage': ['storage', 'storing', 'preservation', 'keeping', 'holding'],
        'temperature': ['temperature', 'temp', 'thermal', 'heat', 'cold'],
        'food': ['food', 'product', 'item', 'material', 'ingredient'],
        'safety': ['safety', 'safe', 'secure', 'protection', 'precaution'],
        'hygiene': ['hygiene', 'cleanliness', 'sanitation', 'clean', 'sanitary'],
        'procedure': ['procedure', 'process', 'method', 'step', 'workflow', 'protocol'],
        'requirement': ['requirement', 'requirement', 'mandate', 'specification', 'standard'],
        'documentation': ['documentation', 'document', 'record', 'file', 'paper'],
        'check': ['check', 'verify', 'inspect', 'examine', 'validate', 'confirm'],
        'maintain': ['maintain', 'maintenance', 'keep', 'preserve', 'sustain'],
        'clean': ['clean', 'cleaning', 'sanitize', 'sanitization', 'disinfect'],
        'monitor': ['monitor', 'monitoring', 'track', 'observe', 'watch', 'supervise'],
    }
    
    # Question word expansions
    QUESTION_EXPANSIONS = {
        'what': ['what', 'which', 'describe', 'explain', 'tell me about'],
        'how': ['how', 'method', 'way', 'process', 'procedure'],
        'when': ['when', 'time', 'schedule', 'timing'],
        'where': ['where', 'location', 'place', 'position'],
        'why': ['why', 'reason', 'purpose', 'rationale'],
        'who': ['who', 'person', 'responsible', 'personnel'],
    }
    
    @staticmethod
    def generate_query_variations(query: str) -> List[str]:
        """
        Generate multiple query variations to cast a wider net
        
        Args:
            query: Original query text
            
        Returns:
            List of query variations including original
        """
        variations = [query]  # Always include original
        
        query_lower = query.lower()
        words = re.findall(r'\b[a-z0-9]+\b', query_lower)
        
        # Variation 1: Replace words with synonyms
        synonym_variations = []
        for word in words:
            if word in QueryUnderstanding.SYNONYMS:
                for synonym in QueryUnderstanding.SYNONYMS[word][:2]:  # Top 2 synonyms
                    if synonym != word:
                        variation = query_lower.replace(word, synonym)
                        if variation not in variations:
                            synonym_variations.append(variation)
        
        # Add top synonym variations
        variations.extend(synonym_variations[:3])
        
        # Variation 2: Expand question words
        question_word = None
        for qw in ['what', 'how', 'when', 'where', 'why', 'who']:
            if query_lower.startswith(qw):
                question_word = qw
                break
        
        if question_word and question_word in QueryUnderstanding.QUESTION_EXPANSIONS:
            for expansion in QueryUnderstanding.QUESTION_EXPANSIONS[question_word][:2]:
                if expansion != question_word:
                    variation = query_lower.replace(question_word, expansion, 1)
                    if variation not in variations:
                        variations.append(variation)
        
        # Variation 3: Remove question words and keep core terms
        core_terms = [w for w in words if w not in ['what', 'how', 'when', 'where', 'why', 'who', 'is', 'are', 'was', 'were', 'the', 'a', 'an']]
        if len(core_terms) >= 2:
            core_query = ' '.join(core_terms)
            if core_query not in variations and len(core_query) > 5:
                variations.append(core_query)
        
        # Variation 4: Add related technical terms
        technical_terms = []
        if any(term in query_lower for term in ['storage', 'store', 'storing']):
            technical_terms.extend(['storage conditions', 'storage requirements', 'storage guidelines'])
        if any(term in query_lower for term in ['temperature', 'temp']):
            technical_terms.extend(['temperature control', 'temperature monitoring', 'temperature requirements'])
        if any(term in query_lower for term in ['food', 'product']):
            technical_terms.extend(['food safety', 'product handling', 'food storage'])
        
        for term in technical_terms[:2]:  # Top 2 related terms
            if term not in variations:
                variations.append(term)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)
        
        # Limit to 5 variations total (including original)
        return unique_variations[:5]
    
    @staticmethod
    def extract_key_concepts(query: str) -> Set[str]:
        """
        Extract key concepts from query for broader matching
        
        Args:
            query: Query text
            
        Returns:
            Set of key concept terms
        """
        query_lower = query.lower()
        words = re.findall(r'\b[a-z0-9]{3,}\b', query_lower)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'what', 'are', 'were', 'was', 'is', 'how', 'when',
            'where', 'why', 'who', 'this', 'that', 'these', 'those', 'from', 'into'
        }
        
        concepts = {w for w in words if w not in stop_words}
        
        # Add synonyms for key concepts
        expanded_concepts = set(concepts)
        for concept in concepts:
            if concept in QueryUnderstanding.SYNONYMS:
                expanded_concepts.update(QueryUnderstanding.SYNONYMS[concept][:2])
        
        return expanded_concepts

