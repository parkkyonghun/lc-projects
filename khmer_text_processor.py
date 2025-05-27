#!/usr/bin/env python3
"""
Khmer Text Processor for OCR Enhancement

This module provides advanced Khmer text processing capabilities including
normalization, validation, and correction specifically for Cambodian ID cards.

Features:
- Unicode normalization for Khmer text
- Character validation and correction
- Word segmentation and tokenization
- OCR error detection and correction
- Integration with Khmer language models
"""

import re
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import sqlite3

logger = logging.getLogger(__name__)


class KhmerTextProcessor:
    """
    Advanced Khmer text processor for OCR enhancement.
    
    This class provides comprehensive text processing capabilities
    specifically designed for Khmer script and Cambodian ID cards.
    """
    
    def __init__(self, resources_dir: str = "khmer_resources"):
        """
        Initialize the Khmer text processor.
        
        Args:
            resources_dir: Directory containing Khmer resources
        """
        self.resources_dir = Path(resources_dir)
        self.db_path = self.resources_dir / "khmer_resources.db"
        
        # Khmer Unicode ranges
        self.khmer_range = (0x1780, 0x17FF)  # Main Khmer block
        self.khmer_symbols_range = (0x19E0, 0x19FF)  # Khmer symbols
        
        # Load normalization rules
        self.normalization_rules = self._load_normalization_rules()
        
        # Common Khmer characters for validation
        self.khmer_consonants = set(chr(i) for i in range(0x1780, 0x17A3))
        self.khmer_vowels = set(chr(i) for i in range(0x17A3, 0x17B4))
        self.khmer_vowels.update(chr(i) for i in range(0x17B6, 0x17D4))
        
        # Common OCR errors in Khmer
        self.common_errors = {
            # Common character confusions
            'ក': ['គ', 'ខ'],
            'ង': ['ញ'],
            'ច': ['ជ'],
            'ត': ['ទ', 'ថ'],
            'ន': ['ណ'],
            'ប': ['ព', 'ភ'],
            'ម': ['ម្'],
            'រ': ['ល'],
            'ស': ['ហ'],
            # Vowel confusions
            'ា': ['ាំ'],
            'ិ': ['ី'],
            'ុ': ['ូ'],
            'េ': ['ែ'],
            'ោ': ['ៅ']
        }
        
        logger.info("Khmer text processor initialized")
    
    def _load_normalization_rules(self) -> List[Dict[str, Any]]:
        """Load normalization rules from database."""
        if not self.db_path.exists():
            return []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT rule_name, rule_type, input_pattern, output_pattern, priority
                FROM normalization_rules 
                WHERE is_active = 1 
                ORDER BY priority DESC
            """)
            
            rules = []
            for row in cursor.fetchall():
                rules.append({
                    'name': row[0],
                    'type': row[1],
                    'input': row[2],
                    'output': row[3],
                    'priority': row[4]
                })
            
            conn.close()
            return rules
            
        except Exception as e:
            logger.error(f"Failed to load normalization rules: {e}")
            return []
    
    def normalize_khmer_text(self, text: str) -> str:
        """
        Normalize Khmer text for better OCR processing.
        
        Args:
            text: Raw Khmer text
            
        Returns:
            Normalized Khmer text
        """
        if not text:
            return text
        
        # Step 1: Unicode normalization
        normalized = unicodedata.normalize('NFC', text)
        
        # Step 2: Remove zero-width characters
        normalized = normalized.replace('\u200B', '')  # Zero Width Space
        normalized = normalized.replace('\u200C', '')  # Zero Width Non-Joiner
        normalized = normalized.replace('\u200D', '')  # Zero Width Joiner
        
        # Step 3: Apply custom normalization rules
        for rule in self.normalization_rules:
            if rule['type'] == 'cleanup':
                normalized = re.sub(rule['input'], rule['output'], normalized)
            elif rule['type'] == 'reordering':
                normalized = re.sub(rule['input'], rule['output'], normalized)
        
        # Step 4: Standardize spacing
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        
        # Step 5: Fix common character ordering issues
        normalized = self._fix_character_ordering(normalized)
        
        return normalized
    
    def _fix_character_ordering(self, text: str) -> str:
        """Fix common Khmer character ordering issues."""
        # Fix vowel ordering (some vowels should come before consonants)
        # This is a simplified version - real implementation would be more complex
        
        # Fix COENG (subscript consonant marker) positioning
        text = re.sub(r'(\u17D2)(\s+)', r'\1', text)
        
        # Fix vowel sign positioning
        text = re.sub(r'(\u17B6)(\u17C6)', r'\2\1', text)  # SIGN AA + NIKAHIT
        
        return text
    
    def validate_khmer_text(self, text: str) -> Dict[str, Any]:
        """
        Validate Khmer text and identify potential issues.
        
        Args:
            text: Text to validate
            
        Returns:
            Validation results with issues and suggestions
        """
        results = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'confidence': 1.0,
            'character_count': len(text),
            'khmer_character_count': 0
        }
        
        if not text:
            results['is_valid'] = False
            results['issues'].append('Empty text')
            return results
        
        # Count Khmer characters
        khmer_chars = 0
        for char in text:
            code_point = ord(char)
            if (self.khmer_range[0] <= code_point <= self.khmer_range[1] or
                self.khmer_symbols_range[0] <= code_point <= self.khmer_symbols_range[1]):
                khmer_chars += 1
        
        results['khmer_character_count'] = khmer_chars
        
        # Check if text contains sufficient Khmer content
        if khmer_chars == 0:
            results['is_valid'] = False
            results['issues'].append('No Khmer characters found')
            results['confidence'] = 0.0
        elif khmer_chars / len(text) < 0.3:
            results['issues'].append('Low Khmer character ratio')
            results['confidence'] *= 0.7
        
        # Check for invalid character sequences
        invalid_sequences = self._find_invalid_sequences(text)
        if invalid_sequences:
            results['issues'].extend(invalid_sequences)
            results['confidence'] *= 0.8
        
        # Check for OCR artifacts
        ocr_artifacts = self._detect_ocr_artifacts(text)
        if ocr_artifacts:
            results['issues'].extend(ocr_artifacts)
            results['suggestions'].extend(self._suggest_corrections(text))
            results['confidence'] *= 0.9
        
        return results
    
    def _find_invalid_sequences(self, text: str) -> List[str]:
        """Find invalid Khmer character sequences."""
        issues = []
        
        # Check for invalid consonant clusters
        # This is a simplified check - real implementation would be more comprehensive
        invalid_patterns = [
            r'\u17D2{2,}',  # Multiple COENG characters
            r'[\u17B6-\u17D3]{3,}',  # Too many consecutive vowels
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, text):
                issues.append(f'Invalid character sequence found: {pattern}')
        
        return issues
    
    def _detect_ocr_artifacts(self, text: str) -> List[str]:
        """Detect common OCR artifacts in Khmer text."""
        artifacts = []
        
        # Check for mixed scripts that shouldn't be together
        if re.search(r'[a-zA-Z][\u1780-\u17FF]', text):
            artifacts.append('Mixed Latin-Khmer characters detected')
        
        # Check for unusual character combinations
        if re.search(r'[\u1780-\u17FF][0-9][\u1780-\u17FF]', text):
            artifacts.append('Numbers embedded in Khmer text')
        
        # Check for repeated characters (common OCR error)
        if re.search(r'(.)\1{3,}', text):
            artifacts.append('Repeated characters detected')
        
        return artifacts
    
    def _suggest_corrections(self, text: str) -> List[str]:
        """Suggest corrections for common OCR errors."""
        suggestions = []
        
        # Suggest corrections based on common errors
        for correct_char, error_chars in self.common_errors.items():
            for error_char in error_chars:
                if error_char in text:
                    suggestions.append(f'Consider replacing "{error_char}" with "{correct_char}"')
        
        return suggestions
    
    def segment_khmer_words(self, text: str) -> List[str]:
        """
        Segment Khmer text into words.
        
        This is a simplified implementation. In production, you would use
        a proper Khmer word segmentation library.
        
        Args:
            text: Khmer text to segment
            
        Returns:
            List of segmented words
        """
        # Normalize first
        normalized = self.normalize_khmer_text(text)
        
        # Simple segmentation based on spaces and punctuation
        # Real implementation would use machine learning models
        words = re.split(r'[\s\u17D4\u17D5\u17D6\u17D7\u17D8\u17D9\u17DA]+', normalized)
        
        # Filter out empty strings
        words = [word.strip() for word in words if word.strip()]
        
        return words
    
    def correct_ocr_errors(self, text: str, confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Attempt to correct common OCR errors in Khmer text.
        
        Args:
            text: Text with potential OCR errors
            confidence_threshold: Minimum confidence for corrections
            
        Returns:
            Correction results with original and corrected text
        """
        results = {
            'original_text': text,
            'corrected_text': text,
            'corrections_made': [],
            'confidence': 1.0
        }
        
        if not text:
            return results
        
        corrected = text
        
        # Apply common character corrections
        for correct_char, error_chars in self.common_errors.items():
            for error_char in error_chars:
                if error_char in corrected:
                    # Simple replacement - in production, use context-aware correction
                    old_text = corrected
                    corrected = corrected.replace(error_char, correct_char)
                    if corrected != old_text:
                        results['corrections_made'].append({
                            'type': 'character_replacement',
                            'from': error_char,
                            'to': correct_char,
                            'confidence': 0.7
                        })
        
        # Remove obvious artifacts
        corrected = re.sub(r'[^\u1780-\u17FF\u19E0-\u19FF\s\d\u17D4-\u17DA]', '', corrected)
        
        # Normalize the corrected text
        corrected = self.normalize_khmer_text(corrected)
        
        results['corrected_text'] = corrected
        
        # Calculate overall confidence
        if results['corrections_made']:
            avg_confidence = sum(c['confidence'] for c in results['corrections_made']) / len(results['corrections_made'])
            results['confidence'] = avg_confidence
        
        return results
    
    def extract_khmer_fields(self, text: str) -> Dict[str, str]:
        """
        Extract structured fields from Khmer text (for ID cards).
        
        Args:
            text: Raw OCR text
            
        Returns:
            Extracted fields
        """
        # Normalize text first
        normalized = self.normalize_khmer_text(text)
        
        fields = {}
        
        # Common Khmer field patterns for ID cards
        patterns = {
            'name_kh': r'ឈ្មោះ[:\s]*([^\n\r]+)',
            'nationality': r'សញ្ជាតិ[:\s]*([^\n\r]+)',
            'place_of_birth': r'ទីកន្លែងកំណើត[:\s]*([^\n\r]+)',
            'address': r'អាសយដ្ឋាន[:\s]*([^\n\r]+)'
        }
        
        for field_name, pattern in patterns.items():
            match = re.search(pattern, normalized)
            if match:
                fields[field_name] = match.group(1).strip()
        
        return fields


def create_khmer_processor(resources_dir: str = "khmer_resources") -> KhmerTextProcessor:
    """Create a Khmer text processor instance."""
    return KhmerTextProcessor(resources_dir)


def process_khmer_text_quick(text: str) -> Dict[str, Any]:
    """Quick Khmer text processing for testing."""
    processor = create_khmer_processor()
    
    return {
        'normalized': processor.normalize_khmer_text(text),
        'validation': processor.validate_khmer_text(text),
        'words': processor.segment_khmer_words(text),
        'corrections': processor.correct_ocr_errors(text),
        'fields': processor.extract_khmer_fields(text)
    }


if __name__ == "__main__":
    # Test the processor
    test_text = "ឈ្មោះ: សុខ សុភាព"
    
    print("🇰🇭 Khmer Text Processor Test")
    print("=" * 50)
    print(f"Input: {test_text}")
    
    results = process_khmer_text_quick(test_text)
    
    print(f"Normalized: {results['normalized']}")
    print(f"Validation: {results['validation']}")
    print(f"Words: {results['words']}")
    print(f"Fields: {results['fields']}")
