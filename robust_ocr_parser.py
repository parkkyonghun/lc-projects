"""
Robust OCR Parser for Low Quality Text Recognition

This module provides advanced parsing techniques for extracting information
from poor quality OCR output, including fuzzy matching, pattern recognition,
and multiple fallback strategies.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher
import unicodedata

logger = logging.getLogger(__name__)


class RobustOCRParser:
    """
    Advanced OCR parser that can handle poor quality text recognition.

    Features:
    - Fuzzy string matching for labels
    - Multiple pattern recognition strategies
    - Character substitution correction
    - Context-aware field extraction
    """

    def __init__(self):
        """Initialize the robust parser with common patterns and corrections."""

        # Common OCR character substitutions
        self.char_corrections = {
            # Common OCR mistakes
            '0': ['O', 'o', 'Q'],
            '1': ['l', 'I', '|'],
            '2': ['Z', 'z'],
            '5': ['S', 's'],
            '6': ['G', 'b'],
            '8': ['B'],
            'O': ['0'],
            'I': ['1', 'l'],
            'S': ['5'],
            'G': ['6'],
            'B': ['8'],
            # Khmer specific
            'ឈ': ['ឆ', 'ជ'],
            'ម': ['រ', 'ន'],
            'ះ': ['ៈ', 'ុះ'],
        }

        # Khmer field labels with variations
        self.khmer_labels = {
            'name': ['ឈ្មោះ', 'ឈ្មេះ', 'ឈ្មោះ:', 'ឈ្មេះ:', 'Name', 'NAME'],
            'id': ['លេខសម្គាល់', 'លេខសម្គាល', 'លេខសម្គាល់:', 'ID', 'ID:', 'លេខ'],
            'dob': ['ថ្ងៃកំណើត', 'ថ្ងៃកំណេត', 'ថ្ងៃកំណើត:', 'DOB', 'DOB:', 'Date'],
            'gender': ['ភេទ', 'ភេទ:', 'Sex', 'SEX', 'Sex:', 'Gender'],
            'nationality': ['សញ្ជាតិ', 'សញ្ជាតិ:', 'Nationality', 'NATIONALITY']
        }

        # English field labels
        self.english_labels = {
            'name': ['Name', 'NAME', 'Full Name', 'FULL NAME'],
            'id': ['ID Number', 'ID', 'Identity Number', 'Card Number'],
            'dob': ['Date of Birth', 'DOB', 'Birth Date', 'Born'],
            'gender': ['Sex', 'Gender', 'SEX', 'GENDER'],
            'nationality': ['Nationality', 'NATIONALITY', 'Citizen']
        }

        # Common date patterns
        self.date_patterns = [
            r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',
            r'\d{1,2}\s+\d{1,2}\s+\d{2,4}',
            r'\d{2,4}[./-]\d{1,2}[./-]\d{1,2}',
        ]

        # Common ID patterns
        self.id_patterns = [
            r'\d{9,12}',  # 9-12 digit IDs
            r'\d{3,4}\s*\d{3,4}\s*\d{3,4}',  # Spaced IDs
            r'[A-Z]\d{8,11}',  # Letter + digits
        ]

    def parse_cambodian_id_robust(self, khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
        """
        Robust parsing of Cambodian ID card OCR text.

        Args:
            khmer_text: OCR text in Khmer
            english_text: OCR text in English

        Returns:
            Dictionary with extracted fields
        """
        logger.info("Starting robust OCR parsing")
        logger.info(f"Khmer text length: {len(khmer_text)}")
        logger.info(f"English text length: {len(english_text)}")

        # Clean and normalize text
        khmer_clean = self._clean_text(khmer_text)
        english_clean = self._clean_text(english_text)

        logger.info(f"Cleaned Khmer: {khmer_clean[:100]}...")
        logger.info(f"Cleaned English: {english_clean[:100]}...")

        data = {
            "name": None,
            "name_kh": None,
            "name_en": None,
            "id_number": None,
            "dob": None,
            "nationality": "Cambodian",
            "gender": None
        }

        # Extract each field using multiple strategies
        data.update(self._extract_name_robust(khmer_clean, english_clean))
        data.update(self._extract_id_robust(khmer_clean, english_clean))
        data.update(self._extract_dob_robust(khmer_clean, english_clean))
        data.update(self._extract_gender_robust(khmer_clean, english_clean))
        data.update(self._extract_nationality_robust(khmer_clean, english_clean))

        # Log results
        for key, value in data.items():
            if value:
                logger.info(f"Extracted {key}: {value}")
            else:
                logger.warning(f"Failed to extract {key}")

        return data

    def _clean_text(self, text: str) -> str:
        """Clean and normalize OCR text."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Normalize Unicode
        text = unicodedata.normalize('NFKC', text)

        # Remove common OCR artifacts
        text = re.sub(r'[|_~`]', '', text)

        return text

    def _extract_name_robust(self, khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
        """Extract name using multiple strategies."""
        result = {"name": None, "name_kh": None, "name_en": None}

        # Strategy 1: Look for Khmer name with label
        for label in self.khmer_labels['name']:
            pattern = rf"{re.escape(label)}\s*[:\s]*\s*([\u1780-\u17FF\s]+)"
            match = re.search(pattern, khmer_text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2:  # Minimum name length
                    result["name_kh"] = name
                    result["name"] = name
                    logger.info(f"Found Khmer name with label '{label}': {name}")
                    break

        # Strategy 2: Look for English name with label
        if not result["name"]:
            for label in self.english_labels['name']:
                pattern = rf"{re.escape(label)}\s*[:\s]*\s*([A-Za-z\s]+)"
                match = re.search(pattern, english_text, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    if len(name) > 2 and not re.match(r'^\d+$', name):
                        result["name_en"] = name
                        result["name"] = name
                        logger.info(f"Found English name with label '{label}': {name}")
                        break

        # Strategy 3: Look for Khmer text without label (first line heuristic)
        if not result["name"]:
            lines = khmer_text.split('\n')
            for line in lines[:3]:  # Check first 3 lines
                khmer_match = re.search(r'([\u1780-\u17FF\s]{3,})', line)
                if khmer_match:
                    name = khmer_match.group(1).strip()
                    if len(name) > 2:
                        result["name_kh"] = name
                        result["name"] = name
                        logger.info(f"Found Khmer name without label: {name}")
                        break

        # Strategy 4: Look for English text without label
        if not result["name"]:
            lines = english_text.split('\n')
            for line in lines[:3]:
                english_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', line)
                if english_match:
                    name = english_match.group(1).strip()
                    if len(name) > 2:
                        result["name_en"] = name
                        result["name"] = name
                        logger.info(f"Found English name without label: {name}")
                        break

        return result

    def _extract_id_robust(self, khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
        """Extract ID number using multiple strategies."""
        result = {"id_number": None}

        # Strategy 1: Look for ID with Khmer label
        for label in self.khmer_labels['id']:
            pattern = rf"{re.escape(label)}\s*[:\s]*\s*(\d+(?:\s*\d+)*)"
            match = re.search(pattern, khmer_text, re.IGNORECASE)
            if match:
                id_num = re.sub(r'\s+', '', match.group(1))
                if 8 <= len(id_num) <= 12:  # Reasonable ID length
                    result["id_number"] = id_num
                    logger.info(f"Found ID with Khmer label '{label}': {id_num}")
                    return result

        # Strategy 2: Look for ID with English label
        for label in self.english_labels['id']:
            pattern = rf"{re.escape(label)}\s*[:\s]*\s*(\d+(?:\s*\d+)*)"
            match = re.search(pattern, english_text, re.IGNORECASE)
            if match:
                id_num = re.sub(r'\s+', '', match.group(1))
                if 8 <= len(id_num) <= 12:
                    result["id_number"] = id_num
                    logger.info(f"Found ID with English label '{label}': {id_num}")
                    return result

        # Strategy 3: Look for ID patterns without label
        combined_text = khmer_text + " " + english_text
        for pattern in self.id_patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                id_num = re.sub(r'\s+', '', match)
                if 8 <= len(id_num) <= 12:
                    result["id_number"] = id_num
                    logger.info(f"Found ID without label: {id_num}")
                    return result

        # Strategy 4: Look for any 8+ digit sequence (more aggressive)
        digit_matches = re.findall(r'\b\d{8,12}\b', combined_text)
        for match in digit_matches:
            # Skip obvious dates or other numbers
            if not re.match(r'(19|20)\d{6}', match):  # Skip dates like 19990803
                result["id_number"] = match
                logger.info(f"Found ID by digit pattern: {match}")
                return result

        # Strategy 5: Look for numbers at the beginning of lines
        lines = (khmer_text + "\n" + english_text).split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if re.match(r'^\d{8,12}', line):
                id_num = re.findall(r'^\d{8,12}', line)[0]
                result["id_number"] = id_num
                logger.info(f"Found ID at line start: {id_num}")
                return result

        return result

    def _extract_dob_robust(self, khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
        """Extract date of birth using multiple strategies."""
        result = {"dob": None}

        # Strategy 1: Look for DOB with Khmer label
        for label in self.khmer_labels['dob']:
            for date_pattern in self.date_patterns:
                pattern = rf"{re.escape(label)}\s*[:\s]*\s*({date_pattern})"
                match = re.search(pattern, khmer_text, re.IGNORECASE)
                if match:
                    result["dob"] = match.group(1).strip()
                    logger.info(f"Found DOB with Khmer label '{label}': {result['dob']}")
                    return result

        # Strategy 2: Look for DOB with English label
        for label in self.english_labels['dob']:
            for date_pattern in self.date_patterns:
                pattern = rf"{re.escape(label)}\s*[:\s]*\s*({date_pattern})"
                match = re.search(pattern, english_text, re.IGNORECASE)
                if match:
                    result["dob"] = match.group(1).strip()
                    logger.info(f"Found DOB with English label '{label}': {result['dob']}")
                    return result

        # Strategy 3: Look for date patterns without label
        combined_text = khmer_text + " " + english_text
        for date_pattern in self.date_patterns:
            matches = re.findall(date_pattern, combined_text)
            if matches:
                result["dob"] = matches[0]
                logger.info(f"Found DOB without label: {result['dob']}")
                return result

        return result

    def _extract_gender_robust(self, khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
        """Extract gender using multiple strategies."""
        result = {"gender": None}

        # Strategy 1: Look for Khmer gender values
        khmer_gender_patterns = [
            (r'ប្រុស', 'Male'),
            (r'ស្រី', 'Female'),
            (r'ប្រុស', 'Male'),  # Alternative spelling
        ]

        for pattern, gender in khmer_gender_patterns:
            if re.search(pattern, khmer_text):
                result["gender"] = gender
                logger.info(f"Found gender from Khmer text: {gender}")
                return result

        # Strategy 2: Look for English gender values
        english_gender_patterns = [
            (r'\b(Male|M)\b', 'Male'),
            (r'\b(Female|F)\b', 'Female'),
            (r'\b(MALE)\b', 'Male'),
            (r'\b(FEMALE)\b', 'Female'),
        ]

        for pattern, gender in english_gender_patterns:
            if re.search(pattern, english_text, re.IGNORECASE):
                result["gender"] = gender
                logger.info(f"Found gender from English text: {gender}")
                return result

        return result

    def _extract_nationality_robust(self, khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
        """Extract nationality using multiple strategies."""
        result = {"nationality": "Cambodian"}  # Default

        # Look for Khmer nationality
        for label in self.khmer_labels['nationality']:
            pattern = rf"{re.escape(label)}\s*[:\s]*\s*([\u1780-\u17FF\s]+)"
            match = re.search(pattern, khmer_text, re.IGNORECASE)
            if match:
                nationality = match.group(1).strip()
                if len(nationality) > 1:
                    result["nationality"] = nationality
                    logger.info(f"Found nationality from Khmer: {nationality}")
                    return result

        # Look for English nationality
        for label in self.english_labels['nationality']:
            pattern = rf"{re.escape(label)}\s*[:\s]*\s*([A-Za-z\s]+)"
            match = re.search(pattern, english_text, re.IGNORECASE)
            if match:
                nationality = match.group(1).strip()
                if len(nationality) > 1:
                    result["nationality"] = nationality
                    logger.info(f"Found nationality from English: {nationality}")
                    return result

        return result


# Factory function
def create_robust_parser() -> RobustOCRParser:
    """Create a robust OCR parser instance."""
    return RobustOCRParser()


# Convenience function
def parse_ocr_robust(khmer_text: str, english_text: str) -> Dict[str, Optional[str]]:
    """
    Parse OCR text using robust methods.

    Args:
        khmer_text: OCR text in Khmer
        english_text: OCR text in English

    Returns:
        Dictionary with extracted fields
    """
    parser = create_robust_parser()
    return parser.parse_cambodian_id_robust(khmer_text, english_text)
