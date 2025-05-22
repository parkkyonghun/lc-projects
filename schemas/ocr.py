from pydantic import BaseModel
from typing import Optional

class CambodianIDCardOCRResult(BaseModel):
    # Raw OCR results
    full_name: Optional[str] = None  # Combined text
    raw_khmer: Optional[str] = None  # Raw Khmer OCR result
    raw_english: Optional[str] = None  # Raw English OCR result
    
    # Parsed fields
    id_number: Optional[str] = None
    name_kh: Optional[str] = None # Specific Khmer name if extracted
    name_en: Optional[str] = None # Specific English name if extracted (usually fallback)
    date_of_birth: Optional[str] = None # Renamed from birth_date
    gender: Optional[str] = None # Renamed from sex
    nationality: Optional[str] = None # Added field
    height: Optional[str] = None
    birth_place: Optional[str] = None
    address: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    description: Optional[str] = None
