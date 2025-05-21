from fastapi import APIRouter, UploadFile, File, HTTPException
from controllers.ocr_controller import process_cambodian_id_ocr
from schemas.ocr import CambodianIDCardOCRResult

router = APIRouter(prefix="/ocr", tags=["ocr"])

@router.post("/idcard", response_model=CambodianIDCardOCRResult)
async def ocr_idcard(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    result = await process_cambodian_id_ocr(file)
    return result
