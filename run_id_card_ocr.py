import asyncio
from fastapi import UploadFile
from controllers.ocr_controller import process_cambodian_id_ocr
from schemas.ocr import CambodianIDCardOCRResult
from PIL import Image
import io

class SimpleUploadFile:
    def __init__(self, filename, content_type, content_bytes):
        self.filename = filename
        self.content_type = content_type
        self._content_bytes = content_bytes
    async def read(self):
        return self._content_bytes

async def main():
    image_path = 'scan_id.jpg'
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    upload_file = SimpleUploadFile(
        filename=image_path,
        content_type='image/jpeg',
        content_bytes=image_bytes
    )
    result: CambodianIDCardOCRResult = await process_cambodian_id_ocr(upload_file)
    print('--- Cambodian ID Card OCR Result ---')
    print(result.model_dump_json(indent=2))

if __name__ == '__main__':
    asyncio.run(main())
