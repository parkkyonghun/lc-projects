import os
import uuid
import aiofiles
from typing import List, Dict, Optional
from fastapi import UploadFile, HTTPException
from PIL import Image
import magic
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class FileService:
    """Service for handling file uploads and management"""
    
    def __init__(self):
        self.upload_dir = settings.upload_directory
        self.max_file_size = settings.max_file_size
        self.allowed_types = settings.allowed_file_types
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Create subdirectories for different file types
        for subdir in ['documents', 'id_cards', 'profile_photos']:
            os.makedirs(os.path.join(self.upload_dir, subdir), exist_ok=True)
    
    async def upload_file(
        self, 
        file: UploadFile, 
        file_type: str = "documents",
        user_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Upload a file and return file information"""
        
        # Validate file
        await self._validate_file(file)
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        
        # Determine file path
        if user_id:
            file_path = os.path.join(self.upload_dir, file_type, user_id)
            os.makedirs(file_path, exist_ok=True)
            full_path = os.path.join(file_path, unique_filename)
        else:
            full_path = os.path.join(self.upload_dir, file_type, unique_filename)
        
        try:
            # Save file
            async with aiofiles.open(full_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # If it's an image, create thumbnail
            thumbnail_path = None
            if file.content_type.startswith('image/'):
                thumbnail_path = await self._create_thumbnail(full_path)
            
            logger.info(f"File uploaded successfully: {full_path}")
            
            return {
                "filename": unique_filename,
                "original_filename": file.filename,
                "file_path": full_path,
                "thumbnail_path": thumbnail_path,
                "file_type": file_type,
                "content_type": file.content_type,
                "file_size": len(content)
            }
            
        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    async def upload_multiple_files(
        self, 
        files: List[UploadFile], 
        file_type: str = "documents",
        user_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Upload multiple files"""
        results = []
        
        for file in files:
            try:
                result = await self.upload_file(file, file_type, user_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to upload file {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "status": "failed"
                })
        
        return results
    
    async def _validate_file(self, file: UploadFile):
        """Validate uploaded file"""
        # Check file size
        content = await file.read()
        await file.seek(0)  # Reset file pointer
        
        if len(content) > self.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size is {self.max_file_size / (1024*1024):.1f}MB"
            )
        
        # Check file type
        if file.content_type not in self.allowed_types:
            raise HTTPException(
                status_code=415, 
                detail=f"File type not allowed. Allowed types: {', '.join(self.allowed_types)}"
            )
        
        # Additional validation for images
        if file.content_type.startswith('image/'):
            try:
                # Verify it's a valid image
                image = Image.open(file.file)
                image.verify()
                await file.seek(0)  # Reset file pointer
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image file")
    
    async def _create_thumbnail(self, image_path: str, size: tuple = (200, 200)) -> Optional[str]:
        """Create thumbnail for image files"""
        try:
            # Generate thumbnail path
            base_path, ext = os.path.splitext(image_path)
            thumbnail_path = f"{base_path}_thumb{ext}"
            
            # Create thumbnail
            with Image.open(image_path) as image:
                # Convert to RGB if necessary
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                
                # Create thumbnail
                image.thumbnail(size, Image.Resampling.LANCZOS)
                image.save(thumbnail_path, optimize=True, quality=85)
            
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {str(e)}")
            return None
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                
                # Also delete thumbnail if exists
                base_path, ext = os.path.splitext(file_path)
                thumbnail_path = f"{base_path}_thumb{ext}"
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)
                
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False
    
    async def get_file_info(self, file_path: str) -> Optional[Dict[str, any]]:
        """Get file information"""
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            
            # Get MIME type
            mime_type = magic.from_file(file_path, mime=True)
            
            return {
                "file_path": file_path,
                "filename": os.path.basename(file_path),
                "file_size": stat.st_size,
                "content_type": mime_type,
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {str(e)}")
            return None
    
    async def cleanup_old_files(self, days_old: int = 30) -> int:
        """Clean up files older than specified days"""
        import time
        
        deleted_count = 0
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        try:
            for root, dirs, files in os.walk(self.upload_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) < cutoff_time:
                        if await self.delete_file(file_path):
                            deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {str(e)}")
            return 0

# Global file service instance
file_service = FileService()