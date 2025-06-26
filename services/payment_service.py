from typing import Dict, Optional, Any
from datetime import datetime
import asyncio
import aiohttp
from core.config import settings
from schemas.payment import PaymentCreate, PaymentSchema
from models.payment import Payment
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import logging

logger = logging.getLogger(__name__)

class PaymentGatewayError(Exception):
    """Custom exception for payment gateway errors"""
    pass

class PaymentService:
    """Service for handling payment processing with multiple gateways"""
    
    def __init__(self):
        self.momo_api_key = settings.momo_api_key
        self.zalopay_api_key = settings.zalopay_api_key
    
    async def process_payment(
        self, 
        payment_data: PaymentCreate, 
        gateway: str = "momo",
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Process payment through specified gateway"""
        
        try:
            if gateway.lower() == "momo":
                result = await self._process_momo_payment(payment_data)
            elif gateway.lower() == "zalopay":
                result = await self._process_zalopay_payment(payment_data)
            else:
                raise PaymentGatewayError(f"Unsupported payment gateway: {gateway}")
            
            # Save payment record to database if successful
            if result.get("status") == "success" and db:
                await self._save_payment_record(payment_data, result, db)
            
            return result
            
        except Exception as e:
            logger.error(f"Payment processing failed: {str(e)}")
            raise PaymentGatewayError(f"Payment processing failed: {str(e)}")
    
    async def _process_momo_payment(self, payment_data: PaymentCreate) -> Dict[str, Any]:
        """Process payment through MOMO gateway"""
        if not self.momo_api_key:
            raise PaymentGatewayError("MOMO API key not configured")
        
        # Mock MOMO API integration
        # In production, replace with actual MOMO API calls
        momo_payload = {
            "amount": payment_data.amount,
            "currency": "KHR",  # Cambodian Riel
            "description": f"Loan payment for loan {payment_data.loanId}",
            "reference_id": str(uuid.uuid4()),
            "api_key": self.momo_api_key
        }
        
        # Simulate API call delay
        await asyncio.sleep(1)
        
        # Mock successful response
        return {
            "status": "success",
            "transaction_id": f"momo_{uuid.uuid4().hex[:12]}",
            "gateway": "momo",
            "amount": payment_data.amount,
            "currency": "KHR",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_zalopay_payment(self, payment_data: PaymentCreate) -> Dict[str, Any]:
        """Process payment through ZaloPay gateway"""
        if not self.zalopay_api_key:
            raise PaymentGatewayError("ZaloPay API key not configured")
        
        # Mock ZaloPay API integration
        # In production, replace with actual ZaloPay API calls
        zalopay_payload = {
            "amount": payment_data.amount,
            "currency": "KHR",
            "description": f"Loan payment for loan {payment_data.loanId}",
            "reference_id": str(uuid.uuid4()),
            "api_key": self.zalopay_api_key
        }
        
        # Simulate API call delay
        await asyncio.sleep(1)
        
        # Mock successful response
        return {
            "status": "success",
            "transaction_id": f"zalo_{uuid.uuid4().hex[:12]}",
            "gateway": "zalopay",
            "amount": payment_data.amount,
            "currency": "KHR",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _save_payment_record(
        self, 
        payment_data: PaymentCreate, 
        gateway_response: Dict[str, Any], 
        db: AsyncSession
    ):
        """Save payment record to database"""
        payment = Payment(
            id=str(uuid.uuid4()),
            loanId=payment_data.loanId,
            amount=payment_data.amount,
            paymentMethod=gateway_response.get("gateway", "unknown"),
            transactionId=gateway_response.get("transaction_id"),
            status="completed",
            paymentDate=datetime.utcnow()
        )
        
        db.add(payment)
        await db.commit()
        await db.refresh(payment)
        
        return payment
    
    async def verify_payment(self, transaction_id: str, gateway: str) -> Dict[str, Any]:
        """Verify payment status with gateway"""
        try:
            if gateway.lower() == "momo":
                return await self._verify_momo_payment(transaction_id)
            elif gateway.lower() == "zalopay":
                return await self._verify_zalopay_payment(transaction_id)
            else:
                raise PaymentGatewayError(f"Unsupported gateway for verification: {gateway}")
        except Exception as e:
            logger.error(f"Payment verification failed: {str(e)}")
            raise PaymentGatewayError(f"Payment verification failed: {str(e)}")
    
    async def _verify_momo_payment(self, transaction_id: str) -> Dict[str, Any]:
        """Verify MOMO payment status"""
        # Mock verification - replace with actual MOMO API call
        await asyncio.sleep(0.5)
        return {
            "transaction_id": transaction_id,
            "status": "verified",
            "gateway": "momo"
        }
    
    async def _verify_zalopay_payment(self, transaction_id: str) -> Dict[str, Any]:
        """Verify ZaloPay payment status"""
        # Mock verification - replace with actual ZaloPay API call
        await asyncio.sleep(0.5)
        return {
            "transaction_id": transaction_id,
            "status": "verified",
            "gateway": "zalopay"
        }

# Global payment service instance
payment_service = PaymentService()