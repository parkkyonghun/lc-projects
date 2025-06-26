import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class NotificationService:
    """Service for handling email and SMS notifications"""
    
    def __init__(self):
        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password
    
    async def send_loan_approval_notification(
        self, 
        user_email: str, 
        user_name: str, 
        loan_amount: float, 
        loan_id: str
    ) -> bool:
        """Send loan approval notification"""
        subject = "ការអនុម័តប្រាក់កម្ចី - Loan Approval Notification"
        
        # Khmer and English content
        khmer_content = f"""
        សូមស្វាគមន៍ {user_name},
        
        យើងសូមជូនដំណឹងដ៏រីករាយថា ការស្នើសុំប្រាក់កម្ចីរបស់អ្នកត្រូវបានអនុម័ត។
        
        ព័ត៌មានលម្អិត:
        - លេខកម្ចី: {loan_id}
        - ចំនួនទឹកប្រាក់: {loan_amount:,.0f} រៀល
        - កាលបរិច្ឆេទអនុម័ត: {datetime.now().strftime('%d/%m/%Y')}
        
        សូមទាក់ទងមកយើងខ្ញុំសម្រាប់ព័ត៌មានបន្ថែម។
        
        សូមអរគុណ!
        """
        
        english_content = f"""
        Dear {user_name},
        
        We are pleased to inform you that your loan application has been approved.
        
        Loan Details:
        - Loan ID: {loan_id}
        - Amount: {loan_amount:,.0f} KHR
        - Approval Date: {datetime.now().strftime('%d/%m/%Y')}
        
        Please contact us for further information.
        
        Thank you!
        """
        
        content = khmer_content + "\n\n" + english_content
        
        return await self._send_email(user_email, subject, content)
    
    async def send_payment_reminder(
        self, 
        user_email: str, 
        user_name: str, 
        loan_id: str, 
        amount_due: float, 
        due_date: str
    ) -> bool:
        """Send payment reminder notification"""
        subject = "ការរំលឹកការបង់ប្រាក់ - Payment Reminder"
        
        khmer_content = f"""
        សូមស្វាគមន៍ {user_name},
        
        នេះជាការរំលឹកអំពីការបង់ប្រាក់កម្ចីដែលនឹងដល់កំណត់។
        
        ព័ត៌មានការបង់ប្រាក់:
        - លេខកម្ចី: {loan_id}
        - ចំនួនត្រូវបង់: {amount_due:,.0f} រៀល
        - កាលបរិច្ឆេទកំណត់: {due_date}
        
        សូមបង់ប្រាក់ឱ្យបានទាន់ពេលវេលា។
        
        សូមអរគុណ!
        """
        
        english_content = f"""
        Dear {user_name},
        
        This is a reminder about your upcoming loan payment.
        
        Payment Details:
        - Loan ID: {loan_id}
        - Amount Due: {amount_due:,.0f} KHR
        - Due Date: {due_date}
        
        Please make your payment on time.
        
        Thank you!
        """
        
        content = khmer_content + "\n\n" + english_content
        
        return await self._send_email(user_email, subject, content)
    
    async def send_payment_confirmation(
        self, 
        user_email: str, 
        user_name: str, 
        loan_id: str, 
        payment_amount: float, 
        transaction_id: str
    ) -> bool:
        """Send payment confirmation notification"""
        subject = "ការបញ្ជាក់ការបង់ប្រាក់ - Payment Confirmation"
        
        khmer_content = f"""
        សូមស្វាគមន៍ {user_name},
        
        យើងបានទទួលការបង់ប្រាក់របស់អ្នកដោយជោគជ័យ។
        
        ព័ត៌មានការបង់ប្រាក់:
        - លេខកម្ចី: {loan_id}
        - ចំនួនបានបង់: {payment_amount:,.0f} រៀល
        - លេខប្រតិបត្តិការ: {transaction_id}
        - កាលបរិច្ឆេទ: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        
        សូមអរគុណចំពោះការបង់ប្រាក់របស់អ្នក!
        """
        
        english_content = f"""
        Dear {user_name},
        
        We have successfully received your payment.
        
        Payment Details:
        - Loan ID: {loan_id}
        - Amount Paid: {payment_amount:,.0f} KHR
        - Transaction ID: {transaction_id}
        - Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        
        Thank you for your payment!
        """
        
        content = khmer_content + "\n\n" + english_content
        
        return await self._send_email(user_email, subject, content)
    
    async def _send_email(self, to_email: str, subject: str, content: str) -> bool:
        """Send email using SMTP"""
        if not all([self.smtp_server, self.smtp_username, self.smtp_password]):
            logger.warning("SMTP configuration incomplete, email not sent")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_username
            msg['To'] = to_email
            
            # Add content
            text_part = MIMEText(content, 'plain', 'utf-8')
            msg.attach(text_part)
            
            # Send email
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_smtp_email, msg, to_email
            )
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False
    
    def _send_smtp_email(self, msg: MIMEMultipart, to_email: str):
        """Send email via SMTP (synchronous)"""
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg, to_addresses=[to_email])
    
    async def send_bulk_notifications(
        self, 
        notifications: List[Dict[str, str]], 
        notification_type: str = "reminder"
    ) -> Dict[str, int]:
        """Send bulk notifications"""
        results = {"sent": 0, "failed": 0}
        
        for notification in notifications:
            try:
                if notification_type == "reminder":
                    success = await self.send_payment_reminder(
                        notification["email"],
                        notification["name"],
                        notification["loan_id"],
                        float(notification["amount"]),
                        notification["due_date"]
                    )
                elif notification_type == "approval":
                    success = await self.send_loan_approval_notification(
                        notification["email"],
                        notification["name"],
                        float(notification["amount"]),
                        notification["loan_id"]
                    )
                else:
                    success = False
                
                if success:
                    results["sent"] += 1
                else:
                    results["failed"] += 1
                    
                # Add delay between emails to avoid spam detection
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to send notification: {str(e)}")
                results["failed"] += 1
        
        return results

# Global notification service instance
notification_service = NotificationService()