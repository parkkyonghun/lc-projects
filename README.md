# Khmer Loan Management System

A comprehensive loan management application designed specifically for Cambodian financial institutions, featuring Khmer language support, ID card OCR scanning, and modern web technologies.

## 🌟 Features

### Core Functionality
- **Multi-language Support**: Full Khmer and English language support
- **ID Card OCR**: Automatic extraction of information from Cambodian ID cards
- **Loan Management**: Complete loan lifecycle from application to repayment
- **Payment Processing**: Integration with local payment gateways (Momo, ZaloPay)
- **User Authentication**: Secure JWT-based authentication with role-based access
- **Dashboard Analytics**: Comprehensive reporting and analytics
- **Document Management**: Secure file upload and management

### User Roles
- **Customer**: Apply for loans, make payments, view loan status
- **Staff**: Process applications, manage customer accounts
- **Manager**: Approve loans, view branch analytics
- **Admin**: System administration, user management

### Technical Features
- **FastAPI Backend**: High-performance async API
- **PostgreSQL Database**: Robust data storage with async support
- **OCR Processing**: Tesseract-based text extraction for Khmer text
- **Security**: Password hashing, JWT tokens, input validation
- **Email Notifications**: Automated email notifications for loan events
- **File Upload**: Secure document and image upload handling

## 🏗️ Architecture

```
lc-projects/
├── api/                    # API route definitions
├── controllers/            # Business logic controllers
│   ├── dashboard_controller.py
│   ├── loan_controller.py
│   ├── payment_controller.py
│   └── user_controller.py
├── core/                   # Core configuration and utilities
│   ├── config.py          # Application configuration
│   ├── database.py        # Database setup and connection
│   └── security.py        # Security utilities
├── models/                 # SQLAlchemy database models
│   ├── branch.py
│   ├── loan.py
│   ├── payment.py
│   └── user.py
├── schemas/                # Pydantic schemas for API
│   ├── dashboard.py
│   ├── loan.py
│   ├── payment.py
│   └── user.py
├── services/               # Business logic services
│   ├── auth_service.py     # Authentication and user management
│   ├── dashboard_service.py # Analytics and reporting
│   ├── file_service.py     # File upload and management
│   ├── loan_service.py     # Loan processing and management
│   ├── notification_service.py # Email notifications
│   ├── ocr_service.py      # OCR processing
│   └── payment_service.py  # Payment processing
├── views/                  # API route handlers
│   ├── dashboard_view.py
│   ├── loan_view.py
│   ├── ocr_view.py
│   ├── payment_view.py
│   └── user_view.py
├── khmer_resources/        # Khmer language resources
├── training_data/          # OCR training data
├── main.py                 # FastAPI application entry point
├── init_db.py             # Database initialization script
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Tesseract OCR with Khmer language support
- Redis (for caching and background tasks)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lc-projects
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-khm
   
   # macOS
   brew install tesseract tesseract-lang
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

5. **Setup PostgreSQL**
   ```bash
   # Create database
   createdb loan_management
   
   # Create user (optional)
   psql -c "CREATE USER loan_user WITH PASSWORD 'your_password';"
   psql -c "GRANT ALL PRIVILEGES ON DATABASE loan_management TO loan_user;"
   ```

6. **Configure environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your configuration
   DATABASE_URL=postgresql+asyncpg://loan_user:your_password@localhost/loan_management
   SECRET_KEY=your-secret-key-here
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   ```

7. **Initialize database**
   ```bash
   python init_db.py
   ```

8. **Start the application**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

9. **Access the application**
   - API Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## 🔐 Default Login Credentials

After running the database initialization script, you can use these credentials:

| Role | Email | Password |
|------|-------|---------|
| Admin | admin@loanapp.kh | admin123 |
| Manager | manager@loanapp.kh | manager123 |
| Staff | staff@loanapp.kh | staff123 |
| Customer | customer1@example.com | customer123 |
| Customer | customer2@example.com | customer123 |

## 📚 API Documentation

### Authentication

**Login**
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "admin@loanapp.kh",
  "password": "admin123"
}
```

**Register**
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "full_name": "John Doe",
  "phone": "012-345-678"
}
```

### Loan Management

**Create Loan Application**
```http
POST /api/loans/
Authorization: Bearer <token>
Content-Type: application/json

{
  "amount": 5000000,
  "term_months": 24,
  "loan_type": "PERSONAL",
  "purpose": "Business expansion",
  "monthly_income": 800000
}
```

**Get User Loans**
```http
GET /api/loans/my-loans
Authorization: Bearer <token>
```

### OCR Processing

**Process ID Card**
```http
POST /api/ocr/process-id-card
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <id-card-image>
```

### Payment Processing

**Make Payment**
```http
POST /api/payments/
Authorization: Bearer <token>
Content-Type: application/json

{
  "loan_id": "loan-uuid",
  "amount": 254000,
  "payment_method": "MOMO"
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `SECRET_KEY` | JWT secret key | Required |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration time | 30 |
| `SMTP_SERVER` | Email server | localhost |
| `SMTP_PORT` | Email server port | 587 |
| `UPLOAD_DIRECTORY` | File upload directory | ./uploads |
| `MAX_FILE_SIZE` | Maximum file size (bytes) | 10485760 |
| `TESSERACT_CMD` | Tesseract executable path | tesseract |
| `REDIS_URL` | Redis connection string | redis://localhost:6379 |

### Payment Gateway Configuration

```python
# Momo Configuration
MOMO_API_URL = "https://test-payment.momo.vn/v2/gateway/api/create"
MOMO_PARTNER_CODE = "your-partner-code"
MOMO_ACCESS_KEY = "your-access-key"
MOMO_SECRET_KEY = "your-secret-key"

# ZaloPay Configuration
ZALOPAY_APP_ID = "your-app-id"
ZALOPAY_KEY1 = "your-key1"
ZALOPAY_KEY2 = "your-key2"
ZALOPAY_ENDPOINT = "https://sb-openapi.zalopay.vn/v2/create"
```

## 🧪 Testing

**Run tests**
```bash
pytest
```

**Run with coverage**
```bash
pytest --cov=. --cov-report=html
```

**Test specific module**
```bash
pytest tests/test_loan_service.py
```

## 📊 Database Schema

### Key Tables

- **users**: User accounts and authentication
- **branches**: Bank branch information
- **loans**: Loan applications and details
- **payments**: Payment records and transactions
- **files**: Uploaded document metadata

### Relationships

- Users belong to branches
- Loans are associated with users and branches
- Payments are linked to loans and users
- Files can be attached to loans or users

## 🔒 Security Features

- **Password Hashing**: Bcrypt with configurable rounds
- **JWT Authentication**: Secure token-based authentication
- **Input Validation**: Pydantic schema validation
- **SQL Injection Protection**: SQLAlchemy ORM
- **File Upload Security**: Type and size validation
- **Rate Limiting**: Configurable request limits
- **CORS Protection**: Configurable CORS policies

## 🌐 Khmer Language Support

- **Unicode Support**: Full Khmer Unicode character support
- **OCR Recognition**: Tesseract trained for Khmer text
- **Font Resources**: Khmer fonts included
- **Validation**: Khmer text validation utilities
- **Localization**: Khmer translations for UI elements

## 📱 Mobile Considerations

- **Responsive Design**: Mobile-first approach
- **Touch Optimization**: Touch-friendly interfaces
- **Offline Support**: Basic offline functionality
- **Camera Integration**: Direct camera access for ID scanning

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-khm \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

- Use environment-specific configuration
- Set up proper logging
- Configure reverse proxy (Nginx)
- Set up SSL certificates
- Configure database backups
- Set up monitoring and alerting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Email: support@loanapp.kh
- Documentation: [Project Wiki](wiki-url)

## 🗺️ Roadmap

### Phase 1 (Current)
- ✅ Core loan management
- ✅ User authentication
- ✅ OCR processing
- ✅ Basic payment processing

### Phase 2 (Planned)
- 📱 Mobile application
- 🔔 Push notifications
- 📊 Advanced analytics
- 🤖 AI-powered risk assessment

### Phase 3 (Future)
- 🌐 Multi-tenant support
- 📈 Credit scoring integration
- 🔗 Third-party integrations
- 📱 Mobile SDK

---

**Built with ❤️ for the Cambodian financial sector**

## Setup Instructions

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd lc-projects
   ```

2. **(Recommended) Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

The main entry point is `main.py`. To start the FastAPI server, run:

```bash
uvicorn main:app --reload
```

- The API will be available at: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs

## Project Structure

- `main.py` - Application entry point
- `controllers/` - Route controllers
- `models/` - Database models
- `schemas/` - Pydantic schemas
- `views/` - (If used) View logic
- `tests/` - Test cases

## Environment Variables
If your project uses environment variables, create a `.env` file in the root directory and add your variables there.

---
If you have any questions or need additional setup, let me know!
