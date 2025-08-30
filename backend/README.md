# Attendance System Backend

A FastAPI-based backend for the Face Recognition Attendance System.

## Features

- **Person Management**: Full CRUD operations for managing enrolled persons
- **Face Enrollment**: Process and store face encodings for recognition
- **Attendance Tracking**: Log entry/exit events with face recognition
- **Authentication**: JWT-based authentication system
- **Database**: SQLAlchemy ORM with SQLite database

## Tech Stack

- **Framework**: FastAPI
- **Database**: SQLite with SQLAlchemy ORM
- **Authentication**: JWT tokens with passlib
- **Face Processing**: OpenCV, face-recognition library
- **Image Processing**: Pillow, NumPy
- **Validation**: Pydantic models

## Installation

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Backend

1. Start the FastAPI server:
```bash
python main.py
```

2. Or use uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## API Endpoints

### Authentication
- `POST /api/auth/login` - Login and get access token
- `POST /api/auth/register` - Register a new person

### Persons Management
- `GET /api/persons` - Get all persons with filtering
- `GET /api/persons/{id}` - Get specific person with enrollments
- `POST /api/persons` - Create new person
- `PUT /api/persons/{id}` - Update person
- `DELETE /api/persons/{id}` - Delete person
- `PATCH /api/persons/{id}/toggle-active` - Toggle person active status
- `PATCH /api/persons/{id}/deactivate` - Deactivate person
- `PATCH /api/persons/{id}/activate` - Activate person

### Enrollment Management
- `GET /api/enrollment` - Get all enrollments
- `GET /api/enrollment/{id}` - Get specific enrollment
- `POST /api/enrollment` - Create new enrollment (with file upload)
- `PUT /api/enrollment/{id}` - Update enrollment
- `DELETE /api/enrollment/{id}` - Delete enrollment
- `PATCH /api/enrollment/{id}/toggle-active` - Toggle enrollment active status

### Health Check
- `GET /health` - Health check endpoint
- `GET /` - Root endpoint

## Database Schema

### Persons Table
- `id`: Primary key
- `first_name`: Person's first name
- `last_name`: Person's last name
- `personnel_code`: Unique personnel code
- `department`: Department name
- `position`: Job position
- `email`: Email address
- `phone`: Phone number
- `is_active`: Active status flag
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### Enrollments Table
- `id`: Primary key
- `person_id`: Foreign key to persons
- `face_encoding_path`: Path to stored face encoding
- `face_image_path`: Path to stored face image
- `confidence_score`: Face recognition confidence score
- `is_active`: Active status flag
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

### Attendance Logs Table
- `id`: Primary key
- `person_id`: Foreign key to persons
- `gate_name`: Gate name for entry/exit
- `direction`: Entry/exit direction
- `confidence_score`: Recognition confidence score
- `camera_name`: Camera name
- `status`: Authorization status
- `created_at`: Log timestamp

## Usage Examples

### Create a New Person
```bash
curl -X POST "http://localhost:8000/api/persons" \
     -H "Content-Type: application/json" \
     -d '{
       "first_name": "John",
       "last_name": "Doe",
       "personnel_code": "EMP001",
       "department": "IT",
       "position": "Developer",
       "email": "john.doe@company.com",
       "phone": "+1234567890"
     }'
```

### Enroll a Face
```bash
curl -X POST "http://localhost:8000/api/enrollment" \
     -F "person_id=1" \
     -F "file=@/path/to/face_image.jpg"
```

### Get All Persons
```bash
curl -X GET "http://localhost:8000/api/persons?is_active=true&department=IT"
```

## Face Processing

The system uses the `face-recognition` library for:
- Face detection in images
- Face encoding generation
- Face comparison and recognition
- Confidence score calculation

### Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## File Storage

- **Face Images**: Stored in `uploads/images/`
- **Face Encodings**: Stored in `uploads/encodings/`
- **Database**: SQLite database file `attendance.db`

## Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Input validation with Pydantic
- File type validation
- SQL injection protection with SQLAlchemy

## Configuration

Environment variables (see `.env` file):
- `DATABASE_URL`: Database connection string
- `SECRET_KEY`: JWT secret key
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time
- `UPLOAD_DIR`: Upload directory path
- `FACE_TOLERANCE`: Face recognition tolerance threshold

## Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Database Migrations
For production deployment, consider using Alembic for database migrations:
```bash
pip install alembic
alembic init alembic
# Configure alembic.ini
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Deployment

### Production Considerations
1. Use a production database (PostgreSQL/MySQL)
2. Set up proper environment variables
3. Use a WSGI server like Gunicorn
4. Set up reverse proxy (Nginx)
5. Configure SSL/TLS
6. Set up file backup for uploads
7. Monitor system resources

### Docker Support
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

1. **Face Recognition Not Working**
   - Ensure OpenCV and face-recognition are properly installed
   - Check image file formats and permissions
   - Verify face detection confidence thresholds

2. **Database Connection Issues**
   - Check DATABASE_URL configuration
   - Ensure database file permissions
   - Verify SQLAlchemy connection settings

3. **File Upload Problems**
   - Check upload directory permissions
   - Verify file size limits
   - Ensure proper file type validation

## License

This project is part of the Attendance System and is subject to the main project's license terms.