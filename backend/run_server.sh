echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    echo "Please install Python and try again"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python command: $PYTHON_CMD"

# Check if uvicorn is installed
if ! $PYTHON_CMD -c "import uvicorn" &> /dev/null; then
    echo "Warning: uvicorn is not installed"
    echo "Installing uvicorn..."
    $PYTHON_CMD -m pip install uvicorn
fi

# Check if fastapi is installed
if ! $PYTHON_CMD -c "import fastapi" &> /dev/null; then
    echo "Warning: fastapi is not installed"
    echo "Installing required packages..."
    $PYTHON_CMD -m pip install fastapi uvicorn python-multipart
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API documentation will be available at http://localhost:8000/docs"
echo ""

# Change to the script's directory
cd "$(dirname "$0")"

# Start the FastAPI server
$PYTHON_CMD -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload