#!/bin/bash

echo "Killing process on port 3000..."

# Find PID(s) using port 3000 and kill them
PID=$(lsof -t -i:3000)

if [ -n "$PID" ]; then
  kill -9 $PID
  echo "Port 3000 has been cleared."
else
  echo "No process found on port 3000."
fi
