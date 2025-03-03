#!/bin/bash

# Execute the first Python script
echo "Indexing docs..."
python3 chatbot/doc/index.py || {
    echo "Error executing chatbot/doc/index.py"
    exit 1
}

# Check if Streamlit is installed
if ! command -v streamlit >/dev/null 2>&1; then
    echo "Streamlit not found. Please install Streamlit before running the script."
    exit 1
fi

echo "Start chatbot UI..."
streamlit run chatbot/ui/chatwindow.py || {
    echo "Error executing chatbot/ui/chatwindows.py"
    exit 1
}