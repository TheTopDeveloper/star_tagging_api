# Keyword Extraction API

A FastAPI-based service that extracts keywords from text using a combination of spaCy for noun extraction and BART for text summarization. The service prioritizes proper nouns (names, places, organizations) while maintaining good coverage of other important terms.

## Features

- Extracts up to 15 keywords from input text
- Prioritizes proper nouns (names, places, organizations)
- GPU acceleration support for faster processing
- Clean and modern web interface
- RESTful API endpoint
- Comprehensive metadata generation

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- virtualenv (recommended)
- CUDA-capable GPU (optional, for faster processing)
- Hugging Face account and API token

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd tagging_api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

5. Set up your Hugging Face token:
   - Create a Hugging Face account at https://huggingface.co/
   - Get your API token from https://huggingface.co/settings/tokens
   - Set the token as an environment variable:
     ```bash
     # Linux/Mac
     export HF_TOKEN=your_token_here
     
     # Windows (PowerShell)
     $env:HF_TOKEN="your_token_here"
     
     # Windows (Command Prompt)
     set HF_TOKEN=your_token_here
     ```

## Usage

1. Start the server:
```bash
uvicorn app.main:app --reload
```

2. Access the web interface:
- Open your browser and navigate to `https://apiai.radioafrica.digital`
- Paste your text in the text area
- Click "Generate Keywords" or "Generate Metadata"

3. Using the API directly:
```bash
# Generate keywords
curl -X POST "https://apiai.radioafrica.digital/generate_keywords/" \
     -H "Content-Type: application/json" \
     -d '{"content": "Your text here", "max_keywords": 15}'

# Generate metadata
curl -X POST "https://apiai.radioafrica.digital/generate_metadata/" \
     -H "Content-Type: application/json" \
     -d '{"content": "Your text here"}'
```

## GPU Acceleration

To enable GPU acceleration:

1. Ensure you have CUDA installed on your system:
```bash
nvidia-smi  # Check if CUDA is available
```

2. Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

3. Modify the `use_cuda` parameter in `app/main.py`:
```python
extractor = KeywordExtractor(use_cuda=True)
metadata_generator = MetadataGenerator(use_cuda=True)
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `https://apiai.radioafrica.digital/docs`
- ReDoc: `https://apiai.radioafrica.digital/redoc`

## Project Structure

```
tagging_api/
├── app/
│   ├── main.py          # FastAPI application and routes
│   ├── model.py         # Keyword extraction and metadata models
│   ├── utils.py         # Utility functions
│   └── static/
│       └── index.html   # Web interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Troubleshooting

1. If you encounter CUDA-related errors:
   - Ensure your GPU drivers are up to date
   - Verify CUDA installation with `nvidia-smi`
   - Try running without GPU by setting `use_cuda=False`

2. If spaCy model fails to load:
   - Reinstall the model: `python -m spacy download en_core_web_sm`
   - Check your Python version compatibility

3. If Hugging Face models fail to load:
   - Verify your HF_TOKEN is set correctly
   - Check your internet connection
   - Ensure you have access to the required models

4. If the server fails to start:
   - Check if port 8000 is available
   - Ensure all dependencies are installed
   - Check the logs for specific error messages

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license] 