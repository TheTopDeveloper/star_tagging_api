# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.model import KeywordExtractor, MetadataGenerator, ArticleGenerator, TextToSpeech
from app.utils import clean_text, filter_keywords, is_valid_input
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import os

app = FastAPI()

# Serve frontend
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("app/static/index.html")

# Load models once at startup
extractor = KeywordExtractor(use_cuda=False)  # set True for server deployment
metadata_generator = MetadataGenerator(use_cuda=False)  # set True for server deployment
article_generator = ArticleGenerator(use_cuda=False)  # set True for server deployment
tts_generator = TextToSpeech(use_cuda=False)  # set True for server deployment

class Article(BaseModel):
    content: str
    max_keywords: int = 15

class ArticleGenerationRequest(BaseModel):
    topic: str
    outline: Optional[List[str]] = None

class ArticleCurationRequest(BaseModel):
    content: str
    check_facts: bool = True
    check_journalistic_standards: bool = True
    check_style: bool = True

class TextToSpeechRequest(BaseModel):
    text: str
    voice_type: str = "professional"
    speech_rate: float = 1.0
    pitch: Optional[float] = None
    output_format: str = "wav"
    chunk_size: Optional[int] = 200

@app.post("/generate_keywords/")
def generate_keywords_endpoint(article: Article):
    cleaned = clean_text(article.content)

    if not is_valid_input(cleaned):
        raise HTTPException(status_code=400, detail="Article content is too short.")

    raw_keywords = extractor.extract_keywords(cleaned, article.max_keywords)
    keywords = filter_keywords(raw_keywords, article.max_keywords)

    return {"keywords": keywords}

@app.post("/generate_metadata/")
def generate_metadata_endpoint(article: Article):
    cleaned = clean_text(article.content)

    if not is_valid_input(cleaned):
        raise HTTPException(status_code=400, detail="Article content is too short.")

    metadata = metadata_generator.generate_metadata(cleaned)
    return metadata

@app.post("/generate_article/")
def generate_article_endpoint(request: ArticleGenerationRequest):
    """
    Generate an article based on a topic and optional outline.
    """
    try:
        result = article_generator.generate_article(request.topic, request.outline)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/curate_article/")
def curate_article_endpoint(request: ArticleCurationRequest):
    """
    Analyze an article and provide suggestions for improvement.
    """
    try:
        result = article_generator.curate_article(
            request.content,
            check_facts=request.check_facts,
            check_journalistic_standards=request.check_journalistic_standards,
            check_style=request.check_style
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-voices/")
async def get_available_voices():
    """
    Get list of available voices for text-to-speech.
    """
    try:
        return tts_generator.get_available_voices()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-speech/")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        print(f"Received TTS request for voice type: {request.voice_type}")
        print(f"Text length: {len(request.text)}")
        print(f"Output format: {request.output_format}")
        
        # Generate speech
        output_path = tts_generator.generate_speech(
            text=request.text,
            voice_type=request.voice_type,
            output_format=request.output_format
        )
        
        print(f"Speech generated successfully at: {output_path}")
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type=f"audio/{request.output_format}",
            filename=f"generated_speech.{request.output_format}"
        )
        
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in text-to-speech endpoint: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate speech: {str(e)}"
        )

