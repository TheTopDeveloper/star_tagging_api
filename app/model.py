# app/model.py
# app/model.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification, AutoProcessor, AutoModel,SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from typing import List, Optional, Dict
import torch
import spacy
from app.utils import clean_text
import concurrent.futures
import os
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import tempfile
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_device(use_cuda: bool) -> torch.device:
    """
    Set up the device for model operations with proper error handling.
    Automatically detects and verifies GPU availability.
    """
    try:
        if use_cuda and torch.cuda.is_available():
            try:
                # Test GPU with a small tensor operation
                x = torch.rand(5, 3).cuda()
                y = torch.rand(5, 3).cuda()
                z = x + y
                del x, y, z
                torch.cuda.empty_cache()
                
                # Get GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
                logger.info(f"GPU detected: {gpu_name} with {gpu_memory:.1f}GB memory")
                
                # Set device
                device = torch.device("cuda")
                # Enable cuDNN benchmarking for better performance
                torch.backends.cudnn.benchmark = True
                return device
            except Exception as e:
                logger.error(f"GPU test failed: {str(e)}")
                logger.warning("Falling back to CPU")
                return torch.device("cpu")
        else:
            logger.info("Using CPU for model operations")
            return torch.device("cpu")
    except Exception as e:
        logger.error(f"Error setting up device: {str(e)}")
        return torch.device("cpu")

class KeywordExtractor:
    def __init__(self, model_name: str = "facebook/bart-base", use_cuda: bool = False):
        self.model_name = model_name
        self.device = setup_device(use_cuda)
        
        try:
            # Enable model optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Initialize model with appropriate device settings
            if self.device.type == "cuda":
                # Use device_map for GPU
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map={"": self.device}
                )
                self.model.eval()
                torch.cuda.empty_cache()
            else:
                # Direct device placement for CPU
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
            
            # Load spaCy model for noun extraction
            self.nlp = spacy.load("en_core_web_sm")
            logger.info(f"KeywordExtractor initialized successfully on {self.device.type}")
            
        except Exception as e:
            logger.error(f"Error initializing KeywordExtractor: {str(e)}")
            raise

    def extract_keywords(
        self, 
        text: str, 
        max_keywords: int = 15,  # Increased default to 15
        max_length: int = 64, 
        num_beams: int = 4
    ) -> List[str]:
        cleaned = clean_text(text)
        
        # First, extract nouns using spaCy
        doc = self.nlp(cleaned)
        noun_keywords = []
        
        # Extract proper nouns (names, places) and regular nouns
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'GPE', 'LOC', 'ORG']:  # Person, Location, Organization
                noun_keywords.append(ent.text.lower())
        
        # Add other nouns if we haven't reached max_keywords
        for token in doc:
            if token.pos_ == "NOUN" and len(token.text) > 3:
                if token.text.lower() not in noun_keywords:
                    noun_keywords.append(token.text.lower())
        
        # If we don't have enough nouns, use BART to generate additional keywords
        if len(noun_keywords) < max_keywords:
            with torch.no_grad():
                inputs = self.tokenizer.encode(
                    cleaned, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                summary_ids = self.model.generate(
            inputs, 
            max_length=max_length,
            num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False,
                    use_cache=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Extract additional keywords from summary
            for word in summary.replace('.', ' ').replace(',', ' ').split():
                word = word.strip().lower()
                if len(word) > 3 and word not in noun_keywords:
                    noun_keywords.append(word)
        
        # Return up to max_keywords, prioritizing the nouns we found
        return noun_keywords[:max_keywords]

class MetadataGenerator:
    def __init__(self, use_cuda: bool = False):
        self.device = setup_device(use_cuda)
        
        try:
            # Enable model optimizations
            torch.backends.cudnn.benchmark = True if self.device.type == "cuda" else False
            
            # Initialize summarizer with a public model
            summarizer_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/bart-base",
                cache_dir="./model_cache"
            )
            summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/bart-base",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                cache_dir="./model_cache"
            ).to(self.device)
            
            # Enable model optimizations
            if self.device.type == "cuda":
                summarizer_model.eval()
                torch.cuda.empty_cache()
            
            self.summarizer = pipeline(
                "summarization",
                model=summarizer_model,
                tokenizer=summarizer_tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                max_length=100,
                min_length=20,
                do_sample=False,
                num_beams=2,
                early_stopping=True,
                batch_size=4
            )
            
            # Initialize classifier with optimizations
            classifier_tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english",
                cache_dir="./model_cache"
            )
            classifier_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                cache_dir="./model_cache"
            ).to(self.device)
            
            if self.device.type == "cuda":
                classifier_model.eval()
                torch.cuda.empty_cache()
            
            self.classifier = pipeline(
                "text-classification",
                model=classifier_model,
                tokenizer=classifier_tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                batch_size=8
            )
            
            logger.info(f"MetadataGenerator initialized successfully on {self.device.type}")
            
        except Exception as e:
            logger.error(f"Error initializing MetadataGenerator: {str(e)}")
            raise
        
        # Load spaCy with optimizations
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'textcat'])
        self.nlp.add_pipe('sentencizer')  # Add sentencizer component
        
        # Define possible categories
        self.categories = [
            "Technology", "Business", "Politics", "Science", "Health",
            "Entertainment", "Sports", "Education", "Environment", "World"
        ]
        
        # Initialize cache for summaries
        self.summary_cache = {}
        self.title_cache = {}

    def _extract_entities(self, text: str) -> Dict:
        """Extract entities from text using spaCy"""
        doc = self.nlp(text)
        entities = {
            'people': set(),  # Changed to sets for uniqueness
            'organizations': set(),
            'locations': set(),
            'dates': set()
        }
        
        for ent in doc.ents:
            # Clean and normalize the entity text
            cleaned_text = ent.text.strip()
            if not cleaned_text:  # Skip empty strings
                continue
                
            if ent.label_ == 'PERSON':
                entities['people'].add(cleaned_text)
            elif ent.label_ in ['ORG', 'GPE']:
                # Normalize organization names
                org_name = cleaned_text
                # Remove common suffixes
                org_name = org_name.replace(' Inc.', '').replace(' Inc', '')
                org_name = org_name.replace(' Ltd.', '').replace(' Ltd', '')
                org_name = org_name.replace(' LLC', '').replace(' L.L.C.', '')
                org_name = org_name.replace(' Corp.', '').replace(' Corp', '')
                # Remove extra whitespace
                org_name = ' '.join(org_name.split())
                entities['organizations'].add(org_name)
            elif ent.label_ == 'LOC':
                entities['locations'].add(cleaned_text)
            elif ent.label_ == 'DATE':
                entities['dates'].add(cleaned_text)
        
        # Convert sets back to sorted lists
        return {
            'people': sorted(list(entities['people'])),
            'organizations': sorted(list(entities['organizations'])),
            'locations': sorted(list(entities['locations'])),
            'dates': sorted(list(entities['dates']))
        }

    def _chunk_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split text into chunks of maximum length"""
        # Use spaCy for better sentence splitting
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text.split())
            
            if current_length + sent_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sent_text]
                current_length = sent_length
            else:
                current_chunk.append(sent_text)
                current_length += sent_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _generate_summary(self, text: str, max_length: int = 100, min_length: int = 20) -> str:
        """Generate a summary with specific length constraints"""
        # Check cache first
        cache_key = f"{text[:100]}_{max_length}_{min_length}"
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]
        
        try:
            # If text is too short, return it as is
            if len(text.split()) <= min_length:
                return text

            # Generate summary with adjusted parameters
            with torch.no_grad():  # Disable gradient calculation
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=2,  # Reduced for speed
                    early_stopping=True
                )[0]['summary_text']

            # Cache the result
            self.summary_cache[cache_key] = summary
            return summary
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return text[:max_length] + "..."

    def _generate_title(self, text: str) -> str:
        """Generate a concise title"""
        # Check cache first
        cache_key = text[:100]
        if cache_key in self.title_cache:
            return self.title_cache[cache_key]
        
        try:
            # Use shorter max_length for title
            with torch.no_grad():  # Disable gradient calculation
                title = self.summarizer(
                    text[:512],
                    max_length=30,
                    min_length=5,
                    do_sample=False,
                    num_beams=2,
                    early_stopping=True
                )[0]['summary_text']

            # Clean up the title
            title = title.strip()
            if title.endswith('.'):
                title = title[:-1]
            
            # Cache the result
            self.title_cache[cache_key] = title
            return title
        except Exception as e:
            print(f"Error generating title: {str(e)}")
            return text.split('.')[0][:50]

    def _predict_categories(self, text: str) -> List[str]:
        """Predict categories for the text using a simpler approach"""
        # Use only first 512 characters for faster processing
        text = text[:512]
        
        # Process categories in parallel with batching
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for category in self.categories:
                prompt = f"Text: {text}\nCategory: {category}"
                future = executor.submit(
                    self.classifier,
                    prompt,
                    truncation=True,
                    max_length=512
                )
                futures.append((category, future))
            
            # Collect results
            category_scores = []
            for category, future in futures:
                try:
                    result = future.result()[0]
                    category_scores.append((category, result['score']))
                except Exception as e:
                    print(f"Error processing category {category}: {str(e)}")
                    continue
        
        # Get top 2 categories
        sorted_categories = sorted(category_scores, key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in sorted_categories[:min(2, len(sorted_categories))]]

    def generate_metadata(self, text: str) -> Dict:
        """
        Generate comprehensive metadata for the article.
        """
        try:
            cleaned_text = clean_text(text)
            
            # Split text into chunks if it's too long
            chunks = self._chunk_text(cleaned_text)
            
            # Process summary and entities in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Start summary generation for each chunk
                summary_futures = []
                for chunk in chunks:
                    future = executor.submit(
                        self._generate_summary,
                        chunk,
                        max_length=100,
                        min_length=20
                    )
                    summary_futures.append(future)
                
                # Start entity extraction and category prediction in parallel
                entities_future = executor.submit(self._extract_entities, cleaned_text)
                categories_future = executor.submit(self._predict_categories, cleaned_text)
                
                # Get results
                summaries = []
                for future in summary_futures:
                    try:
                        summary = future.result()
                        if summary:
                            summaries.append(summary)
                    except Exception as e:
                        print(f"Error generating summary: {str(e)}")
                        continue
                
                # Combine summaries and generate final summary
                combined_summary = ' '.join(summaries)
                final_summary = self._generate_summary(
                    combined_summary,
                    max_length=150,
                    min_length=30
                )
                
                # Generate title from the first chunk
                title = self._generate_title(chunks[0])
                
                entities = entities_future.result()
                categories = categories_future.result()
            
            return {
                "title": title,
                "summary": final_summary,
                "categories": categories,
                "entities": entities,
                "word_count": len(cleaned_text.split()),
                "estimated_read_time": max(1, len(cleaned_text.split()) // 200)
            }
        except Exception as e:
            print(f"Error in generate_metadata: {str(e)}")
            raise

class ArticleGenerator:
    def __init__(self, use_cuda: bool = False):
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        
        try:
            # Initialize article generation model
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/bart-large-cnn",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            
            # Initialize content quality classifier
            quality_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            quality_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            
            self.quality_classifier = pipeline(
                "text-classification",
                model=quality_model,
                tokenizer=quality_tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
            # Initialize sentiment analyzer
            sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                tokenizer=sentiment_tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
            # Load spaCy for content analysis
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize fact-checking model
            self.fact_checker = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device.type == "cuda" else -1
            )
            
            # Define style metrics
            self.style_metrics = {
                "formal": {
                    "indicators": ["therefore", "furthermore", "moreover", "consequently", "thus", "hence", "accordingly"],
                    "threshold": 0.3
                },
                "academic": {
                    "indicators": ["research", "study", "analysis", "evidence", "theory", "methodology", "findings"],
                    "threshold": 0.2
                },
                "journalistic": {
                    "indicators": ["reported", "according", "sources", "announced", "revealed", "stated", "confirmed"],
                    "threshold": 0.25
                },
                "conversational": {
                    "indicators": ["you", "we", "let's", "think", "feel", "believe", "consider"],
                    "threshold": 0.3
                }
            }
            
            # Define tone indicators
            self.tone_indicators = {
                "objective": ["fact", "data", "evidence", "research", "study", "analysis"],
                "subjective": ["think", "feel", "believe", "opinion", "view", "perspective"],
                "critical": ["however", "but", "although", "despite", "nevertheless", "yet"],
                "supportive": ["indeed", "certainly", "clearly", "obviously", "undoubtedly", "absolutely"],
                "neutral": ["according", "reported", "stated", "mentioned", "noted", "observed"]
            }
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def generate_article(self, topic: str, outline: Optional[List[str]] = None) -> Dict:
        """
        Generate an article based on a topic and optional outline.
        """
        # Create a prompt from the topic and outline
        prompt = f"Write an article about {topic}."
        if outline:
            prompt += " Follow this outline:\n" + "\n".join(f"- {point}" for point in outline)
        
        # Generate the article
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=1000,
            min_length=200,
            num_beams=4,
            early_stopping=True
        )
        
        article = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Analyze the generated content
        analysis = self._analyze_content(article)
        
        return {
            "article": article,
            "analysis": analysis
        }

    def curate_article(self, article: str, check_facts: bool = True, 
                      check_journalistic_standards: bool = True, 
                      check_style: bool = True) -> Dict:
        """
        Analyze and provide suggestions for improving an article.
        """
        # Clean the article text
        cleaned_text = clean_text(article)
        
        # Initialize result dictionary
        result = {
            "analysis": {},
            "suggestions": [],
            "fact_check": {"claims": []},
            "journalistic_analysis": {"bias_analysis": {}},
            "style_analysis": {}
        }
        
        # Analyze the content if style checking is enabled
        if check_style:
            result["style_analysis"] = self._analyze_style(cleaned_text)
            result["suggestions"].extend(result["style_analysis"].get("recommendations", []))
        
        # Perform fact checking if enabled
        if check_facts:
            result["fact_check"] = self._check_facts(cleaned_text)
        
        # Check for bias if enabled
        if check_journalistic_standards:
            doc = self.nlp(cleaned_text)
            bias_analysis = self._check_bias_indicators(doc)
            result["journalistic_analysis"]["bias_analysis"] = bias_analysis
            result["suggestions"].extend(self._generate_bias_suggestions(bias_analysis))
        
        return result

    def _analyze_style(self, text: str) -> Dict:
        """
        Analyze the writing style of the text.
        """
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            words = [token.text.lower() for token in doc if not token.is_punct]
            
            # Calculate basic metrics
            total_words = len(words)
            total_sentences = len(sentences)
            avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
            
            # Analyze sentence complexity
            complex_sentences = sum(1 for sent in sentences if len(sent) > 20)
            sentence_complexity = complex_sentences / total_sentences if total_sentences > 0 else 0
            
            # Calculate style scores
            style_scores = {}
            for style, config in self.style_metrics.items():
                indicator_count = sum(1 for word in words if word in config["indicators"])
                style_scores[style] = indicator_count / total_words if total_words > 0 else 0
            
            # Analyze tone
            tone_scores = {}
            for tone, indicators in self.tone_indicators.items():
                indicator_count = sum(1 for word in words if word in indicators)
                tone_scores[tone] = indicator_count / total_words if total_words > 0 else 0
            
            # Calculate readability scores
            readability_scores = self._calculate_readability_metrics(text)
            
            # Analyze vocabulary diversity
            unique_words = len(set(words))
            vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
            
            # Analyze sentence structure
            sentence_structure = {
                "simple": 0,
                "compound": 0,
                "complex": 0
            }
            
            for sent in sentences:
                if len(sent) < 10:
                    sentence_structure["simple"] += 1
                elif any(token.dep_ == "cc" for token in sent):
                    sentence_structure["compound"] += 1
                else:
                    sentence_structure["complex"] += 1
            
            # Normalize sentence structure counts
            total = sum(sentence_structure.values())
            if total > 0:
                for key in sentence_structure:
                    sentence_structure[key] = sentence_structure[key] / total
            
            # Determine dominant style and tone
            dominant_style = max(style_scores.items(), key=lambda x: x[1])[0]
            dominant_tone = max(tone_scores.items(), key=lambda x: x[1])[0]
            
            # Generate style recommendations
            recommendations = self._generate_style_recommendations(
                style_scores,
                tone_scores,
                readability_scores,
                sentence_structure,
                vocabulary_diversity
            )
            
            return {
                "metrics": {
                    "avg_sentence_length": round(avg_sentence_length, 2),
                    "sentence_complexity": round(sentence_complexity, 2),
                    "vocabulary_diversity": round(vocabulary_diversity, 2)
                },
                "style_scores": {k: round(v, 2) for k, v in style_scores.items()},
                "tone_scores": {k: round(v, 2) for k, v in tone_scores.items()},
                "readability_scores": readability_scores,
                "sentence_structure": {k: round(v, 2) for k, v in sentence_structure.items()},
                "dominant_style": dominant_style,
                "dominant_tone": dominant_tone,
                "recommendations": recommendations
            }
            
        except Exception as e:
            print(f"Error in style analysis: {str(e)}")
            return {
                "error": str(e),
                "metrics": {},
                "style_scores": {},
                "tone_scores": {},
                "readability_scores": {},
                "sentence_structure": {},
                "recommendations": []
            }

    def _calculate_readability_metrics(self, text: str) -> Dict:
        """
        Calculate various readability metrics.
        """
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            words = [token.text for token in doc if not token.is_punct]
            
            # Calculate basic metrics
            total_words = len(words)
            total_sentences = len(sentences)
            total_syllables = sum(self._count_syllables(word) for word in words)
            
            # Calculate average metrics
            avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
            avg_syllables_per_word = total_syllables / total_words if total_words > 0 else 0
            
            # Calculate Flesch Reading Ease
            flesch_score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            flesch_score = max(0, min(100, flesch_score))
            
            # Calculate Flesch-Kincaid Grade Level
            fk_grade = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
            fk_grade = max(0, min(12, fk_grade))
            
            # Determine readability level
            if flesch_score >= 90:
                readability_level = "Very Easy"
            elif flesch_score >= 80:
                readability_level = "Easy"
            elif flesch_score >= 70:
                readability_level = "Fairly Easy"
            elif flesch_score >= 60:
                readability_level = "Standard"
            elif flesch_score >= 50:
                readability_level = "Fairly Difficult"
            elif flesch_score >= 30:
                readability_level = "Difficult"
            else:
                readability_level = "Very Difficult"
            
            return {
                "flesch_reading_ease": round(flesch_score, 2),
                "flesch_kincaid_grade": round(fk_grade, 2),
                "readability_level": readability_level,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "avg_syllables_per_word": round(avg_syllables_per_word, 2)
            }
            
        except Exception as e:
            print(f"Error calculating readability metrics: {str(e)}")
            return {
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "readability_level": "Error",
                "avg_words_per_sentence": 0,
                "avg_syllables_per_word": 0
            }

    def _count_syllables(self, word: str) -> int:
        """
        Count the number of syllables in a word.
        """
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        previous_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_is_vowel:
                count += 1
            previous_is_vowel = is_vowel
        
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
            
        return count

    def _generate_style_recommendations(self, style_scores: Dict, tone_scores: Dict,
                                      readability_scores: Dict, sentence_structure: Dict,
                                      vocabulary_diversity: float) -> List[str]:
        """
        Generate recommendations based on style analysis.
        """
        recommendations = []
        
        # Check readability
        if readability_scores["flesch_reading_ease"] < 60:
            recommendations.append("Consider simplifying language to improve readability")
        elif readability_scores["flesch_reading_ease"] > 90:
            recommendations.append("The text might be too simple for the target audience")
        
        # Check sentence structure
        if sentence_structure["complex"] > 0.5:
            recommendations.append("Consider breaking down complex sentences for better clarity")
        elif sentence_structure["simple"] > 0.7:
            recommendations.append("Add more variety to sentence structure")
        
        # Check vocabulary diversity
        if vocabulary_diversity < 0.4:
            recommendations.append("Consider using more diverse vocabulary")
        
        # Check style balance
        max_style_score = max(style_scores.values())
        if max_style_score > 0.4:
            dominant_style = max(style_scores.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Consider balancing the {dominant_style} style with other styles")
        
        # Check tone balance
        max_tone_score = max(tone_scores.values())
        if max_tone_score > 0.4:
            dominant_tone = max(tone_scores.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Consider balancing the {dominant_tone} tone with other tones")
        
        return recommendations

    def _analyze_content(self, text: str) -> Dict:
        """
        Analyze the content quality and structure.
        """
        # Process with spaCy
        doc = self.nlp(text)
        
        # Basic statistics
        word_count = len(doc)
        sentence_count = len(list(doc.sents))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Analyze readability
        readability_score = self._calculate_readability(text)
        
        # Analyze content quality
        quality_score = self._assess_quality(text)
        
        # Extract key entities
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': []
        }
        
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                entities['people'].append(ent.text)
            elif ent.label_ in ['ORG', 'GPE']:
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'LOC':
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": round(avg_sentence_length, 2),
            "readability_score": readability_score,
            "quality_score": quality_score,
            "entities": entities
        }

    def _calculate_readability(self, text: str) -> float:
        """
        Calculate a simple readability score.
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        words = [token.text for token in doc if not token.is_punct]
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability formula (lower is more readable)
        readability = (avg_sentence_length * 0.39) + (avg_word_length * 11.8) - 15.59
        return max(0, min(100, 100 - readability))

    def _assess_quality(self, text: str) -> float:
        """
        Assess the quality of the content using the classifier.
        """
        try:
            result = self.quality_classifier(text[:512])[0]
            return result['score'] * 100
        except Exception as e:
            print(f"Error assessing quality: {str(e)}")
            return 0.0

    def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """
        Generate suggestions for improving the article based on analysis.
        """
        suggestions = []
        
        # Check sentence length
        if analysis['avg_sentence_length'] > 20:
            suggestions.append("Consider breaking down long sentences for better readability.")
        elif analysis['avg_sentence_length'] < 10:
            suggestions.append("Some sentences might be too short. Consider combining related ideas.")
        
        # Check readability
        if analysis['readability_score'] < 60:
            suggestions.append("The text might be too complex. Consider simplifying language and structure.")
        
        # Check quality score
        if analysis['quality_score'] < 70:
            suggestions.append("Consider adding more specific details and examples to improve content quality.")
        
        # Check entity coverage
        if not any(analysis['entities'].values()):
            suggestions.append("Consider adding more specific names, organizations, and locations to make the content more concrete.")
        
        return suggestions

    def _check_facts(self, text: str) -> Dict:
        """
        Enhanced fact checking with detailed analysis, source suggestions, and claim categorization.
        """
        try:
            # Split text into sentences
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            # Initialize results
            claims_to_verify = []
            claim_categories = {
                "statistical": 0,
                "historical": 0,
                "scientific": 0,
                "geographical": 0,
                "temporal": 0
            }
            confidence_distribution = {"high": 0, "medium": 0, "low": 0}
            
            # Define claim categories and their indicators
            category_indicators = {
                "statistical": ["percent", "percentage", "rate", "average", "mean", "median", "statistic", "survey", "poll", "study", "data", "figure", "number", "count"],
                "historical": ["in", "during", "century", "decade", "year", "period", "era", "ancient", "medieval", "modern", "history", "past", "ago", "since"],
                "scientific": ["research", "study", "experiment", "scientists", "discovered", "found", "proved", "evidence", "data", "analysis", "results", "conclusion", "theory"],
                "geographical": ["located", "country", "city", "region", "area", "distance", "population", "climate", "geography", "place", "location", "territory"],
                "temporal": ["recently", "currently", "now", "today", "yesterday", "tomorrow", "future", "past", "present", "time", "date", "when"]
            }
            
            # Define factual claim indicators
            factual_indicators = [
                "is", "are", "was", "were", "has", "have", "had",
                "will", "would", "can", "could", "should", "must",
                "always", "never", "every", "all", "none", "no",
                "according to", "reported", "stated", "claimed",
                "announced", "revealed", "confirmed", "found",
                "discovered", "showed", "demonstrated", "proved",
                "indicates", "suggests", "shows", "reveals",
                "confirms", "demonstrates", "proves", "establishes",
                "determines", "concludes", "finds", "identifies"
            ]
            
            for sentence in sentences:
                # Skip very short sentences or questions
                if len(sentence.split()) < 5 or sentence.endswith('?'):
                    continue
                
                # Determine claim category
                category = None
                max_matches = 0
                for cat, keywords in category_indicators.items():
                    matches = sum(1 for keyword in keywords if keyword in sentence.lower())
                    if matches > max_matches:
                        max_matches = matches
                        category = cat
                
                if category:
                    claim_categories[category] += 1
                
                # Check if sentence contains potential factual claims
                if any(indicator in sentence.lower() for indicator in factual_indicators):
                    try:
                        # Use the fact-checking model to assess the claim
                        result = self.fact_checker(
                            sentence,
                            candidate_labels=["factual", "unverified", "misleading"],
                            hypothesis_template="This statement is {}."
                        )
                        
                        confidence = result["scores"][0]
                        status = result["labels"][0]
                        
                        # Update confidence distribution
                        if confidence > 0.8:
                            confidence_distribution["high"] += 1
                        elif confidence > 0.6:
                            confidence_distribution["medium"] += 1
                        else:
                            confidence_distribution["low"] += 1
                        
                        # If the claim needs verification
                        if status in ["unverified", "misleading"] or confidence < 0.6:
                            # Get suggested sources based on claim category
                            suggested_sources = []
                            if category == "statistical":
                                suggested_sources = ["academic journals", "government databases", "research institutions", "statistical agencies"]
                            elif category == "historical":
                                suggested_sources = ["historical archives", "academic historians", "primary sources", "historical databases"]
                            elif category == "scientific":
                                suggested_sources = ["scientific journals", "research papers", "academic databases", "peer-reviewed studies"]
                            elif category == "geographical":
                                suggested_sources = ["geographical databases", "census data", "government records", "geographical surveys"]
                            elif category == "temporal":
                                suggested_sources = ["news archives", "current events databases", "timeline records", "recent reports"]
                            
                            # Analyze claim structure
                            claim_doc = self.nlp(sentence)
                            entities = {
                                "people": [],
                                "organizations": [],
                                "locations": [],
                                "dates": []
                            }
                            
                            for ent in claim_doc.ents:
                                if ent.label_ == 'PERSON':
                                    entities["people"].append(ent.text)
                                elif ent.label_ in ['ORG', 'GPE']:
                                    entities["organizations"].append(ent.text)
                                elif ent.label_ == 'LOC':
                                    entities["locations"].append(ent.text)
                                elif ent.label_ == 'DATE':
                                    entities["dates"].append(ent.text)
                            
                            # Identify claim type
                            claim_type = "statement"
                            if any(word in sentence.lower() for word in ["if", "when", "unless"]):
                                claim_type = "conditional"
                            elif any(word in sentence.lower() for word in ["should", "must", "ought"]):
                                claim_type = "prescriptive"
                            elif any(word in sentence.lower() for word in ["will", "going to", "plan to"]):
                                claim_type = "predictive"
                            
                            claims_to_verify.append({
                                "text": sentence,
                                "confidence": float(confidence),
                                "status": status,
                                "category": category,
                                "suggested_sources": suggested_sources,
                                "analysis": {
                                    "entities": entities,
                                    "claim_type": claim_type,
                                    "word_count": len(claim_doc),
                                    "has_subject": any(token.dep_ == "nsubj" for token in claim_doc),
                                    "has_verb": any(token.pos_ == "VERB" for token in claim_doc),
                                    "has_object": any(token.dep_ == "dobj" for token in claim_doc)
                                }
                            })
                    except Exception as e:
                        print(f"Error processing claim: {str(e)}")
                        continue
            
            # Generate summary
            total_claims = len(claims_to_verify)
            if total_claims > 0:
                high_confidence_ratio = confidence_distribution["high"] / total_claims
                if high_confidence_ratio > 0.7:
                    reliability = "high"
                elif high_confidence_ratio > 0.4:
                    reliability = "moderate"
                else:
                    reliability = "low"
                
                key_findings = []
                if claim_categories:
                    most_common_category = max(claim_categories.items(), key=lambda x: x[1])
                    key_findings.append(f"Most common claim type: {most_common_category[0]} ({most_common_category[1]} claims)")
                
                if confidence_distribution["low"] > 0:
                    key_findings.append(f"{confidence_distribution['low']} claims need verification")
                
                recommendations = []
                if reliability == "low":
                    recommendations.append("Consider adding more citations and sources")
                    recommendations.append("Review claims with low confidence scores")
                
                if claim_categories["statistical"] > 0:
                    recommendations.append("Verify statistical claims with official sources")
                
                if claim_categories["scientific"] > 0:
                    recommendations.append("Ensure scientific claims are backed by peer-reviewed research")
                
                summary = {
                    "overall_reliability": reliability,
                    "key_findings": key_findings,
                    "recommendations": recommendations
                }
            else:
                summary = {
                    "overall_reliability": "No claims to verify",
                    "key_findings": [],
                    "recommendations": []
                }
            
            return {
                "claims": claims_to_verify,
                "total_claims_checked": len(sentences),
                "claims_needing_verification": len(claims_to_verify),
                "claim_categories": claim_categories,
                "confidence_distribution": confidence_distribution,
                "summary": summary
            }
            
        except Exception as e:
            print(f"Error in fact checking: {str(e)}")
            return {
                "claims": [],
                "total_claims_checked": 0,
                "claims_needing_verification": 0,
                "error": str(e)
            }

    def _check_bias_indicators(self, doc) -> Dict:
        """
        Check for various types of bias in the text.
        """
        # Initialize bias categories
        bias_analysis = {
            "emotional_bias": [],
            "political_bias": [],
            "gender_bias": [],
            "racial_bias": [],
            "tribal_bias": [],
            "overall_bias_score": 0
        }
        
        # Define bias indicators
        bias_indicators = {
            "emotional_bias": [
                "amazing", "incredible", "terrible", "horrible", "wonderful",
                "fantastic", "awful", "beautiful", "ugly", "perfect"
            ],
            "political_bias": [
                "liberal", "conservative", "left-wing", "right-wing",
                "democrat", "republican", "progressive", "traditional"
            ],
            "gender_bias": [
                "he", "she", "his", "her", "him", "man", "woman",
                "male", "female", "gentleman", "lady"
            ],
            "racial_bias": [
                "race", "ethnic", "nationality", "culture", "background",
                "origin", "heritage", "ancestry"
            ],
            "tribal_bias": [
                "us", "them", "our", "their", "we", "they",
                "group", "team", "community", "society"
            ]
        }
        
        # Check each sentence for bias indicators
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Check each bias category
            for bias_type, indicators in bias_indicators.items():
                for indicator in indicators:
                    if indicator in sent_text:
                        # Get the context (few words before and after the indicator)
                        words = sent_text.split()
                        try:
                            idx = words.index(indicator)
                            start = max(0, idx - 3)
                            end = min(len(words), idx + 4)
                            context = " ".join(words[start:end])
                            
                            if context not in bias_analysis[bias_type]:
                                bias_analysis[bias_type].append(context)
                        except ValueError:
                            continue
        
        # Calculate overall bias score
        total_biases = sum(len(biases) for biases in bias_analysis.values() if isinstance(biases, list))
        max_possible_biases = 20  # Arbitrary maximum for normalization
        bias_analysis["overall_bias_score"] = min(100, (total_biases / max_possible_biases) * 100)
        
        return bias_analysis

    def _generate_bias_suggestions(self, bias_analysis: Dict) -> List[str]:
        """
        Generate suggestions for reducing bias in the text.
        """
        suggestions = []
        
        # Check emotional bias
        if len(bias_analysis["emotional_bias"]) > 0:
            suggestions.append("Consider using more neutral language instead of emotionally charged words.")
        
        # Check political bias
        if len(bias_analysis["political_bias"]) > 0:
            suggestions.append("Try to present political views in a more balanced way.")
        
        # Check gender bias
        if len(bias_analysis["gender_bias"]) > 0:
            suggestions.append("Consider using gender-neutral language where possible.")
        
        # Check racial bias
        if len(bias_analysis["racial_bias"]) > 0:
            suggestions.append("Ensure racial and ethnic references are relevant and necessary.")
        
        # Check tribal bias
        if len(bias_analysis["tribal_bias"]) > 0:
            suggestions.append("Be mindful of us-vs-them language that might create unnecessary divisions.")
        
        # Overall bias score suggestions
        if bias_analysis["overall_bias_score"] > 70:
            suggestions.append("The text shows significant bias. Consider a more balanced approach.")
        elif bias_analysis["overall_bias_score"] > 40:
            suggestions.append("Some bias detected. Review the text for potential improvements.")
        
        return suggestions

class TextToSpeech:
    def __init__(self, use_cuda: bool = False):
        self.device = setup_device(use_cuda)
        
        try:
            logger.info("Initializing TextToSpeech model...")
            # Load processor, model, vocoder
            self.processor = SpeechT5Processor.from_pretrained(
                "microsoft/speecht5_tts",
                cache_dir="./model_cache"
            )
            logger.info("Loaded processor successfully")
            
            self.model = SpeechT5ForTextToSpeech.from_pretrained(
                "microsoft/speecht5_tts",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                cache_dir="./model_cache"
            ).to(self.device)
            logger.info("Loaded model successfully")
            
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan",
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                cache_dir="./model_cache"
            ).to(self.device)
            logger.info("Loaded vocoder successfully")
            
            # Load speaker embeddings dataset
            logger.info("Loading speaker embeddings...")
            self.speaker_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            logger.info("Loaded speaker embeddings successfully")
            
            if self.device.type == "cuda":
                self.model.eval()
                self.vocoder.eval()
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
            
            logger.info(f"TextToSpeech initialization completed successfully on {self.device.type}")
            
        except Exception as e:
            logger.error(f"Error initializing TTS models: {str(e)}")
            raise RuntimeError(f"Failed to initialize TextToSpeech: {str(e)}")

    def _preprocess_text_for_pronunciation(self, text: str) -> str:
        """
        Preprocess text to improve pronunciation of names and places using our comprehensive dictionary.
        """
        try:
            # Load spaCy model if not already loaded
            if not hasattr(self, 'nlp'):
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            
            # Import our pronunciation dictionary
            from app.utils.pronunciation_dict import PRONUNCIATION_DICT
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Process each sentence
            processed_sentences = []
            for sent in doc.sents:
                processed_words = []
                for token in sent:
                    # Check if the token is a proper noun or part of a named entity
                    if token.ent_type_ in ['PERSON', 'GPE', 'LOC', 'ORG']:
                        # Get the full entity span
                        entity = token.ent_
                        # Look up pronunciation in dictionary
                        if entity in PRONUNCIATION_DICT:
                            processed_words.append(PRONUNCIATION_DICT[entity])
                        else:
                            # For unknown entities, add spaces between letters to help pronunciation
                            processed_words.append(" ".join(entity))
                    else:
                        processed_words.append(token.text)
                
                processed_sentences.append(" ".join(processed_words))
            
            return ". ".join(processed_sentences)
            
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return text

    def generate_speech(self, text: str, voice_type: str = "professional", 
                       output_format: str = "wav", output_path: str = None) -> str:
        """
        Generate speech from text with specified voice.
        """
        try:
            logger.info(f"Generating speech for voice type: {voice_type}")
            
            # Validate inputs
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
                
            if voice_type not in self.voices:
                raise ValueError(f"Invalid voice type. Choose from: {list(self.voices.keys())}")
                
            if output_format not in ["wav", "mp3", "ogg"]:
                raise ValueError("Output format must be wav, mp3, or ogg")
            
            # Get voice configuration
            voice_config = self.voices[voice_type]
            logger.info(f"Using speaker ID: {voice_config['speaker_id']}")
            
            # Clean and prepare text
            cleaned_text = clean_text(text)
            logger.info(f"Cleaned text length: {len(cleaned_text)}")
            
            # Preprocess text for better pronunciation
            processed_text = self._preprocess_text_for_pronunciation(cleaned_text)
            logger.info("Text preprocessed for pronunciation")
            
            # Split text into sentences
            sentences = [s.strip() for s in processed_text.split('.') if s.strip()]
            logger.info(f"Split into {len(sentences)} sentences")
            
            # Process text in chunks
            all_speech = []
            chunk_size = 3  # Reduced chunk size for better stability
            max_length = 600  # Maximum input length for the model
            
            for i in range(0, len(sentences), chunk_size):
                chunk = '. '.join(sentences[i:i + chunk_size]) + '.'
                logger.info(f"Processing chunk {i//chunk_size + 1} of {(len(sentences) + chunk_size - 1)//chunk_size}")
                
                try:
                    # Process text with truncation
                    inputs = self.processor(
                        text=chunk,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding="max_length"  # Ensure consistent length
                    ).to(self.device)
                    
                    # Get speaker embedding
                    speaker_embedding = torch.tensor(
                        self.speaker_embeddings_dataset[voice_config["speaker_id"]]["xvector"]
                    ).unsqueeze(0).to(self.device)
                    
                    # Generate speech for chunk
                    with torch.no_grad():
                        speech = self.model.generate_speech(
                            inputs["input_ids"],
                            speaker_embeddings=speaker_embedding,
                            vocoder=self.vocoder
                        )
                    
                    # Convert to numpy and ensure consistent shape
                    speech_np = speech.cpu().numpy()
                    if len(speech_np.shape) > 1:
                        speech_np = speech_np.squeeze()
                    
                    all_speech.append(speech_np)
                    
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk: {str(chunk_error)}")
                    continue
            
            if not all_speech:
                raise RuntimeError("No speech chunks were successfully generated")
            
            # Ensure all chunks have the same length before concatenating
            min_length = min(chunk.shape[0] for chunk in all_speech)
            all_speech = [chunk[:min_length] for chunk in all_speech]
            
            # Concatenate all speech chunks
            final_speech = np.concatenate(all_speech)
            
            # Create temporary file if no output path provided
            if output_path is None:
                temp_dir = tempfile.gettempdir()
                output_path = os.path.join(temp_dir, f"generated_speech_{voice_type}.{output_format}")
            
            # Get sampling rate from model config (SpeechT5 uses 16000 Hz)
            sampling_rate = 16000
            
            # Save audio file in specified format
            logger.info(f"Saving audio in {output_format} format...")
            if output_format == "wav":
                sf.write(output_path, final_speech, sampling_rate)
            elif output_format in ["mp3", "ogg"]:
                # Convert to MP3/OGG using pydub
                from pydub import AudioSegment
                audio = AudioSegment(
                    final_speech.tobytes(),
                    frame_rate=sampling_rate,
                    sample_width=2,
                    channels=1
                )
                audio.export(output_path, format=output_format)
            logger.info(f"Audio saved successfully to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to generate speech: {str(e)}")

    def get_available_voices(self) -> List[Dict]:
        """Get list of available voices with descriptions."""
        try:
            return [
                {
                    "id": voice_id,
                    "name": voice_id.replace("_", " ").title(),
                    "description": config["description"]
                }
                for voice_id, config in self.voices.items()
            ]
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}")
            raise RuntimeError(f"Failed to get available voices: {str(e)}")

