# API Documentation

## Overview
This API provides text-to-speech conversion and article curation services. It supports multiple voice types, output formats, and comprehensive article analysis features.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API does not require authentication.

## Endpoints

### 1. Text-to-Speech API

#### Endpoint
```
POST /text-to-speech/
```

#### Description
Converts text to speech with various voice options and output formats.

#### Request Body
```typescript
interface TextToSpeechRequest {
    text: string;           // Required: The text to convert to speech
    voice_type?: string;    // Optional: Type of voice to use
    output_format?: string; // Optional: Audio output format
}
```

#### Parameters
- `text` (required): The text to convert to speech
  - Type: string
  - Max length: 5000 characters
  - Example: "Hello, this is a test message."
- `voice_type` (optional): The type of voice to use
  - Type: string
  - Default: "professional"
  - Options:
    - "professional" - Clear, formal voice
    - "news_anchor" - Authoritative voice
    - "casual" - Friendly, conversational voice
    - "deep_voice" - Rich, deep voice
    - "young_voice" - Bright, energetic voice
    - "senior_voice" - Mature, experienced voice
    - "energetic" - Dynamic, enthusiastic voice
    - "calm" - Soothing, peaceful voice
- `output_format` (optional): The audio output format
  - Type: string
  - Default: "wav"
  - Options: "wav", "mp3", "ogg"

#### Response
```typescript
interface TextToSpeechResponse {
    audio_path: string;    // Path to the generated audio file
    duration: number;      // Duration of the audio in seconds
    format: string;        // Format of the generated audio
    voice_type: string;    // Voice type used for generation
}
```

#### NextJS Example
```typescript
// pages/api/text-to-speech.ts
import type { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const response = await fetch('http://localhost:8000/text-to-speech/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    res.status(500).json({ message: 'Error processing request' });
  }
}

// components/TextToSpeech.tsx
import { useState } from 'react';

export default function TextToSpeech() {
  const [text, setText] = useState('');
  const [voiceType, setVoiceType] = useState('professional');
  const [audioUrl, setAudioUrl] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const response = await fetch('/api/text-to-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          voice_type: voiceType,
          output_format: 'wav',
        }),
      });

      const data = await response.json();
      setAudioUrl(data.audio_path);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to convert to speech"
      />
      <select
        value={voiceType}
        onChange={(e) => setVoiceType(e.target.value)}
      >
        <option value="professional">Professional</option>
        <option value="news_anchor">News Anchor</option>
        <option value="casual">Casual</option>
        {/* Add other voice options */}
      </select>
      <button type="submit">Generate Speech</button>
      {audioUrl && (
        <audio controls src={audioUrl}>
          Your browser does not support the audio element.
        </audio>
      )}
    </form>
  );
}
```

### 2. Article Curation API

#### Endpoint
```
POST /curate_article/
```

#### Description
Analyzes and provides suggestions for improving article content.

#### Request Body
```typescript
interface ArticleCurationRequest {
    content: string;                    // Required: Article text to analyze
    check_facts?: boolean;             // Optional: Whether to perform fact-checking
    check_journalistic_standards?: boolean; // Optional: Check journalistic standards
    check_style?: boolean;             // Optional: Analyze writing style
}
```

#### Parameters
- `content` (required): The article text to analyze
  - Type: string
  - Max length: 10000 characters
  - Example: "Your article text here..."
- `check_facts` (optional): Whether to perform fact-checking
  - Type: boolean
  - Default: true
- `check_journalistic_standards` (optional): Whether to check for journalistic standards
  - Type: boolean
  - Default: true
- `check_style` (optional): Whether to analyze writing style
  - Type: boolean
  - Default: true

#### Response
```typescript
interface ArticleCurationResponse {
    analysis: {
        readability: {
            score: number;           // Readability score (0-100)
            level: string;           // Reading level
            suggestions: string[];    // Improvement suggestions
        };
        style: {
            formal: number;          // Formality score (0-1)
            academic: number;        // Academic style score (0-1)
            journalistic: number;    // Journalistic style score (0-1)
            conversational: number;  // Conversational style score (0-1)
        };
        bias_analysis: {
            emotional_bias: string[];    // Detected emotional biases
            political_bias: string[];    // Detected political biases
            gender_bias: string[];       // Detected gender biases
            overall_bias_score: number;  // Overall bias score (0-100)
        };
    };
    fact_check: {
        claims: Array<{
            text: string;           // Claim text
            confidence: number;     // Confidence score (0-1)
            status: string;         // Verification status
            suggested_sources: string[]; // Suggested sources
        }>;
        summary: {
            overall_reliability: string;  // Overall reliability rating
            key_findings: string[];       // Key findings
            recommendations: string[];    // Improvement recommendations
        };
    };
    suggestions: string[];  // General improvement suggestions
}
```

#### NextJS Example
```typescript
// pages/api/curate-article.ts
import type { NextApiRequest, NextApiResponse } from 'next';

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const response = await fetch('http://localhost:8000/curate_article/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (error) {
    res.status(500).json({ message: 'Error processing request' });
  }
}

// components/ArticleCuration.tsx
import { useState } from 'react';

export default function ArticleCuration() {
  const [content, setContent] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('/api/curate-article', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          check_facts: true,
          check_journalistic_standards: true,
          check_style: true,
        }),
      });

      const data = await response.json();
      setAnalysis(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder="Enter your article text"
          rows={10}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Article'}
        </button>
      </form>

      {analysis && (
        <div className="analysis-results">
          <h3>Analysis Results</h3>
          <div className="readability">
            <h4>Readability Score: {analysis.analysis.readability.score}</h4>
            <p>Level: {analysis.analysis.readability.level}</p>
            <ul>
              {analysis.analysis.readability.suggestions.map((suggestion, index) => (
                <li key={index}>{suggestion}</li>
              ))}
            </ul>
          </div>

          <div className="style-analysis">
            <h4>Style Analysis</h4>
            <ul>
              <li>Formal: {analysis.analysis.style.formal}</li>
              <li>Academic: {analysis.analysis.style.academic}</li>
              <li>Journalistic: {analysis.analysis.style.journalistic}</li>
              <li>Conversational: {analysis.analysis.style.conversational}</li>
            </ul>
          </div>

          <div className="suggestions">
            <h4>Suggestions</h4>
            <ul>
              {analysis.suggestions.map((suggestion, index) => (
                <li key={index}>{suggestion}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
```

## Error Handling

### Error Response Format
```typescript
interface ErrorResponse {
    detail: string;        // Error message description
    error_type: string;    // Error category
    status_code: number;   // HTTP status code
}
```

### Common Error Codes
- `400 Bad Request`: Invalid input parameters
- `422 Unprocessable Entity`: Invalid request format
- `500 Internal Server Error`: Server-side processing error

## Rate Limiting
The API implements rate limiting to ensure fair usage:
- 100 requests per hour for text-to-speech
- 200 requests per hour for article curation

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625097600
```

## Best Practices
1. Always check the response status code
2. Handle rate limiting by implementing exponential backoff
3. Cache responses when appropriate
4. Use appropriate content types in requests
5. Implement error handling for all API calls
6. Use TypeScript interfaces for type safety
7. Implement loading states for better UX
8. Add error boundaries in React components
9. Use proper form validation
10. Implement proper error messages for users 