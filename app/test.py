import os
import torch
import tempfile
import numpy as np
import soundfile as sf
from typing import Dict, Optional
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset


class TextToSpeech:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the TextToSpeech class with a SpeechT5 model and processor.
        Optionally specify a device (e.g., 'cpu' or 'cuda').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "microsoft/speecht5_tts"
        self.processor = SpeechT5Processor.from_pretrained(self.model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name).to(self.device)

        # Load speaker embeddings dataset once
        print("Loading speaker embeddings...")
        self.speaker_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_id = 7306  # You can make this dynamic later

    def generate_speech(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        output_dir: str = "outputs",
        return_wav_path: bool = True
    ) -> Optional[str]:
        """
        Generate speech from input text and save it as a WAV file.

        Args:
            text (str): Input text.
            speaker_id (int, optional): Index for speaker embedding.
            output_dir (str): Directory to save the audio file.
            return_wav_path (bool): If True, return path to saved WAV file.

        Returns:
            str: Path to saved WAV file (if return_wav_path=True).
        """
        os.makedirs(output_dir, exist_ok=True)
        speaker_id = speaker_id or self.speaker_id

        # Clean and encode text
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        # Get speaker embedding
        speaker_embedding = torch.tensor(
            self.speaker_embeddings_dataset[speaker_id]["xvector"]
        ).unsqueeze(0).to(self.device)

        # Generate speech
        with torch.no_grad():
            speech_output = self.model.generate_speech(inputs["input_ids"], speaker_embedding)

        # Save to file
        wav_path = os.path.join(output_dir, "tts_output.wav")
        sf.write(wav_path, speech_output.cpu().numpy(), samplerate=16000)

        return wav_path if return_wav_path else None

import os
import torch
import tempfile
import numpy as np
import soundfile as sf
from typing import Dict, Optional
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset


class TextToSpeech:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the TextToSpeech class with a SpeechT5 model and processor.
        Optionally specify a device (e.g., 'cpu' or 'cuda').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "microsoft/speecht5_tts"
        self.processor = SpeechT5Processor.from_pretrained(self.model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name).to(self.device)

        # Load speaker embeddings dataset once
        print("Loading speaker embeddings...")
        self.speaker_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_id = 7306  # You can make this dynamic later

    def generate_speech(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        output_dir: str = "outputs",
        return_wav_path: bool = True
    ) -> Optional[str]:
        """
        Generate speech from input text and save it as a WAV file.

        Args:
            text (str): Input text.
            speaker_id (int, optional): Index for speaker embedding.
            output_dir (str): Directory to save the audio file.
            return_wav_path (bool): If True, return path to saved WAV file.

        Returns:
            str: Path to saved WAV file (if return_wav_path=True).
        """
        os.makedirs(output_dir, exist_ok=True)
        speaker_id = speaker_id or self.speaker_id

        # Clean and encode text
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        # Get speaker embedding
        speaker_embedding = torch.tensor(
            self.speaker_embeddings_dataset[speaker_id]["xvector"]
        ).unsqueeze(0).to(self.device)

        # Generate speech
        with torch.no_grad():
            speech_output = self.model.generate_speech(inputs["input_ids"], speaker_embedding)

        # Save to file
        wav_path = os.path.join(output_dir, "tts_output.wav")
        sf.write(wav_path, speech_output.cpu().numpy(), samplerate=16000)

        return wav_path if return_wav_path else None
        
if __name__ == "__main__":
    tts = TextToSpeech()
    wav_file = tts.generate_speech("Hello Joshua, welcome to your new text to speech engine!")
    print(f"Audio saved to: {wav_file}")
