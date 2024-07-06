import os
from groq import Groq as groq

GROQ_API_KEY = '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'


class GroqSTT:
    def __init__(self, model_path="whisper-large-v3",
                 api_key_='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', proxy_url=None):
        self.client = groq(api_key=api_key_)
        self.model = model_path

    def transcribe_audio(self, audio_file, lang):
        with open(audio_file, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model=self.model,
                prompt="Specify context or spelling",
                response_format="json",
                language=lang,
                temperature=0.0
            )
        return transcription.text
