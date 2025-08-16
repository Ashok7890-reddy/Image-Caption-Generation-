"""
Translation utilities for multilingual caption generation.
"""

from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptionTranslator:
    """Translator for generating multilingual captions."""
    
    def __init__(self):
        """Initialize the translator."""
        # For demo purposes, we'll use a simple dictionary-based approach
        # In production, you would use Google Translate API or similar service
        
        # Common language codes
        self.supported_languages = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Chinese (Simplified)': 'zh-cn',
            'Arabic': 'ar',
            'Hindi': 'hi'
        }
    
    def translate_caption(
        self,
        caption: str,
        target_language: str,
        source_language: str = 'en'
    ) -> str:
        """
        Translate a caption to target language.
        
        Args:
            caption: Caption to translate
            target_language: Target language code
            source_language: Source language code (default: 'en')
            
        Returns:
            Translated caption (demo version returns original with language note)
        """
        try:
            if target_language == source_language:
                return caption
            
            # Demo implementation - in production, use actual translation service
            language_names = {v: k for k, v in self.supported_languages.items()}
            target_lang_name = language_names.get(target_language, target_language)
            
            return f"[{target_lang_name}] {caption}"
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return f"Translation failed: {caption}"
    
    def translate_multiple_captions(
        self,
        captions: List[str],
        target_language: str,
        source_language: str = 'en'
    ) -> List[str]:
        """Translate multiple captions."""
        translated = []
        for caption in captions:
            translated.append(
                self.translate_caption(caption, target_language, source_language)
            )
        return translated
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported language names and codes."""
        return self.supported_languages.copy()
    
    def detect_language(self, text: str) -> str:
        """Detect the language of given text."""
        # Demo implementation
        return 'en'  # Default to English
