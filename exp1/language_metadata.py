"""
Language Metadata
Provides language-specific information including Unicode ranges,
answer prompts, and token classification with proper number/punctuation handling
"""

from typing import Dict, List


class LanguageMetadata:
    """Static class for language metadata"""
    
    # Extended Unicode ranges for script detection
    UNICODE_RANGES = {
        # Latin script - will need vocabulary-based disambiguation
        'latin': [(0x0041, 0x007A), (0x00C0, 0x00FF)],  # Basic + Extended Latin
        
        # Chinese
        'cjk': [(0x4E00, 0x9FFF)],  # CJK Unified Ideographs
        
        # Japanese
        'hiragana': [(0x3040, 0x309F)],  # Hiragana
        'katakana': [(0x30A0, 0x30FF)],  # Katakana
        
        # Korean
        'hangul': [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],  # Hangul Syllables + Jamo
        
        # Hindi
        'devanagari': [(0x0900, 0x097F)],  # Devanagari (Hindi)
        
        # Bengali
        'bengali': [(0x0980, 0x09FF)],  # Bengali
        
        # Telugu
        'telugu': [(0x0C00, 0x0C7F)],  # Telugu
        
        # Thai
        'thai': [(0x0E00, 0x0E7F)],  # Thai
        
        # Cyrillic (Russian)
        'cyrillic': [(0x0400, 0x04FF)],  # Cyrillic
    }
    
    # Language-specific numbers
    LANGUAGE_NUMBERS = {
        'zh': {
            # Chinese numbers (traditional and simplified)
            'digits': ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九',
                      '十', '百', '千', '万', '亿', '兆',
                      # Full-width digits
                      '０', '１', '２', '３', '４', '５', '６', '７', '８', '９'],
        },
        'ja': {
            # Japanese numbers (kanji and full-width)
            'digits': ['〇', '一', '二', '三', '四', '五', '六', '七', '八', '九',
                      '十', '百', '千', '万', '億', '兆',
                      # Full-width digits
                      '０', '１', '２', '３', '４', '５', '６', '７', '８', '９'],
        },
        'hi': {
            # Hindi/Devanagari numbers
            'digits': ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९'],
        },
        'bn': {
            # Bengali numbers
            'digits': ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯'],
        },
        'te': {
            # Telugu numbers
            'digits': ['౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯'],
        },
        'th': {
            # Thai numbers
            'digits': ['๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙'],
        },
        'ko': {
            # Korean numbers (Korean words for numbers, plus they use Arabic)
            'digits': ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십', '백', '천', '만'],
        },
        # Latin-script languages (en, es, fr, de, ru) use Arabic numerals 0-9
        'en': {'digits': []},  # Uses ASCII 0-9
        'es': {'digits': []},  # Uses ASCII 0-9
        'fr': {'digits': []},  # Uses ASCII 0-9
        'de': {'digits': []},  # Uses ASCII 0-9
        'ru': {'digits': []},  # Uses ASCII 0-9
    }
    
    # Language-specific punctuation
    LANGUAGE_PUNCTUATION = {
        'zh': {
            # Chinese punctuation
            'marks': ['。', '，', '、', '；', '：', '？', '！', '…', '—', '·',
                     '「', '」', '『', '』', '〈', '〉', '《', '》', '【', '】',
                     '（', '）', '"', '"', ''', '''],
        },
        'ja': {
            # Japanese punctuation (similar to Chinese but with some differences)
            'marks': ['。', '、', '！', '？', '：', '；', '～', '…', '—',
                     '「', '」', '『', '』', '〈', '〉', '《', '》', '【', '】',
                     '（', '）', '〔', '〕'],
        },
        'ko': {
            # Korean punctuation
            'marks': ['。', '，', '、', '！', '？', '：', '；', '…',
                     '（', '）', '【', '】', '『', '』', '「', '」'],
        },
        'hi': {
            # Hindi punctuation
            'marks': ['।', '॥', '॰'],  # Devanagari danda
        },
        'bn': {
            # Bengali punctuation (uses some Devanagari punctuation)
            'marks': ['।', '॥'],
        },
        'te': {
            # Telugu punctuation
            'marks': ['।', '॥'],
        },
        'th': {
            # Thai punctuation
            'marks': ['ๆ', 'ฯ', '๚', '๛'],
        },
        'es': {
            # Spanish-specific punctuation
            'marks': ['¿', '¡'],  # Inverted question/exclamation marks
        },
        'fr': {
            # French-specific punctuation
            'marks': ['«', '»', '‹', '›'],  # French quotation marks
        },
        'de': {
            # German-specific punctuation
            'marks': ['„', '"'],  # German quotation marks
        },
        'ru': {
            # Russian-specific punctuation
            'marks': ['«', '»'],  # Russian quotation marks
        },
        'en': {
            # English punctuation (standard ASCII)
            'marks': [],  # Uses standard ASCII punctuation
        },
    }
    
    # Language-specific common tokens/patterns for Latin-script disambiguation
    LANGUAGE_SPECIFIC_TOKENS = {
        'es': {
            # Spanish-specific words and patterns
            'words': ['de', 'la', 'el', 'que', 'en', 'los', 'las', 'del', 'por', 'para', 
                     'con', 'una', 'su', 'al', 'más', 'como', 'pero', 'sus', 'le',
                     'es', 'son', 'está', 'están', 'ser', 'tiene', 'tienen'],
            # Spanish diacritics
            'chars': ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡'],
            # Common suffixes
            'suffixes': ['ción', 'dad', 'mente', 'ando', 'iendo'],
        },
        'fr': {
            # French-specific words and patterns
            'words': ['le', 'la', 'les', 'de', 'un', 'une', 'des', 'et', 'est', 'que',
                     'dans', 'pour', 'par', 'sur', 'avec', 'ce', 'cette', 'du', 'au',
                     'sont', 'être', 'ont', 'était', 'été'],
            # French diacritics
            'chars': ['à', 'â', 'é', 'è', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ç', 'œ', '«', '»'],
            # Common patterns
            'patterns': ['ent', 'tion', 'ment', 'eur'],
        },
        'de': {
            # German-specific words (expanded)
            'words': ['der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'und',
                    'in', 'von', 'zu', 'mit', 'auf', 'für', 'ist', 'sind', 'war',
                    'werden', 'wurde', 'nicht', 'auch', 'sich', 'aber',
                    'ich', 'sie', 'er', 'wir', 'es', 'oder', 'als', 'bei',
                    'nach', 'aus', 'an', 'kann', 'durch', 'hat', 'wenn'],
            # German umlauts (higher weight)
            'chars': ['ä', 'ö', 'ü', 'ß'],
            # Common patterns
            'patterns': ['ung', 'heit', 'keit', 'schaft', 'chen', 'lein', 'lich', 'bar'],
        },
        'en': {
            # English-specific words
            'words': ['the', 'of', 'and', 'to', 'in', 'is', 'it', 'that', 'for', 'as',
                     'with', 'was', 'on', 'are', 'be', 'at', 'by', 'this', 'have',
                     'from', 'or', 'had', 'but', 'what', 'which', 'their'],
            'chars': [],  # No unique diacritics
            'patterns': ['ing', 'tion', 'ness', 'ment'],
        },
    }
    
    # Script to language mapping (for non-Latin scripts)
    SCRIPT_TO_LANGUAGE = {
        'cjk': 'zh',
        'hiragana': 'ja',
        'katakana': 'ja',
        'hangul': 'ko',
        'devanagari': 'hi',
        'bengali': 'bn',
        'telugu': 'te',
        'thai': 'th',
        'cyrillic': 'ru',
    }
    
    # Answer prompts for each language
    ANSWER_PROMPTS = {
        'en': ' Answer:',
        'zh': ' 答案:',
        'bn': ' উত্তর:',
        'te': ' సమాధానం:',
        'th': ' คำตอบ:',
        'es': ' Respuesta:',
        'fr': ' Réponse:',
        'de': ' Antwort:',
        'ja': ' 答え:',
        'ru': ' Ответ:',
        'hi': ' उत्तर:',
        'ko': ' 답변:',
    }
    
    @classmethod
    def detect_script_from_char(cls, char: str) -> str:
        """
        Detect script from a single character
        
        Args:
            char: Single character
            
        Returns:
            Script name or 'unknown'
        """
        if not char:
            return 'unknown'
        
        code_point = ord(char)
        
        # ASCII whitespace → latin
        if char in ' \t\n\r':
            return 'latin'
        
        # ASCII letters, digits, basic punctuation → latin
        if ((ord('a') <= code_point <= ord('z')) or
            (ord('A') <= code_point <= ord('Z')) or
            (ord('0') <= code_point <= ord('9')) or
            char in '.,!?;:\'"()[]{}+-=*/<>@#$%^&_`~|\\'):
            return 'latin'
        
        # Check language-specific numbers first (higher priority)
        for lang, num_data in cls.LANGUAGE_NUMBERS.items():
            if char in num_data.get('digits', []):
                # Map to script
                if lang == 'zh':
                    return 'cjk'
                elif lang == 'ja':
                    # Could be hiragana/katakana/cjk, but we'll use cjk for numbers
                    return 'cjk'
                elif lang == 'hi':
                    return 'devanagari'
                elif lang == 'bn':
                    return 'bengali'
                elif lang == 'te':
                    return 'telugu'
                elif lang == 'th':
                    return 'thai'
                elif lang == 'ko':
                    return 'hangul'
        
        # Check language-specific punctuation
        for lang, punct_data in cls.LANGUAGE_PUNCTUATION.items():
            if char in punct_data.get('marks', []):
                # Map to script
                if lang == 'zh':
                    return 'cjk'
                elif lang == 'ja':
                    return 'hiragana'  # Japanese punctuation marker
                elif lang == 'hi':
                    return 'devanagari'
                elif lang == 'bn':
                    return 'bengali'
                elif lang == 'te':
                    return 'telugu'
                elif lang == 'th':
                    return 'thai'
                elif lang == 'ko':
                    return 'hangul'
                elif lang == 'es':
                    return 'latin'
                elif lang == 'fr':
                    return 'latin'
                elif lang == 'de':
                    return 'latin'
                elif lang == 'ru':
                    return 'cyrillic'
        
        # Check Unicode ranges
        for script, ranges in cls.UNICODE_RANGES.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    return script
        
        return 'unknown'
    
    @classmethod
    def contains_language_specific_chars(cls, text: str, lang: str) -> bool:
        """
        Check if text contains language-specific numbers or punctuation
        
        Args:
            text: Text to check
            lang: Language code
            
        Returns:
            True if contains language-specific chars
        """
        # Check for language-specific numbers
        if lang in cls.LANGUAGE_NUMBERS:
            for digit in cls.LANGUAGE_NUMBERS[lang].get('digits', []):
                if digit in text:
                    return True
        
        # Check for language-specific punctuation
        if lang in cls.LANGUAGE_PUNCTUATION:
            for mark in cls.LANGUAGE_PUNCTUATION[lang].get('marks', []):
                if mark in text:
                    return True
        
        return False
    
    @classmethod
    def detect_language_from_token(cls, token_text: str, available_languages: List[str]) -> str:
        """
        Detect language from token text using script and vocabulary analysis
        
        Args:
            token_text: Decoded token text
            available_languages: List of language codes to consider
            
        Returns:
            Language code or 'other'
        """
        if not token_text:
            return 'other'
        
        # Remove leading/trailing whitespace for analysis
        text = token_text.strip()
        if not text:
            # Pure whitespace tokens → classify by whitespace type
            return cls._classify_whitespace_token(token_text)
        
        # First, check for language-specific numbers/punctuation (strong signal)
        for lang in available_languages:
            if cls.contains_language_specific_chars(text, lang):
                return lang
        
        # Count characters by script
        script_counts = {}
        for char in text:
            script = cls.detect_script_from_char(char)
            if script != 'unknown':
                script_counts[script] = script_counts.get(script, 0) + 1
        
        if not script_counts:
            return 'other'
        
        # Get dominant script
        dominant_script = max(script_counts.items(), key=lambda x: x[1])[0]
        
        # For non-Latin scripts, use direct mapping
        if dominant_script != 'latin':
            lang = cls.SCRIPT_TO_LANGUAGE.get(dominant_script, 'other')
            return lang if lang in available_languages else 'other'
        
        # For Latin script, use vocabulary-based detection
        return cls._detect_latin_language(token_text, available_languages)
    
    @classmethod
    def _classify_whitespace_token(cls, token_text: str) -> str:
        """Classify tokens that are pure whitespace"""
        # All ASCII whitespace → 'en' (Latin-script)
        return 'en'
    
    @classmethod
    def _detect_latin_language(cls, token_text: str, available_languages: List[str]) -> str:
        """
        Detect specific Latin-script language using vocabulary
        
        Args:
            token_text: Token text
            available_languages: List of available languages
            
        Returns:
            Language code ('en', 'es', 'fr', 'de') or 'other'
        """
        text_lower = token_text.lower().strip()
        
        # Check each Latin-script language
        latin_langs = ['es', 'fr', 'de', 'en']
        scores = {lang: 0 for lang in latin_langs if lang in available_languages}
        
        if not scores:
            return 'other'
        
        for lang in scores.keys():
            lang_data = cls.LANGUAGE_SPECIFIC_TOKENS.get(lang, {})
            
            # Check for exact word matches (strong signal)
            if text_lower in lang_data.get('words', []):
                scores[lang] += 10
            
            # Check for language-specific characters (strong signal)
            for char in lang_data.get('chars', []):
                if char in text_lower:
                    scores[lang] += 5
            
            # Check for language-specific suffixes/patterns (moderate signal)
            for pattern in lang_data.get('patterns', []):
                if text_lower.endswith(pattern):
                    scores[lang] += 2
            
            for suffix in lang_data.get('suffixes', []):
                if text_lower.endswith(suffix):
                    scores[lang] += 2
        
        # If we have a clear winner, return it
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                # Return language with highest score
                return max(scores.items(), key=lambda x: x[1])[0]
        
        # Default to 'en' for Latin script if no specific markers found
        return 'en' if 'en' in available_languages else 'other'
    
    @classmethod
    def get_answer_prompt(cls, language: str) -> str:
        """Get answer prompt for a language"""
        return cls.ANSWER_PROMPTS.get(language, ' Answer:')