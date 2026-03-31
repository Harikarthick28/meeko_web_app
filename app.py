import os
import random
import base64
import json
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)
# Allow requests from your shiny new Firebase deployment!
CORS(app, origins=["https://meeko-song-player.web.app", "http://localhost:5000", "http://127.0.0.1:5000"])

# --- Configuration ---
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyCWW0PBpslyXj7z5Mkgk8UYBqB-7YAmeM4')
genai.configure(api_key=GEMINI_API_KEY)

# Lazy-loaded emotion detector (only initialized when camera is used)
_emotion_detector = None

def get_emotion_detector():
    """Lazy-load the emotion detector to avoid slow startup."""
    global _emotion_detector
    if _emotion_detector is None:
        from auto_dataset_emotion_detector import AutoDatasetEmotionDetector
        _emotion_detector = AutoDatasetEmotionDetector()
    return _emotion_detector


class MeekoEngine:
    """Core mood analysis and music recommendation engine."""

    PLATFORMS = {
        'spotify':  {'name': 'Spotify',       'color': '#6DBF8B'},
        'jiosaavn': {'name': 'JioSaavn',      'color': '#7BBFB7'},
        'youtube':  {'name': 'YouTube Music', 'color': '#D97B8F'},
        'apple':    {'name': 'Apple Music',   'color': '#C97B9E'},
        'amazon':   {'name': 'Amazon Music',  'color': '#7BB8C9'},
        'gaana':    {'name': 'Gaana',         'color': '#C98B7B'},
    }

    LANGUAGES = {
        'english':   {'name': 'English',   'flag': 'EN'},
        'hindi':     {'name': 'Hindi',     'flag': 'HI'},
        'punjabi':   {'name': 'Punjabi',   'flag': 'PA'},
        'tamil':     {'name': 'Tamil',     'flag': 'TA'},
        'telugu':    {'name': 'Telugu',    'flag': 'TE'},
        'bengali':   {'name': 'Bengali',   'flag': 'BN'},
        'marathi':   {'name': 'Marathi',   'flag': 'MR'},
        'gujarati':  {'name': 'Gujarati',  'flag': 'GU'},
        'kannada':   {'name': 'Kannada',   'flag': 'KN'},
        'malayalam': {'name': 'Malayalam', 'flag': 'ML'},
        'spanish':   {'name': 'Spanish',   'flag': 'ES'},
        'french':    {'name': 'French',    'flag': 'FR'},
        'german':    {'name': 'German',    'flag': 'DE'},
        'japanese':  {'name': 'Japanese',  'flag': 'JA'},
        'korean':    {'name': 'Korean',    'flag': 'KO'},
        'arabic':    {'name': 'Arabic',    'flag': 'AR'},
    }

    MOOD_DATA = {
        'happy': {
            'color': '#F5C26B',
            'gradient': 'linear-gradient(135deg, #F5C26B 0%, #F0A0A0 100%)',
            'playlists': {
                'english': ['Upbeat Pop', 'Dance Hits', 'Feel Good Rock', 'Happy Indie', 'Pop Classics', 'Funk & Soul'],
                'hindi': ['Bollywood Dance', 'Happy Hindi Songs', 'Punjabi Beats', 'Hindi Pop', 'Festive Songs'],
                'punjabi': ['Bhangra Hits', 'Punjabi Folk', 'Wedding Songs', 'Dhol Beats', 'Punjabi Pop'],
                'tamil': ['Tamil Kuthu', 'Gaana Songs', 'Tamil Folk', 'Celebration Songs', 'Mass Songs'],
                'telugu': ['Telugu Mass', 'Folk Songs', 'Item Numbers', 'Tollywood Hits', 'Classical Fusion'],
                'bengali': ['Rabindra Sangeet (Happy)', 'Bengali Folk', 'Adhunik Gaan', 'Pujo Songs'],
                'marathi': ['Marathi Folk', 'Lavani', 'Bhavgeet', 'Festive Marathi', 'Modern Marathi'],
                'gujarati': ['Gujarati Folk', 'Garba Songs', 'Bhajan', 'Modern Gujarati', 'Wedding Songs'],
                'kannada': ['Kannada Folk', 'Sandalwood Hits', 'Classical Kannada', 'Modern Kannada'],
                'malayalam': ['Malayalam Folk', 'Mollywood Hits', 'Mappila Songs', 'Classical Malayalam'],
                'spanish': ['Latin Pop', 'Reggaeton', 'Salsa', 'Spanish Rock', 'Flamenco Modern'],
                'french': ['French Pop', 'Chanson', 'French Electronic', 'Modern French'],
                'german': ['German Pop', 'Oktoberfest', 'German Rock', 'Electronic German'],
                'japanese': ['J-Pop', 'Anime Songs', 'Japanese Rock', 'City Pop', 'Kawaii Music'],
                'korean': ['K-Pop', 'Korean Ballads', 'K-Rock', 'Trot', 'Korean Folk'],
                'arabic': ['Arabic Pop', 'Dabke', 'Modern Arabic', 'Classical Arabic', 'Middle Eastern'],
            },
            'fortunes': [
                "Your joy is a quiet force — it moves things without making noise.",
                "Happiness isn't about having everything; it's about savoring what's here.",
                "Today your spirit is lighter than it's been in a while. Lean into that.",
                "Good moods are underrated — they change more than we give them credit for.",
            ],
        },
        'love': {
            'color': '#E8A0C0',
            'gradient': 'linear-gradient(135deg, #E8A0C0 0%, #B8A0D8 100%)',
            'playlists': {
                'english': ['Romantic Pop', 'Love Ballads', 'R&B Love Songs', 'Indie Romance', 'Classic Love Songs', 'Soulful Romance'],
                'hindi': ['Romantic Bollywood', 'Hindi Love Songs', 'Ghazals Romance', 'Sufi Love', 'Hindi Ballads'],
                'punjabi': ['Punjabi Love Songs', 'Romantic Punjabi', 'Punjabi Ballads', 'Love Duets'],
                'tamil': ['Tamil Love Songs', 'Romantic Tamil', 'Tamil Duets', 'Melody Love'],
                'telugu': ['Telugu Love Songs', 'Romantic Telugu', 'Telugu Duets', 'Love Ballads'],
                'bengali': ['Rabindra Prem', 'Bengali Love Songs', 'Romantic Bengali', 'Love Poetry'],
                'marathi': ['Marathi Love Songs', 'Romantic Marathi', 'Love Bhavgeet', 'Marathi Romance'],
                'gujarati': ['Gujarati Love Songs', 'Romantic Gujarati', 'Love Bhajans', 'Gujarati Romance'],
                'kannada': ['Kannada Love Songs', 'Romantic Kannada', 'Love Duets', 'Kannada Romance'],
                'malayalam': ['Malayalam Love Songs', 'Romantic Malayalam', 'Love Duets', 'Malayalam Romance'],
                'spanish': ['Latin Love Songs', 'Romantic Spanish', 'Bolero Romance', 'Spanish Ballads'],
                'french': ['French Romance', "Chanson d'Amour", 'Love Ballads', 'French Romantic'],
                'german': ['German Love Songs', 'Romantic German', 'German Ballads', 'Love Songs'],
                'japanese': ['J-Ballads Love', 'Romantic J-Pop', 'Love Anime Songs', 'Japanese Romance'],
                'korean': ['K-Ballads Love', 'Romantic K-Pop', 'Korean Love Songs', 'K-Drama OST'],
                'arabic': ['Arabic Love Songs', 'Romantic Arabic', 'Love Ballads', 'Arabic Romance'],
            },
            'fortunes': [
                "Love doesn't always announce itself. Sometimes it's the quiet warmth you didn't expect.",
                "The people we care about shape us more than any decision we make on our own.",
                "You're carrying something tender today — protect it, but don't hide it.",
                "Connection is the most human thing there is. You're feeling it deeply right now.",
            ],
        },
        'romantic': {
            'color': '#E8A0B8',
            'gradient': 'linear-gradient(135deg, #E8A0B8 0%, #D8A0C0 100%)',
            'playlists': {
                'english': ['Romantic Pop', 'Love Ballads', 'R&B Love Songs', 'Indie Romance', 'Classic Love Songs', 'Soulful Romance'],
                'hindi': ['Romantic Bollywood', 'Hindi Love Songs', 'Ghazals Romance', 'Sufi Love', 'Hindi Ballads'],
                'punjabi': ['Punjabi Love Songs', 'Romantic Punjabi', 'Punjabi Ballads', 'Love Duets'],
                'tamil': ['Tamil Love Songs', 'Romantic Tamil', 'Tamil Duets', 'Melody Love'],
                'telugu': ['Telugu Love Songs', 'Romantic Telugu', 'Telugu Duets', 'Love Ballads'],
                'bengali': ['Rabindra Prem', 'Bengali Love Songs', 'Romantic Bengali', 'Love Poetry'],
                'marathi': ['Marathi Love Songs', 'Romantic Marathi', 'Love Bhavgeet', 'Marathi Romance'],
                'gujarati': ['Gujarati Love Songs', 'Romantic Gujarati', 'Love Bhajans', 'Gujarati Romance'],
                'kannada': ['Kannada Love Songs', 'Romantic Kannada', 'Love Duets', 'Kannada Romance'],
                'malayalam': ['Malayalam Love Songs', 'Romantic Malayalam', 'Love Duets', 'Malayalam Romance'],
                'spanish': ['Latin Love Songs', 'Romantic Spanish', 'Bolero Romance', 'Spanish Ballads'],
                'french': ['French Romance', "Chanson d'Amour", 'Love Ballads', 'French Romantic'],
                'german': ['German Love Songs', 'Romantic German', 'German Ballads', 'Love Songs'],
                'japanese': ['J-Ballads Love', 'Romantic J-Pop', 'Love Anime Songs', 'Japanese Romance'],
                'korean': ['K-Ballads Love', 'Romantic K-Pop', 'Korean Love Songs', 'K-Drama OST'],
                'arabic': ['Arabic Love Songs', 'Romantic Arabic', 'Love Ballads', 'Arabic Romance'],
            },
            'fortunes': [
                "Romance lives in the small details — a look, a pause, a shared silence.",
                "Tonight feels like it was written for you. Let it unfold.",
                "The best love stories are the ones that surprise you.",
                "There's poetry in how you're feeling right now. Don't rush past it.",
            ],
        },
        'excited': {
            'color': '#F0B080',
            'gradient': 'linear-gradient(135deg, #F0B080 0%, #E8D090 100%)',
            'playlists': {
                'english': ['High Energy Pop', 'Electronic Dance', 'Pump Up Songs', 'Party Anthems', 'Upbeat Rock'],
                'hindi': ['Bollywood Party', 'High Energy Hindi', 'Dance Numbers', 'Festive Bollywood'],
                'punjabi': ['Bhangra Energy', 'Party Punjabi', 'High Beat Punjabi', 'Dance Punjabi'],
                'tamil': ['Tamil Energy', 'Kuthu Dance', 'High Beat Tamil', 'Party Tamil'],
                'telugu': ['Telugu Energy', 'Mass Numbers', 'High Beat Telugu', 'Party Telugu'],
                'bengali': ['Bengali Energy', 'Party Bengali', 'High Beat Bengali', 'Dance Bengali'],
                'marathi': ['Marathi Energy', 'Party Marathi', 'High Beat Marathi', 'Dance Marathi'],
                'gujarati': ['Gujarati Energy', 'Garba High Energy', 'Party Gujarati', 'Dance Gujarati'],
                'kannada': ['Kannada Energy', 'Party Kannada', 'High Beat Kannada', 'Dance Kannada'],
                'malayalam': ['Malayalam Energy', 'Party Malayalam', 'High Beat Malayalam', 'Dance Malayalam'],
                'spanish': ['Reggaeton Energy', 'Latin Dance', 'High Energy Spanish', 'Party Latin'],
                'french': ['French Electronic', 'High Energy French', 'Party French', 'Dance French'],
                'german': ['German Electronic', 'High Energy German', 'Party German', 'Dance German'],
                'japanese': ['High Energy J-Pop', 'Dance Japanese', 'Party J-Music', 'Anime Energy'],
                'korean': ['High Energy K-Pop', 'Dance Korean', 'Party K-Music', 'K-Pop Energy'],
                'arabic': ['High Energy Arabic', 'Dance Arabic', 'Party Arabic', 'Arabic Energy'],
            },
            'fortunes': [
                "This kind of energy is rare — channel it before it fades.",
                "Excitement is your body telling you something matters. Pay attention.",
                "You're on the edge of something good. Trust the momentum.",
                "The best ideas come when you feel like this. Write them down.",
            ],
        },
        'sad': {
            'color': '#A0A8E0',
            'gradient': 'linear-gradient(135deg, #A0A8E0 0%, #90B8E8 100%)',
            'playlists': {
                'english': ['Acoustic Comfort', 'Gentle Piano', 'Uplifting Indie', 'Healing Songs', 'Soft Rock', 'Emotional Ballads'],
                'hindi': ['Sad Hindi Songs', 'Ghazals', 'Sufi Music', 'Hindi Ballads', 'Emotional Bollywood'],
                'punjabi': ['Punjabi Sad Songs', 'Sufi Punjabi', 'Punjabi Ghazals', 'Emotional Punjabi'],
                'tamil': ['Tamil Melody', 'Sad Tamil Songs', 'Tamil Ghazals', 'Emotional Tamil'],
                'telugu': ['Telugu Sad Songs', 'Emotional Telugu', 'Telugu Ballads', 'Melancholic Telugu'],
                'bengali': ['Rabindra Sangeet (Sad)', 'Bengali Ballads', 'Nazrul Geeti', 'Emotional Bengali'],
                'marathi': ['Marathi Ballads', 'Emotional Marathi', 'Marathi Ghazals', 'Sad Marathi'],
                'gujarati': ['Gujarati Ballads', 'Emotional Gujarati', 'Gujarati Ghazals', 'Sad Gujarati'],
                'kannada': ['Kannada Ballads', 'Emotional Kannada', 'Sad Kannada Songs', 'Melancholic Kannada'],
                'malayalam': ['Malayalam Ballads', 'Emotional Malayalam', 'Sad Malayalam', 'Melancholic Malayalam'],
                'spanish': ['Spanish Ballads', 'Bolero', 'Emotional Spanish', 'Latin Ballads'],
                'french': ['French Ballads', 'Chanson Triste', 'Emotional French', 'French Acoustic'],
                'german': ['German Ballads', 'Emotional German', 'German Acoustic', 'Melancholic German'],
                'japanese': ['Japanese Ballads', 'Emotional J-Pop', 'Sad Anime Songs', 'Japanese Acoustic'],
                'korean': ['Korean Ballads', 'Emotional K-Pop', 'K-Ballads', 'Sad Korean Songs'],
                'arabic': ['Arabic Ballads', 'Emotional Arabic', 'Classical Arabic Sad', 'Middle Eastern Ballads'],
            },
            'fortunes': [
                "Sadness isn't weakness — it means something mattered to you.",
                "You don't have to fix how you feel. Sometimes you just sit with it.",
                "The heaviest days teach us the most about ourselves.",
                "Even slow rivers eventually find the sea. Give yourself time.",
            ],
        },
        'stressed': {
            'color': '#B8A0D8',
            'gradient': 'linear-gradient(135deg, #B8A0D8 0%, #A0A8E0 100%)',
            'playlists': {
                'english': ['Nature Sounds', 'Meditation Music', 'Calm Instrumentals', 'Lo-fi Beats', 'Ambient', 'Relaxing Piano'],
                'hindi': ['Meditation Mantras', 'Peaceful Hindi', 'Yoga Music', 'Spiritual Hindi', 'Calming Bollywood'],
                'punjabi': ['Peaceful Punjabi', 'Punjabi Mantras', 'Calming Punjabi', 'Spiritual Punjabi'],
                'tamil': ['Classical Ragas Tamil', 'Peaceful Tamil', 'Tamil Instrumentals', 'Spiritual Tamil'],
                'telugu': ['Carnatic Peaceful', 'Telugu Instrumentals', 'Peaceful Telugu', 'Spiritual Telugu'],
                'bengali': ['Rabindra Sangeet (Peaceful)', 'Bengali Classical', 'Peaceful Bengali', 'Meditation Bengali'],
                'marathi': ['Marathi Classical', 'Peaceful Marathi', 'Marathi Bhajans', 'Calming Marathi'],
                'gujarati': ['Gujarati Bhajans', 'Peaceful Gujarati', 'Gujarati Mantras', 'Calming Gujarati'],
                'kannada': ['Kannada Classical', 'Peaceful Kannada', 'Kannada Bhajans', 'Calming Kannada'],
                'malayalam': ['Malayalam Classical', 'Peaceful Malayalam', 'Malayalam Bhajans', 'Calming Malayalam'],
                'spanish': ['Spanish Acoustic', 'Peaceful Spanish', 'Spanish Instrumentals', 'Relaxing Spanish'],
                'french': ['French Ambient', 'Peaceful French', 'French Instrumentals', 'Relaxing French'],
                'german': ['German Ambient', 'Peaceful German', 'German Instrumentals', 'Relaxing German'],
                'japanese': ['Japanese Ambient', 'Peaceful J-Music', 'Japanese Instrumentals', 'Zen Music'],
                'korean': ['Korean Ambient', 'Peaceful K-Music', 'Korean Instrumentals', 'Relaxing Korean'],
                'arabic': ['Arabic Instrumentals', 'Peaceful Arabic', 'Arabic Ambient', 'Spiritual Arabic'],
            },
            'fortunes': [
                "You can't pour from an empty cup. Rest isn't optional.",
                "Stress means you care — but caring shouldn't cost you your peace.",
                "One thing at a time. That's all anyone can ever really do.",
                "The noise will quiet down. In the meantime, let the music do the thinking.",
            ],
        },
        'angry': {
            'color': '#E0A0A0',
            'gradient': 'linear-gradient(135deg, #E0A0A0 0%, #D8908A 100%)',
            'playlists': {
                'english': ['Heavy Metal', 'Punk Rock', 'Aggressive Electronic', 'Hard Rock', 'Alternative Rock', 'Grunge'],
                'hindi': ['Angry Rap Hindi', 'Rock Hindi', 'Intense Bollywood', 'Hindi Metal', 'Aggressive Hindi'],
                'punjabi': ['Punjabi Rock', 'Aggressive Punjabi', 'Punjabi Rap', 'Intense Punjabi'],
                'tamil': ['Tamil Rock', 'Aggressive Tamil', 'Tamil Rap', 'Intense Tamil'],
                'telugu': ['Telugu Mass', 'Telugu Rock', 'Aggressive Telugu', 'Telugu Rap'],
                'bengali': ['Bengali Rock', 'Aggressive Bengali', 'Bengali Rap', 'Intense Bengali'],
                'marathi': ['Marathi Rock', 'Aggressive Marathi', 'Marathi Rap', 'Intense Marathi'],
                'gujarati': ['Gujarati Rock', 'Aggressive Gujarati', 'Gujarati Rap', 'Intense Gujarati'],
                'kannada': ['Kannada Rock', 'Aggressive Kannada', 'Kannada Rap', 'Intense Kannada'],
                'malayalam': ['Malayalam Rock', 'Aggressive Malayalam', 'Malayalam Rap', 'Intense Malayalam'],
                'spanish': ['Spanish Rock', 'Latin Metal', 'Spanish Punk', 'Aggressive Spanish'],
                'french': ['French Rock', 'French Metal', 'French Punk', 'Aggressive French'],
                'german': ['German Metal', 'German Rock', 'German Punk', 'Industrial German'],
                'japanese': ['Japanese Rock', 'J-Metal', 'Japanese Punk', 'Visual Kei'],
                'korean': ['K-Rock', 'Korean Metal', 'Korean Punk', 'Aggressive K-Pop'],
                'arabic': ['Arabic Rock', 'Middle Eastern Metal', 'Aggressive Arabic', 'Arabic Fusion'],
            },
            'fortunes': [
                "Anger is just passion with nowhere to go yet. Find the door.",
                "You're feeling something real. That's not the problem — what you do next is the question.",
                "Storms clear the air. Let this one pass through you, not settle in.",
                "The strongest people aren't the ones who never get angry — they're the ones who choose what to do with it.",
            ],
        },
        'neutral': {
            'color': '#A8B0C0',
            'gradient': 'linear-gradient(135deg, #A8B0C0 0%, #98A8B8 100%)',
            'playlists': {
                'english': ['Chill Vibes', 'Background Jazz', 'Ambient', 'Soft Rock', 'Indie Folk', 'Easy Listening'],
                'hindi': ['Chill Hindi', 'Soft Bollywood', 'Easy Listening Hindi', 'Hindi Jazz', 'Peaceful Hindi'],
                'punjabi': ['Chill Punjabi', 'Soft Punjabi', 'Punjabi Folk', 'Easy Punjabi'],
                'tamil': ['Soft Tamil', 'Tamil Folk', 'Chill Tamil', 'Easy Tamil'],
                'telugu': ['Soft Telugu', 'Telugu Folk', 'Chill Telugu', 'Easy Telugu'],
                'bengali': ['Soft Bengali', 'Bengali Folk', 'Chill Bengali', 'Easy Bengali'],
                'marathi': ['Soft Marathi', 'Marathi Folk', 'Chill Marathi', 'Easy Marathi'],
                'gujarati': ['Soft Gujarati', 'Gujarati Folk', 'Chill Gujarati', 'Easy Gujarati'],
                'kannada': ['Soft Kannada', 'Kannada Folk', 'Chill Kannada', 'Easy Kannada'],
                'malayalam': ['Soft Malayalam', 'Malayalam Folk', 'Chill Malayalam', 'Easy Malayalam'],
                'spanish': ['Chill Spanish', 'Spanish Acoustic', 'Easy Spanish', 'Spanish Folk'],
                'french': ['Chill French', 'French Acoustic', 'Easy French', 'French Folk'],
                'german': ['Chill German', 'German Acoustic', 'Easy German', 'German Folk'],
                'japanese': ['Chill J-Pop', 'Japanese Acoustic', 'Easy Japanese', 'Japanese Folk'],
                'korean': ['Chill K-Pop', 'Korean Acoustic', 'Easy Korean', 'Korean Folk'],
                'arabic': ['Chill Arabic', 'Arabic Acoustic', 'Easy Arabic', 'Arabic Folk'],
            },
            'fortunes': [
                "Not every moment needs to be intense. There's wisdom in the in-between.",
                "A calm mind notices what a busy one misses.",
                "Neutral isn't empty — it's open. You get to choose what fills it.",
                "Sometimes the best thing you can do is simply be present.",
            ],
        },
    }

    def detect_face_emotion(self, image_data):
        """Detect emotion from a base64-encoded camera image."""
        try:
            detector = get_emotion_detector()
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            emotions = detector.detect_emotion_advanced(frame)
            if emotions:
                result = emotions[0]
                return {
                    'emotion': result['emotion'],
                    'confidence': round(result['confidence'], 3),
                    'extended_emotion': result.get('extended_emotion', result['emotion']),
                    'raw_scores': result.get('all_scores', {}),
                    'accuracy_level': (
                        'high' if result['confidence'] > 0.7
                        else 'medium' if result['confidence'] > 0.5
                        else 'low'
                    ),
                }
            return {'emotion': 'neutral', 'confidence': 0.0, 'accuracy_level': 'no_face'}

        except Exception as e:
            print(f"Face detection error: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0, 'accuracy_level': 'error'}

    def analyze_text_mood(self, text):
        """Use Gemini AI to analyze emotional mood with retry and fallback."""
        if not text or not text.strip():
            return 'neutral'

        prompt = f"""You are an empathetic emotion analyst. Read the following text carefully and determine the person's true emotional state. Consider tone, context, subtext, and cultural expressions (including Hindi, Hinglish, and regional Indian languages).

Text: "{text}"

Classify into EXACTLY ONE of these categories:
- happy (joy, contentment, gratitude, feeling good, celebration, satisfaction)
- sad (sadness, grief, loss, loneliness, heartbreak, missing someone, feeling empty)
- angry (anger, frustration, rage, irritation, resentment, being fed up)
- stressed (anxiety, worry, overwhelm, pressure, tension, burnout, nervousness)
- excited (anticipation, thrill, high energy, eagerness, enthusiasm, hype)
- love (romantic love, being in love, deep affection for a person, missing a partner)
- romantic (wanting romance, date vibes, intimacy, flirting, romantic mood)
- neutral (calm, indifferent, factual, or genuinely unclear emotion)

IMPORTANT context rules:
- "I love coding" / "I love pizza" = happy (not romantic love)
- "I'm in love with her" / "I miss my boyfriend" = love
- "feeling low" / "not feeling great" / "kinda down" = sad
- "I can't take this anymore" / "so done with everything" = angry OR stressed (pick dominant)
- "what should I eat" / "just woke up" = neutral
- "mujhe bahut tension ho rahi hai" = stressed
- "bahut khush hu" / "I'm so happy" = happy
- "dil toot gaya" / "heartbroken" = sad
- "I want to go on a date" = romantic
- Profanity or aggressive tone = angry
- Short ambiguous inputs like "hi" or "ok" = neutral

Reply with ONLY the single mood word. No explanation, no punctuation."""

        # Try primary model, then fallback model
        models_to_try = ['gemini-2.0-flash', 'gemini-2.0-flash-lite']
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                ai_mood = response.text.strip().lower().split()[0]  # Take first word only
                # Remove any punctuation
                ai_mood = ''.join(c for c in ai_mood if c.isalpha())

                valid_moods = list(self.MOOD_DATA.keys())
                if ai_mood in valid_moods:
                    print(f"Mood detected ({model_name}): {ai_mood}")
                    return ai_mood
            except Exception as e:
                print(f"Gemini ({model_name}) error: {e}")
                continue

        # All AI models failed — use comprehensive fallback
        print("AI unavailable, using NLP fallback")
        return self._fallback_mood_analysis(text)

    def _fallback_mood_analysis(self, text):
        """Comprehensive NLP-based fallback with phrase matching and negation detection."""
        text_lower = text.lower().strip()

        # --- Phase 1: Exact phrase matching (highest priority) ---
        phrase_map = {
            'love': [
                'in love', 'i love you', 'love you', 'i love her', 'i love him',
                'my love', 'falling in love', 'miss you', 'miss him', 'miss her',
                'thinking about him', 'thinking about her', 'pyaar', 'mohabbat',
                'ishq', 'dil de diya', 'tujhe chahta', 'tujhe chahti',
            ],
            'romantic': [
                'date night', 'candle light', 'candlelight dinner', 'romantic evening',
                'want to go on a date', 'roses', 'romantic mood', 'romance',
                'feeling romantic', 'intimate', 'flirting', 'cuddle',
            ],
            'excited': [
                "can't wait", 'so excited', 'really excited', 'super excited',
                'pumped up', 'looking forward', 'hyped up', 'bring it on',
                'let\'s go', "i'm ready", 'finally happening', 'omg',
                'bahut excited', 'maza aayega',
            ],
            'sad': [
                'feeling low', 'feeling down', 'not feeling great', 'not feeling good',
                'feeling empty', 'feel like crying', 'want to cry', 'heartbroken',
                'heart broken', 'miss my', 'so lonely', 'feel alone', 'all alone',
                'no one cares', 'nobody cares', 'lost someone', 'passed away',
                'feeling lost', 'kinda down', 'not okay', "i'm not ok",
                'dil toot gaya', 'bahut dukhi', 'rona aa raha', 'akela',
                'udaas', 'dukhi', 'tanha',
            ],
            'stressed': [
                'so stressed', 'too much pressure', 'can\'t handle', 'overwhelmed',
                'anxiety', 'panic attack', 'burning out', 'burnt out', 'burnout',
                'too much work', 'deadline', 'can\'t sleep', 'sleepless',
                'tension', 'bahut tension', 'pareshan', 'chinta',
                'dar lag raha', 'nervous', 'worried sick',
            ],
            'angry': [
                'so angry', 'pissed off', 'fed up', 'sick of', 'tired of this',
                'can\'t take it', 'done with this', 'hate this', 'hate everything',
                'so annoyed', 'frustrating', 'makes me mad', 'want to scream',
                'gussa', 'bahut gussa', 'irritating',
                'i\'m done', "i'm so done", 'enough is enough',
            ],
            'happy': [
                'feeling great', 'feeling good', 'feeling amazing', 'feeling awesome',
                'so happy', 'really happy', 'best day', 'wonderful day', 'great day',
                'love life', 'grateful', 'thankful', 'blessed', 'on top of the world',
                'couldn\'t be better', 'living my best', 'bahut khush',
                'maza aa gaya', 'bohot acha', 'feeling blessed',
            ],
        }

        phrase_scores = {mood: 0 for mood in phrase_map}
        for mood, phrases in phrase_map.items():
            for phrase in phrases:
                if phrase in text_lower:
                    phrase_scores[mood] += 3  # Phrases get high weight

        best_phrase = max(phrase_scores, key=phrase_scores.get)
        if phrase_scores[best_phrase] >= 3:
            return best_phrase

        # --- Phase 2: Weighted keyword scoring ---
        keywords = {
            'happy': {
                3: ['happy', 'joy', 'joyful', 'elated', 'cheerful', 'delighted', 'blissful'],
                2: ['great', 'amazing', 'wonderful', 'fantastic', 'awesome', 'excellent',
                    'brilliant', 'good', 'nice', 'pleased', 'glad', 'content', 'satisfied'],
                1: ['fine', 'okay', 'cool', 'sweet', 'yay', 'woohoo', 'smile', 'laugh',
                    'fun', 'enjoy', 'celebrated', 'party', 'won', 'success', 'achievement'],
            },
            'sad': {
                3: ['sad', 'depressed', 'devastated', 'miserable', 'heartbroken', 'grief'],
                2: ['upset', 'crying', 'tears', 'hurt', 'broken', 'lonely', 'alone',
                    'empty', 'numb', 'hopeless', 'helpless', 'lost', 'pain', 'suffering'],
                1: ['miss', 'missing', 'down', 'low', 'blue', 'gloomy', 'melancholy',
                    'disappointed', 'regret', 'sorry', 'failed', 'sigh', 'heavy'],
            },
            'angry': {
                3: ['angry', 'furious', 'enraged', 'livid', 'outraged', 'infuriated'],
                2: ['mad', 'frustrated', 'annoyed', 'irritated', 'pissed', 'hate', 'rage',
                    'disgusted', 'resentful', 'hostile', 'bitter'],
                1: ['bothered', 'agitated', 'displeased', 'offended', 'provoked',
                    'damn', 'ugh', 'wtf', 'stupid', 'idiot', 'worst'],
            },
            'stressed': {
                3: ['stressed', 'anxious', 'overwhelmed', 'panicking', 'terrified'],
                2: ['worried', 'nervous', 'tense', 'pressure', 'burden', 'exhausted',
                    'drained', 'burned', 'overworked', 'restless', 'uneasy'],
                1: ['busy', 'hectic', 'rush', 'deadline', 'chaos', 'mess', 'confused',
                    'uncertain', 'scared', 'afraid', 'fear', 'insomnia', 'overthinking'],
            },
            'excited': {
                3: ['excited', 'thrilled', 'ecstatic', 'stoked', 'exhilarated'],
                2: ['pumped', 'hyped', 'eager', 'enthusiastic', 'fired', 'buzzing',
                    'anticipating', 'psyched', 'amped'],
                1: ['ready', 'waiting', 'upcoming', 'soon', 'finally', 'wow',
                    'incredible', 'unbelievable', 'surprise'],
            },
            'love': {
                3: ['love', 'adore', 'cherish', 'soulmate', 'beloved'],
                2: ['crush', 'attracted', 'infatuated', 'devoted', 'affection',
                    'darling', 'sweetheart', 'babe', 'baby'],
                1: ['heart', 'forever', 'together', 'kiss', 'hug', 'embrace',
                    'butterflies', 'chemistry'],
            },
            'romantic': {
                3: ['romantic', 'romance', 'intimate', 'passionate'],
                2: ['date', 'candlelight', 'roses', 'flirt', 'seduce',
                    'desire', 'longing', 'yearning'],
                1: ['evening', 'together', 'dinner', 'moonlight', 'sunset',
                    'wine', 'cozy', 'warmth'],
            },
        }

        # Check for negation patterns
        negation_words = ['not', "n't", 'no', 'never', 'neither', 'hardly', 'barely', "don't", "doesn't", "isn't", "wasn't", "aren't"]
        words = text_lower.split()
        has_negation = any(neg in text_lower for neg in negation_words)

        scores = {mood: 0.0 for mood in keywords}
        for mood, weight_groups in keywords.items():
            for weight, word_list in weight_groups.items():
                for word in word_list:
                    if word in text_lower:
                        # Check if the word is negated (within 3 words before)
                        try:
                            idx = words.index(word)
                            context_before = words[max(0, idx-3):idx]
                            is_negated = any(neg in ' '.join(context_before) for neg in negation_words)
                        except ValueError:
                            is_negated = False

                        if is_negated:
                            # "not happy" → boost opposite moods
                            if mood == 'happy':
                                scores['sad'] += weight * 0.5
                            elif mood == 'sad':
                                scores['happy'] += weight * 0.3
                        else:
                            scores[mood] += weight

        # Disambiguate: "love" used casually vs romantically
        if scores.get('love', 0) > 0:
            casual_love_signals = ['love it', 'love this', 'love that', 'love food',
                                   'love coding', 'love music', 'love pizza', 'love my job']
            if any(sig in text_lower for sig in casual_love_signals):
                scores['happy'] += scores['love'] * 0.7
                scores['love'] *= 0.2

        # Intensity modifiers boost the top score
        intensifiers = ['very', 'really', 'so', 'extremely', 'super', 'absolutely',
                        'incredibly', 'totally', 'completely', 'utterly', 'bahut', 'bohot']
        intensity_count = sum(1 for word in words if word in intensifiers)
        if intensity_count > 0:
            best_mood = max(scores, key=scores.get)
            if scores[best_mood] > 0:
                scores[best_mood] *= (1 + 0.3 * intensity_count)

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else 'neutral'

    def get_mood_response(self, mood, selected_languages=None):
        """Build the full response for a detected mood."""
        mood_info = self.MOOD_DATA.get(mood, self.MOOD_DATA['neutral'])

        if selected_languages:
            filtered = {lang: mood_info['playlists'][lang]
                        for lang in selected_languages if lang in mood_info['playlists']}
            all_playlists = [p for pl in filtered.values() for p in pl]
        else:
            filtered = mood_info['playlists']
            all_playlists = [p for pl in filtered.values() for p in pl]

        return {
            'mood': mood,
            'color': mood_info['color'],
            'gradient': mood_info['gradient'],
            'playlists': filtered,
            'all_playlists': all_playlists,
            'fortune': random.choice(mood_info['fortunes']),
            'platforms': self.PLATFORMS,
            'available_languages': self.LANGUAGES,
            'selected_languages': selected_languages or [],
        }


engine = MeekoEngine()


# --- Routes ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/app')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/analyze', methods=['POST'])
def analyze_mood():
    try:
        data = request.json or {}
        text = data.get('text', '')
        selected_languages = data.get('languages', [])
        image_data = data.get('image', '')

        # Text analysis (primary signal)
        text_mood = engine.analyze_text_mood(text)

        # Face analysis (secondary signal, only if image provided)
        face_result = None
        face_mood = 'neutral'
        if image_data:
            face_result = engine.detect_face_emotion(image_data)
            face_mood = face_result.get('emotion', 'neutral')

        # Priority: text > high-confidence face > neutral
        if text and text.strip():
            final_mood = text_mood
        elif face_result and face_result.get('confidence', 0) > 0.6:
            final_mood = face_mood
        else:
            final_mood = text_mood if text else 'neutral'

        response = engine.get_mood_response(final_mood, selected_languages)
        response['signals'] = {
            'text': text_mood,
            'face': face_mood,
            'face_confidence': face_result.get('confidence', 0) if face_result else None,
        }
        return jsonify(response)

    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/capture-face', methods=['POST'])
def capture_face():
    try:
        data = request.json or {}
        image_data = data.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        result = engine.detect_face_emotion(image_data)
        return jsonify({
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'accuracy': f"{result['confidence'] * 100:.1f}%",
            'accuracy_level': result.get('accuracy_level', 'unknown'),
        })

    except Exception as e:
        print(f"Face capture error: {e}")
        return jsonify({'error': 'Face detection failed'}), 500

@app.route('/languages', methods=['GET'])
def get_languages():
    return jsonify(engine.LANGUAGES)


if __name__ == '__main__':
    app.run(debug=True)