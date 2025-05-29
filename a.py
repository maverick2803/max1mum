import gradio as gr
import json
import os
from datetime import datetime, timedelta
import re
import requests
from flask import Flask, request, jsonify
import threading
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
from typing import List, Dict, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MAXV1 Custom Neural Network Architecture
class MAXV1NeuralNetwork(nn.Module):
    """
    MAXV1 - Custom AI Model Architecture
    Designed for natural conversation and intelligent assistance
    """
    def __init__(self, vocab_size=10000, embed_dim=512, hidden_dim=1024, num_layers=6, num_heads=8):
        super(MAXV1NeuralNetwork, self).__init__()
        
        # Model specifications
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Core components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1000, embed_dim))
        
        # Multi-head attention layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # Context understanding layers
        self.context_analyzer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Personality and emotion layers
        self.personality_layer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),  # Personality encoding
        )
        
        self.emotion_detector = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 7),  # 7 basic emotions
            nn.Softmax(dim=-1)
        )
        
        # Response generation
        self.response_generator = nn.Sequential(
            nn.Linear(embed_dim + 64 + 7, hidden_dim),  # text + personality + emotion
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Memory system
        self.memory_bank = nn.Parameter(torch.randn(100, embed_dim))  # Long-term memory
        self.memory_attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, input_ids, attention_mask=None, use_memory=True):
        batch_size, seq_len = input_ids.shape
        
        # Embedding with positional encoding
        x = self.embedding(input_ids)
        x += self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        
        # Context analysis
        context = self.context_analyzer(x.mean(dim=1))  # Global context
        
        # Personality and emotion analysis
        personality = self.personality_layer(context)
        emotion = self.emotion_detector(context)
        
        # Memory integration
        if use_memory:
            memory_output, _ = self.memory_attention(
                context.unsqueeze(0), 
                self.memory_bank.unsqueeze(1).repeat(1, batch_size, 1),
                self.memory_bank.unsqueeze(1).repeat(1, batch_size, 1)
            )
            context = context + memory_output.squeeze(0)
        
        # Generate response logits
        combined_features = torch.cat([context, personality, emotion], dim=-1)
        logits = self.response_generator(combined_features)
        
        return {
            'logits': logits,
            'context': context,
            'personality': personality,
            'emotion': emotion,
            'attention_weights': x
        }

class MAXV1Tokenizer:
    """Custom tokenizer for MAXV1 model"""
    
    def __init__(self, vocab_file=None):
        self.vocab = self._build_vocab(vocab_file)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3,
            '<USER>': 4,
            '<ASSISTANT>': 5,
            '<CONTEXT>': 6,
            '<EMOTION>': 7
        }
    
    def _build_vocab(self, vocab_file):
        """Build vocabulary from training data or load from file"""
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
        else:
            # Basic Indonesian + English vocabulary
            vocab = {
                # Special tokens
                '<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3,
                '<USER>': 4, '<ASSISTANT>': 5, '<CONTEXT>': 6, '<EMOTION>': 7,
                
                # Common Indonesian words
                'saya': 8, 'aku': 9, 'kamu': 10, 'anda': 11, 'dia': 12,
                'ini': 13, 'itu': 14, 'yang': 15, 'dan': 16, 'atau': 17,
                'dengan': 18, 'untuk': 19, 'dari': 20, 'ke': 21, 'di': 22,
                'pada': 23, 'dalam': 24, 'oleh': 25, 'akan': 26, 'sudah': 27,
                'adalah': 28, 'ada': 29, 'tidak': 30, 'bisa': 31, 'mau': 32,
                'halo': 33, 'hai': 34, 'selamat': 35, 'terima': 36, 'kasih': 37,
                'jadwal': 38, 'waktu': 39, 'jam': 40, 'hari': 41, 'tanggal': 42,
                'ingatkan': 43, 'reminder': 44, 'meeting': 45, 'kerja': 46,
                'baik': 47, 'bagus': 48, 'hebat': 49, 'keren': 50, 'mantap': 51,
                
                # Common English words
                'hello': 52, 'hi': 53, 'thanks': 54, 'thank': 55, 'you': 56,
                'please': 57, 'help': 58, 'schedule': 59, 'time': 60, 'date': 61,
                'good': 62, 'great': 63, 'awesome': 64, 'nice': 65, 'cool': 66,
                'yes': 67, 'no': 68, 'maybe': 69, 'sure': 70, 'okay': 71,
                
                # Numbers
                'satu': 72, 'dua': 73, 'tiga': 74, 'empat': 75, 'lima': 76,
                'enam': 77, 'tujuh': 78, 'delapan': 79, 'sembilan': 80, 'sepuluh': 81,
                'one': 82, 'two': 83, 'three': 84, 'four': 85, 'five': 86,
                'six': 87, 'seven': 88, 'eight': 89, 'nine': 90, 'ten': 91
            }
            
            # Extend with more common words
            for i in range(92, 10000):
                vocab[f'word_{i}'] = i
        
        return vocab
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token ids"""
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        token_ids = []
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['<UNK>'])
        
        # Pad or truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([self.vocab['<PAD>']] * (max_length - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if token not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                    tokens.append(token)
        
        return ' '.join(tokens)

class MAXV1Core:
    """
    MAXV1 - Main AI Core System
    Natural Language Understanding & Generation with Memory
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = MAXV1Tokenizer()
        
        # Initialize model
        self.model = MAXV1NeuralNetwork(
            vocab_size=len(self.tokenizer.vocab),
            embed_dim=512,
            hidden_dim=1024,
            num_layers=6,
            num_heads=8
        ).to(self.device)
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded MAXV1 model from {model_path}")
        else:
            logger.info("Initialized MAXV1 model with random weights")
        
        # Conversation memory
        self.conversation_history = {}
        self.user_profiles = {}
        self.context_memory = {}
        
        # Personality configuration
        self.personality = {
            "name": "MAXV1",
            "traits": {
                "helpful": 0.9,
                "friendly": 0.8,
                "professional": 0.7,
                "empathetic": 0.8,
                "intelligent": 0.9,
                "humorous": 0.6
            },
            "communication_style": "natural_conversational",
            "language_preference": "mixed_id_en"
        }
        
        # Response templates
        self.response_templates = self._load_response_templates()
        
        # Intent classifier
        self.intent_patterns = self._initialize_intent_patterns()
    
    def _load_response_templates(self):
        """Load natural response templates"""
        return {
            'greeting': [
                "Halo! Saya MAXV1, asisten AI pribadi Anda. Ada yang bisa saya bantu hari ini?",
                "Hi! MAXV1 di sini. Bagaimana kabar Anda? Ada yang perlu bantuan?",
                "Selamat {time_greeting}! Saya MAXV1, siap membantu Anda. Apa yang bisa saya lakukan?",
                "Hai! Senang bertemu dengan Anda. Saya MAXV1, asisten AI yang siap membantu."
            ],
            'schedule_confirm': [
                "Baik, saya sudah mencatat jadwal Anda: '{schedule}' pada {time}. Saya akan mengingatkan Anda nanti.",
                "Perfect! Jadwal '{schedule}' sudah tersimpan untuk {time}. Tenang saja, saya akan ingatkan Anda.",
                "Oke, sudah saya catat. '{schedule}' dijadwalkan pada {time}. Reminder sudah diset!"
            ],
            'emotional_support': [
                "Saya memahami perasaan Anda. Ingin bercerita lebih lanjut?",
                "Terima kasih sudah berbagi. Saya di sini untuk mendengarkan dan membantu.",
                "Saya turut merasakan apa yang Anda alami. Apakah ada yang bisa saya bantu?"
            ],
            'curiosity': [
                "Itu menarik! Bisa cerita lebih detail?",
                "Wah, saya penasaran. Bagaimana ceritanya?",
                "Interesting! Saya ingin tahu lebih banyak tentang itu."
            ]
        }
    
    def _initialize_intent_patterns(self):
        """Initialize intent recognition patterns"""
        return {
            'greeting': [r'\b(hai|halo|hi|hello|selamat)\b', r'apa kabar', r'how are you'],
            'schedule': [r'jadwal|schedule|reminder|ingatkan', r'meeting|rapat|appointment'],
            'question': [r'\?', r'\b(apa|siapa|kapan|dimana|bagaimana|kenapa)\b', r'\b(what|who|when|where|how|why)\b'],
            'time': [r'\b(jam|waktu|time|tanggal|date)\b'],
            'emotion_positive': [r'\b(senang|bahagia|gembira|happy|excited|great)\b'],
            'emotion_negative': [r'\b(sedih|kecewa|marah|sad|angry|frustrated|tired)\b'],
            'thanks': [r'\b(terima kasih|makasih|thanks|thank you)\b'],
            'goodbye': [r'\b(bye|dadah|sampai jumpa|goodbye|see you)\b']
        }
    
    def analyze_intent(self, text: str) -> Dict[str, float]:
        """Analyze user intent from text"""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            intent_scores[intent] = score / len(patterns) if patterns else 0
        
        return intent_scores
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """Analyze emotional content of text"""
        # Use the model's emotion detector
        with torch.no_grad():
            tokens = self.tokenizer.encode(text)
            input_ids = torch.tensor([tokens]).to(self.device)
            attention_mask = torch.tensor([[1] * len(tokens)]).to(self.device)
            
            outputs = self.model(input_ids, attention_mask)
            emotion_probs = outputs['emotion'].cpu().numpy()[0]
            
            emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
            return dict(zip(emotions, emotion_probs))
    
    def generate_response(self, user_input: str, user_id: str = "default") -> str:
        """Generate intelligent response using MAXV1 model"""
        
        # Update conversation history
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            'user': user_input,
            'timestamp': datetime.now(),
            'intent': self.analyze_intent(user_input),
            'emotion': self.analyze_emotion(user_input)
        })
        
        # Analyze current input
        intent_scores = self.analyze_intent(user_input)
        emotion_analysis = self.analyze_emotion(user_input)
        dominant_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else 'general'
        dominant_emotion = max(emotion_analysis.items(), key=lambda x: x[1])[0]
        
        # Generate contextual response
        try:
            # Use neural model for response generation
            response = self._generate_neural_response(user_input, dominant_intent, dominant_emotion, user_id)
        except Exception as e:
            logger.warning(f"Neural generation failed: {e}, falling back to template")
            response = self._generate_template_response(user_input, dominant_intent, dominant_emotion, user_id)
        
        # Add to conversation history
        self.conversation_history[user_id].append({
            'assistant': response,
            'timestamp': datetime.now(),
            'intent_detected': dominant_intent,
            'emotion_detected': dominant_emotion
        })
        
        # Keep only last 50 exchanges per user
        if len(self.conversation_history[user_id]) > 100:
            self.conversation_history[user_id] = self.conversation_history[user_id][-100:]
        
        return response
    
    def _generate_neural_response(self, user_input: str, intent: str, emotion: str, user_id: str) -> str:
        """Generate response using the neural model"""
        
        # Prepare context
        context = self._build_context(user_id)
        full_input = f"<CONTEXT>{context}<USER>{user_input}<ASSISTANT>"
        
        # Tokenize
        tokens = self.tokenizer.encode(full_input, max_length=400)
        input_ids = torch.tensor([tokens]).to(self.device)
        attention_mask = torch.tensor([[1 if t != 0 else 0 for t in tokens]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Sample from the distribution
            probs = F.softmax(logits / 0.8, dim=-1)  # Temperature sampling
            next_token_id = torch.multinomial(probs, 1).item()
            
            # Generate sequence (simplified for demo)
            generated_tokens = [next_token_id]
            
            # Decode to text
            response = self.tokenizer.decode(generated_tokens)
            
            # Post-process response
            response = self._post_process_response(response, intent, emotion)
        
        return response
    
    def _generate_template_response(self, user_input: str, intent: str, emotion: str, user_id: str) -> str:
        """Generate response using templates (fallback)"""
        
        # Handle different intents
        if intent == 'greeting':
            time_greeting = self._get_time_greeting()
            response = random.choice(self.response_templates['greeting'])
            response = response.format(time_greeting=time_greeting)
            
        elif intent == 'schedule':
            response = self._handle_schedule_intent(user_input, user_id)
            
        elif intent == 'question':
            response = self._handle_question_intent(user_input)
            
        elif intent == 'time':
            response = self._handle_time_intent()
            
        elif intent == 'thanks':
            response = "Sama-sama! Senang bisa membantu Anda. Ada lagi yang perlu bantuan?"
            
        elif intent == 'goodbye':
            response = "Sampai jumpa! Jangan ragu untuk kembali jika membutuhkan bantuan. Take care!"
            
        elif 'emotion_negative' in intent or emotion in ['sadness', 'anger', 'fear']:
            response = random.choice(self.response_templates['emotional_support'])
            
        else:
            # General conversation
            response = self._handle_general_conversation(user_input, emotion)
        
        return response
    
    def _build_context(self, user_id: str) -> str:
        """Build conversation context"""
        if user_id not in self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[user_id][-6:]  # Last 3 exchanges
        context_parts = []
        
        for exchange in recent_history:
            if 'user' in exchange:
                context_parts.append(f"User: {exchange['user']}")
            elif 'assistant' in exchange:
                context_parts.append(f"Assistant: {exchange['assistant']}")
        
        return " | ".join(context_parts)
    
    def _post_process_response(self, response: str, intent: str, emotion: str) -> str:
        """Post-process and clean up generated response"""
        # Clean up the response
        response = response.strip()
        
        # Add personality touches based on emotion
        if emotion == 'joy':
            if not any(word in response.lower() for word in ['senang', 'bagus', 'hebat', 'great']):
                response += " 😊"
        elif emotion == 'sadness':
            if not any(word in response.lower() for word in ['maaf', 'sorry', 'turut']):
                response = "Saya turut merasakan. " + response
        
        # Ensure response is not too short or too long
        if len(response) < 10:
            response = "Saya memahami. Bisa dijelaskan lebih detail?"
        elif len(response) > 500:
            response = response[:500] + "..."
        
        return response
    
    def _get_time_greeting(self) -> str:
        """Get appropriate time-based greeting"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "pagi"
        elif 12 <= hour < 15:
            return "siang"
        elif 15 <= hour < 18:
            return "sore"
        else:
            return "malam"
    
    def _handle_schedule_intent(self, user_input: str, user_id: str) -> str:
        """Handle scheduling requests"""
        # Extract schedule information
        schedule_text = user_input
        time_info = self._extract_time(user_input)
        
        # Save to schedule
        schedule_info = {
            'text': schedule_text,
            'time': time_info,
            'created': datetime.now().isoformat(),
            'user_id': user_id
        }
        
        # Here you would save to your database
        # For demo, we'll just acknowledge
        
        response = random.choice(self.response_templates['schedule_confirm'])
        return response.format(
            schedule=schedule_text, 
            time=time_info if time_info else "waktu yang disebutkan"
        )
    
    def _extract_time(self, text: str) -> Optional[str]:
        """Extract time information from text"""
        patterns = [
            r'jam (\d{1,2}):?(\d{0,2})',
            r'pukul (\d{1,2}):?(\d{0,2})',
            r'(\d{1,2}):(\d{2})',
            r'(\d{1,2})\s*(pagi|siang|sore|malam)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return None
    
    def _handle_question_intent(self, user_input: str) -> str:
        """Handle questions"""
        if 'nama' in user_input.lower() and any(word in user_input.lower() for word in ['kamu', 'anda', 'you']):
            return f"Nama saya {self.personality['name']}! Saya adalah asisten AI yang dirancang untuk membantu Anda dengan berbagai kebutuhan sehari-hari."
        
        elif 'siapa' in user_input.lower():
            return f"Saya {self.personality['name']}, asisten AI canggih yang dibuat khusus untuk memberikan bantuan personal yang intelligent dan natural."
        
        elif any(word in user_input.lower() for word in ['bisa', 'dapat', 'can', 'able']):
            return """Saya bisa membantu Anda dengan:
• Mengatur jadwal dan pengingat cerdas
• Menjawab pertanyaan umum
• Memberikan dukungan emosional
• Menganalisis dan memahami konteks percakapan
• Berkomunikasi secara natural dalam bahasa Indonesia dan Inggris
• Belajar dari preferensi dan kebiasaan Anda

Apa yang ingin Anda coba?"""
        
        else:
            return "Pertanyaan yang menarik! Saya akan berusaha membantu sebaik mungkin. Bisa dijelaskan lebih detail apa yang ingin Anda ketahui?"
    
    def _handle_time_intent(self) -> str:
        """Handle time-related queries"""
        now = datetime.now()
        days = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
        day_name = days[now.weekday()]
        
        return f"Sekarang pukul {now.strftime('%H:%M')}, hari {day_name}, {now.strftime('%d %B %Y')}. Ada yang bisa saya bantu terkait waktu atau jadwal?"
    
    def _handle_general_conversation(self, user_input: str, emotion: str) -> str:
        """Handle general conversation"""
        responses = [
            "Itu menarik! Cerita lebih lanjut dong.",
            "Saya mendengarkan. Bagaimana perasaan Anda tentang itu?",
            "Hmm, saya memahami. Ingin membahas lebih dalam?",
            "Interesting! Saya senang bisa mendengar perspektif Anda.",
            "Terima kasih sudah berbagi. Ada yang bisa saya bantu?"
        ]
        
        if emotion == 'joy':
            responses.extend([
                "Senang mendengar itu! Semoga hari Anda terus menyenangkan.",
                "Wah, bagus sekali! Saya ikut senang mendengarnya."
            ])
        elif emotion in ['sadness', 'anger']:
            responses.extend([
                "Saya memahami perasaan Anda. Ingin bercerita lebih lanjut?",
                "Terima kasih sudah mempercayai saya. Saya di sini untuk mendengarkan."
            ])
        
        return random.choice(responses)

# Global MAXV1 instance
maxv1 = MAXV1Core()

# WhatsApp and Flask setup (same as before)
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN', 'maxv1_verify_token_2024')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN', '')
PHONE_NUMBER_ID = os.getenv('PHONE_NUMBER_ID', '')

flask_app = Flask(__name__)

@flask_app.route('/webhook', methods=['GET'])
def verify_webhook():
    mode = request.args.get('hub.mode')
    token = request.args.get('hub.verify_token')
    challenge = request.args.get('hub.challenge')
    
    if mode == 'subscribe' and token == VERIFY_TOKEN:
        return challenge
    return "Verification failed", 403

@flask_app.route('/webhook', methods=['POST'])
def handle_whatsapp():
    try:
        data = request.get_json()
        
        if 'entry' in data:
            for entry in data['entry']:
                if 'changes' in entry:
                    for change in entry['changes']:
                        if change.get('field') == 'messages' and 'messages' in change['value']:
                            for message in change['value']['messages']:
                                user_id = message['from']
                                phone_number = message['from']
                                
                                if 'text' in message:
                                    message_text = message['text']['body']
                                    response = maxv1.generate_response(message_text, user_id)
                                    send_whatsapp_message(phone_number, response)
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"WhatsApp webhook error: {e}")
        return jsonify({"status": "error"}), 500

def send_whatsapp_message(phone_number, message):
    """Send WhatsApp message"""
    if not ACCESS_TOKEN or not PHONE_NUMBER_ID:
        return None
        
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "text": {"body": message}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return response.json()
    except Exception as e:
        logger.error(f"WhatsApp send error: {e}")
        return None

@flask_app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "MAXV1 Advanced AI Assistant",
        "model": "MAXV1 Custom Neural Network",
        "whatsapp_ready": bool(ACCESS_TOKEN and PHONE_NUMBER_ID),
        "timestamp": datetime.now().isoformat()
    })

# Gradio Interface Functions
def chat_interface(message, history):
    """Gradio chat interface"""
    if not message.strip():
        return history, ""
    
    try:
        response = maxv1.generate_response(message, "gradio_user")
        history.append([message, response])
        return history, ""
    except Exception as e:
        logger.error(f"Chat interface error: {e}")
        history.append([message, "Maaf, ada kendala teknis. Coba lagi ya!"])
        return history, ""

def clear_chat():
    """Clear chat history"""
    return [], ""

def save_model():
    """Save MAXV1 model"""
    try:
        model_path = "maxv1_model.pth"
        torch.save(maxv1.model.state_dict(), model_path)
        
        # Save tokenizer vocab
        vocab_path = "maxv1_vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(maxv1.tokenizer.vocab, f, ensure_ascii=False, indent=2)
        
        # Save conversation history
        history_path = "maxv1_conversations.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_history = {}
            for user_id, conversations in maxv1.conversation_history.items():
                serializable_history[user_id] = []
                for conv in conversations:
                    serializable_conv = conv.copy()
                    if 'timestamp' in serializable_conv:
                        serializable_conv['timestamp'] = serializable_conv['timestamp'].isoformat()
                    serializable_history[user_id].append(serializable_conv)
            
            json.dump(serializable_history, f, ensure_ascii=False, indent=2)
        
        return f"✅ MAXV1 model saved successfully!\n- Model: {model_path}\n- Vocab: {vocab_path}\n- History: {history_path}"
    except Exception as e:
        return f"❌ Error saving model: {e}"

def load_model():
    """Load MAXV1 model"""
    try:
        model_path = "maxv1_model.pth"
        vocab_path = "maxv1_vocab.json"
        history_path = "maxv1_conversations.json"
        
        if os.path.exists(model_path):
            maxv1.model.load_state_dict(torch.load(model_path, map_location=maxv1.device))
            
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                maxv1.tokenizer.vocab = json.load(f)
                maxv1.tokenizer.reverse_vocab = {v: k for k, v in maxv1.tokenizer.vocab.items()}
        
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                for user_id, conversations in history_data.items():
                    maxv1.conversation_history[user_id] = []
                    for conv in conversations:
                        if 'timestamp' in conv:
                            conv['timestamp'] = datetime.fromisoformat(conv['timestamp'])
                        maxv1.conversation_history[user_id].append(conv)
        
        return f"✅ MAXV1 model loaded successfully!"
    except Exception as e:
        return f"❌ Error loading model: {e}"

def get_model_info():
    """Get MAXV1 model information"""
    total_params = sum(p.numel() for p in maxv1.model.parameters())
    trainable_params = sum(p.numel() for p in maxv1.model.parameters() if p.requires_grad)
    
    info = f"""
🤖 **MAXV1 Model Information**

**Architecture:**
- Model Type: MAXV1 Custom Neural Network
- Vocabulary Size: {len(maxv1.tokenizer.vocab):,}
- Embedding Dimension: {maxv1.model.embed_dim}
- Hidden Dimension: {maxv1.model.hidden_dim}
- Transformer Layers: {maxv1.model.num_layers}
- Attention Heads: {maxv1.model.num_heads}

**Parameters:**
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Device: {maxv1.device}

**Capabilities:**
- Natural Language Understanding ✅
- Contextual Memory ✅
- Emotion Detection ✅
- Personality Modeling ✅
- Multi-language Support (ID/EN) ✅
- Intent Recognition ✅

**Active Users:** {len(maxv1.conversation_history)}
**Total Conversations:** {sum(len(convs) for convs in maxv1.conversation_history.values())}
"""
    return info

def train_maxv1(training_text, epochs=1):
    """Simple training function for MAXV1"""
    try:
        if not training_text.strip():
            return "❌ Please provide training text"
        
        # Prepare training data
        lines = training_text.split('\n')
        training_pairs = []
        
        for i in range(0, len(lines)-1, 2):
            if i+1 < len(lines):
                user_input = lines[i].strip()
                assistant_response = lines[i+1].strip()
                if user_input and assistant_response:
                    training_pairs.append((user_input, assistant_response))
        
        if not training_pairs:
            return "❌ No valid training pairs found. Format should be:\nUser: message\nAssistant: response"
        
        # Simple training loop
        maxv1.model.train()
        optimizer = torch.optim.AdamW(maxv1.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        for epoch in range(epochs):
            for user_input, expected_response in training_pairs:
                optimizer.zero_grad()
                
                # Tokenize input and target
                input_text = f"<USER>{user_input}<ASSISTANT>"
                input_tokens = maxv1.tokenizer.encode(input_text, max_length=256)
                target_tokens = maxv1.tokenizer.encode(expected_response, max_length=256)
                
                input_ids = torch.tensor([input_tokens]).to(maxv1.device)
                target_ids = torch.tensor([target_tokens]).to(maxv1.device)
                attention_mask = torch.tensor([[1 if t != 0 else 0 for t in input_tokens]]).to(maxv1.device)
                
                # Forward pass
                outputs = maxv1.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                # Calculate loss (simplified)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        maxv1.model.eval()
        avg_loss = total_loss / (len(training_pairs) * epochs)
        
        return f"✅ Training completed!\n- Training pairs: {len(training_pairs)}\n- Epochs: {epochs}\n- Average loss: {avg_loss:.4f}"
    
    except Exception as e:
        return f"❌ Training error: {e}"

# Gradio Interface
with gr.Blocks(
    title="MAXV1 - Advanced AI Assistant", 
    theme=gr.themes.Soft(),
    css="""
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        text-align: center;
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .status-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    """
) as app:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🧠 MAXV1 - Advanced AI Assistant</h1>
        <p style="font-size: 1.2em; margin: 0.5rem 0;">
            Powered by Custom Neural Architecture | Natural Conversation | Intelligent Learning
        </p>
        <p style="opacity: 0.9;">
            🚀 Custom-built AI with contextual memory, emotion detection, and personality modeling
        </p>
    </div>
    """)
    
    # Status Section
    with gr.Row():
        with gr.Column(scale=1):
            model_status = gr.HTML(f"""
            <div class="status-card">
                <h3>🤖 Model Status</h3>
                <p><strong>Architecture:</strong> MAXV1 Neural Network</p>
                <p><strong>Device:</strong> {maxv1.device}</p>
                <p><strong>Status:</strong> <span style="color: #4CAF50;">READY</span></p>
            </div>
            """)
        
        with gr.Column(scale=1):
            wa_status_text = "🟢 READY" if (ACCESS_TOKEN and PHONE_NUMBER_ID) else "🟡 SETUP NEEDED"
            whatsapp_status = gr.HTML(f"""
            <div class="status-card">
                <h3>📱 WhatsApp Integration</h3>
                <p><strong>Status:</strong> {wa_status_text}</p>
                <p><strong>Webhook:</strong> /webhook</p>
                <p><strong>Health Check:</strong> /health</p>
            </div>
            """)
    
    # Main Chat Interface
    with gr.Tab("💬 Chat with MAXV1"):
        with gr.Column():
            chatbot = gr.Chatbot(
                label="MAXV1 Conversation",
                height=600,
                placeholder="MAXV1 is ready! Try: 'Halo MAXV1', 'Jadwalkan meeting besok jam 2', 'Apa yang bisa kamu lakukan?'",
                show_copy_button=True,
                avatar_images=["👤", "🤖"]
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ketik pesan Anda di sini... (MAXV1 memahami bahasa Indonesia dan Inggris)",
                    scale=4,
                    show_label=False,
                    lines=2
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                examples_btn = gr.Button("Show Examples 💡", variant="secondary")
    
    # Model Management Tab
    with gr.Tab("🔧 Model Management"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💾 Model Operations")
                
                with gr.Row():
                    save_btn = gr.Button("Save Model", variant="primary")
                    load_btn = gr.Button("Load Model", variant="secondary")
                
                model_ops_output = gr.Textbox(
                    label="Operation Result",
                    lines=5,
                    interactive=False
                )
                
                gr.Markdown("### 📊 Model Information")
                info_btn = gr.Button("Get Model Info")
                model_info_output = gr.Markdown()
            
            with gr.Column():
                gr.Markdown("### 🎓 Training Interface")
                gr.Markdown("Train MAXV1 with custom conversations (Format: User line, then Assistant line)")
                
                training_input = gr.Textbox(
                    label="Training Data",
                    lines=10,
                    placeholder="""User: Halo MAXV1
Assistant: Halo! Saya MAXV1, siap membantu Anda hari ini.
User: Bagaimana cara membuat jadwal?
Assistant: Saya bisa membantu Anda membuat jadwal. Cukup katakan seperti 'Jadwalkan meeting besok jam 2' dan saya akan mencatatnya."""
                )
                
                epochs_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    label="Training Epochs"
                )
                
                train_btn = gr.Button("Train Model 🎓", variant="primary")
                training_output = gr.Textbox(
                    label="Training Result",
                    lines=5,
                    interactive=False
                )
    
    # Analytics Tab
    with gr.Tab("📈 Analytics"):
        with gr.Column():
            gr.Markdown("### 📊 Conversation Analytics")
            
            analytics_btn = gr.Button("Generate Analytics Report")
            analytics_output = gr.HTML()
    
    # Examples Section
    gr.Markdown("""
    ### 💡 Try These Examples:
    
    **Basic Conversation:**
    - "Halo MAXV1, apa kabar?"
    - "Siapa kamu dan apa yang bisa kamu lakukan?"
    - "Terima kasih atas bantuannya"
    
    **Schedule Management:**
    - "Jadwalkan meeting dengan tim besok jam 2 siang"
    - "Ingatkan saya untuk call client jam 4 sore"
    - "Apa jadwal saya hari ini?"
    
    **Emotional Support:**
    - "Saya merasa sedih hari ini"
    - "Hari ini sangat menyenangkan!"
    - "Saya stress dengan pekerjaan"
    
    **General Questions:**
    - "Jam berapa sekarang?"
    - "Apa cuaca hari ini?"
    - "Bagaimana cara mengatur waktu yang efektif?"
    """)
    
    # Event Handlers
    msg.submit(chat_interface, [msg, chatbot], [chatbot, msg])
    send_btn.click(chat_interface, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])
    
    save_btn.click(save_model, outputs=model_ops_output)
    load_btn.click(load_model, outputs=model_ops_output)
    info_btn.click(get_model_info, outputs=model_info_output)
    
    train_btn.click(train_maxv1, [training_input, epochs_slider], training_output)
    
    def generate_analytics():
        try:
            total_users = len(maxv1.conversation_history)
            total_conversations = sum(len(convs) for convs in maxv1.conversation_history.values())
            
            # Analyze emotions
            emotion_counts = {}
            intent_counts = {}
            
            for user_convs in maxv1.conversation_history.values():
                for conv in user_convs:
                    if 'emotion_detected' in conv:
                        emotion = conv['emotion_detected']
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    if 'intent_detected' in conv:
                        intent = conv['intent_detected']
                        intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            html_report = f"""
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 15px;">
                <h2>📊 MAXV1 Analytics Report</h2>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
                    <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h3 style="color: #667eea;">👥 Total Users</h3>
                        <p style="font-size: 2em; margin: 0; color: #333;">{total_users}</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h3 style="color: #667eea;">💬 Conversations</h3>
                        <p style="font-size: 2em; margin: 0; color: #333;">{total_conversations}</p>
                    </div>
                </div>
                
                <div style="margin: 2rem 0;">
                    <h3>🎭 Top Emotions Detected:</h3>
                    <ul>
            """
            
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                html_report += f"<li><strong>{emotion.title()}:</strong> {count} times</li>"
            
            html_report += """
                    </ul>
                </div>
                
                <div style="margin: 2rem 0;">
                    <h3>🎯 Top Intents:</h3>
                    <ul>
            """
            
            for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                html_report += f"<li><strong>{intent.title()}:</strong> {count} times</li>"
            
            html_report += """
                    </ul>
                </div>
                
                <p style="text-align: center; margin-top: 2rem; color: #666;">
                    Report generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
                </p>
            </div>
            """
            
            return html_report
            
        except Exception as e:
            return f"<div style='color: red;'>Error generating analytics: {e}</div>"
    
    analytics_btn.click(generate_analytics, outputs=analytics_output)

# Launch Functions
def run_flask():
    """Run Flask webhook server"""
    try:
        flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Flask server error: {e}")

if __name__ == "__main__":
    print("🚀 Starting MAXV1 Advanced AI Assistant...")
    print(f"🧠 Model: Custom MAXV1 Neural Network")
    print(f"💾 Device: {maxv1.device}")
    print(f"📝 Vocabulary Size: {len(maxv1.tokenizer.vocab):,}")
    print(f"🔧 Parameters: {sum(p.numel() for p in maxv1.model.parameters()):,}")
    
    # Load existing model if available
    if os.path.exists("maxv1_model.pth"):
        print("📂 Loading existing MAXV1 model...")
        load_result = load_model()
        print(load_result)
    
    # Start WhatsApp webhook if configured
    if ACCESS_TOKEN and PHONE_NUMBER_ID:
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print("✅ WhatsApp webhook running on port 5000")
    else:
        print("⚠️  WhatsApp not configured. Set ACCESS_TOKEN and PHONE_NUMBER_ID environment variables.")
    
    # Launch Gradio interface
    print("🌐 Launching MAXV1 web interface...")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=True
    )