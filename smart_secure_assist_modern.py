# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                SMART SECURE ASSIST v5.0 MODERN EDITION                       ║
║           Your AI Cyber Coach - Stay Secure, Stay Smart                      ║
║                                                                              ║
║  🏆 Competition Ready | 🔒 Security Hardened | 🎨 Modern Light UI           ║
║  ✨ PDF Reports | 📊 Analytics | 🎮 Gamification | ⛓️ Blockchain            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Installation:
    pip install streamlit pandas plotly scikit-learn tldextract numpy joblib reportlab

Run:
    streamlit run smart_secure_assist_modern.py

Features:
    - Modern light pastel theme with Inter font
    - NO password storage (privacy-first)
    - SQLite with connection pooling
    - ML phishing detection with explainability
    - Visual blockchain ledger
    - PDF report generation
    - Gamification with XP and badges
    - Admin analytics dashboard
    - GDPR-compliant data deletion

Author: Enhanced for Modern UI | License: MIT
"""

# IMPORTS
import streamlit as st
import re
import hashlib
import json
import math
import os
import random
import warnings
import sqlite3
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse
from ipaddress import ip_address
from typing import Dict, List, Tuple, Optional
from contextlib import closing
import io

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tldextract
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

warnings.filterwarnings('ignore')

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_secure_assist.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SmartSecureAssist')

# ============================================================================
# MODERN UI THEME & STYLING
# ============================================================================

def apply_modern_theme():
    """Apply modern light pastel theme with Inter font and elegant styling"""
    st.markdown("""
    <style>
    /* Import Inter Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --primary-color: #8b5cf6;
        --secondary-color: #3b82f6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-gradient: linear-gradient(135deg, #e0c3fc 0%, #b8c6f5 50%, #dfe9f3 100%);
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #e0c3fc 0%, #b8c6f5 50%, #dfe9f3 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8eeff 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.1);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    /* Card Styling */
    .stCard, div[data-testid="stMetricValue"], div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.1);
        transition: all 0.3s ease;
    }
    
    .stCard:hover {
        box-shadow: 0 8px 12px rgba(139, 92, 246, 0.15);
        transform: translateY(-2px);
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #8b5cf6;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(139, 92, 246, 0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #5855eb 100%);
        box-shadow: 0 6px 12px rgba(139, 92, 246, 0.3);
        transform: translateY(-2px);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 8px;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(139, 92, 246, 0.1);
        color: #8b5cf6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        font-weight: 600;
        color: #1e293b;
        border: 1px solid rgba(139, 92, 246, 0.1);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(139, 92, 246, 0.05);
        border-color: #8b5cf6;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%);
        border-radius: 10px;
    }
    
    /* Success/Warning/Error Boxes */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        border-radius: 12px;
        color: #065f46;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        color: #92400e;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        color: #991b1b;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        color: #1e40af;
    }
    
    /* Dataframe */
    .dataframe {
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        color: white;
        font-weight: 600;
        padding: 12px;
        border: none;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background: rgba(139, 92, 246, 0.05);
    }
    
    .dataframe tbody tr:hover {
        background: rgba(139, 92, 246, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    
    h1 {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Risk Level Badges */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .risk-very-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
        color: #7c2d12;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
    }
    
    /* Badge Display */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 16px 0;
    }
    
    .badge-item {
        background: rgba(255, 255, 255, 0.95);
        padding: 12px 20px;
        border-radius: 12px;
        border: 2px solid rgba(139, 92, 246, 0.2);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .badge-item:hover {
        border-color: #8b5cf6;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(139, 92, 246, 0.2);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(139, 92, 246, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #5855eb 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATABASE MANAGER (Connection Pool)
# ============================================================================

class DatabaseManager:
    """Enhanced SQLite manager with connection pooling"""

    def __init__(self, db_path: str = 'smart_secure_assist.db'):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get database connection with context manager support"""
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def init_database(self):
        """Initialize enhanced database schema"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()

                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        email TEXT,
                        created_at TEXT,
                        total_score REAL DEFAULT 0,
                        best_score REAL DEFAULT 0,
                        streak_days INTEGER DEFAULT 0,
                        last_login TEXT,
                        xp_points INTEGER DEFAULT 0,
                        level INTEGER DEFAULT 1
                    )
                ''')

                # Check history (NO RAW PASSWORDS)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS check_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        timestamp TEXT,
                        check_type TEXT,
                        score REAL,
                        entropy REAL,
                        risk_level TEXT,
                        FOREIGN KEY (username) REFERENCES users(username)
                    )
                ''')

                # Leaderboard
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS leaderboard (
                        username TEXT PRIMARY KEY,
                        score REAL,
                        xp_points INTEGER,
                        badges_count INTEGER,
                        timestamp TEXT,
                        FOREIGN KEY (username) REFERENCES users(username)
                    )
                ''')

                # Challenge history
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS challenge_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        challenge_type TEXT,
                        correct BOOLEAN,
                        timestamp TEXT,
                        FOREIGN KEY (username) REFERENCES users(username)
                    )
                ''')

                # Blockchain persistence
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS blockchain (
                        block_index INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        events TEXT,
                        previous_hash TEXT,
                        hash TEXT
                    )
                ''')

                conn.commit()
                logger.info("Enhanced database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def create_user(self, username: str, email: str = ""):
        """Create new user"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO users (username, email, created_at, last_login)
                    VALUES (?, ?, ?, ?)
                ''', (username, email, datetime.now().isoformat(), datetime.now().date().isoformat()))
                conn.commit()
                logger.info(f"User created: {username}")
        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")

    def update_user_score(self, username: str, score: float, check_type: str,
                         entropy: float = 0, risk_level: str = 'unknown'):
        """Update user score with XP calculation"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()

                # Calculate XP (score-based)
                xp_gain = int(score)

                # Update user
                cursor.execute('''
                    UPDATE users
                    SET total_score = ?,
                        best_score = MAX(best_score, ?),
                        xp_points = xp_points + ?,
                        level = 1 + (xp_points + ?) / 100,
                        last_login = ?
                    WHERE username = ?
                ''', (score, score, xp_gain, xp_gain, datetime.now().date().isoformat(), username))

                # Add to history
                cursor.execute('''
                    INSERT INTO check_history (username, timestamp, check_type, score, entropy, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (username, datetime.now().isoformat(), check_type, score, entropy, risk_level))

                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update score for {username}: {e}")

    def update_streak(self, username: str):
        """Update login streak"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()

                cursor.execute('SELECT last_login FROM users WHERE username = ?', (username,))
                result = cursor.fetchone()

                if result:
                    last_login = datetime.fromisoformat(result[0]).date() if result[0] else None
                    today = datetime.now().date()

                    if last_login:
                        delta = (today - last_login).days
                        if delta == 1:
                            cursor.execute('UPDATE users SET streak_days = streak_days + 1 WHERE username = ?', (username,))
                        elif delta > 1:
                            cursor.execute('UPDATE users SET streak_days = 1 WHERE username = ?', (username,))
                    else:
                        cursor.execute('UPDATE users SET streak_days = 1 WHERE username = ?', (username,))

                    cursor.execute('UPDATE users SET last_login = ? WHERE username = ?',
                                 (today.isoformat(), username))
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to update streak for {username}: {e}")

    def get_user(self, username: str) -> Optional[Dict]:
        """Get user data"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
                result = cursor.fetchone()

                if result:
                    return {
                        'username': result[0],
                        'email': result[1],
                        'created_at': result[2],
                        'total_score': result[3],
                        'best_score': result[4],
                        'streak_days': result[5],
                        'last_login': result[6],
                        'xp_points': result[7],
                        'level': result[8]
                    }
        except Exception as e:
            logger.error(f"Failed to get user {username}: {e}")
        return None

    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get top users"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT u.username, u.total_score, u.xp_points, u.level,
                           COUNT(DISTINCT ch.id) as badges
                    FROM users u
                    LEFT JOIN check_history ch ON u.username = ch.username
                    GROUP BY u.username
                    ORDER BY u.total_score DESC, u.xp_points DESC
                    LIMIT ?
                ''', (limit,))

                results = cursor.fetchall()
                return [
                    {
                        'username': r[0],
                        'score': r[1],
                        'xp': r[2],
                        'level': r[3],
                        'badges': min(r[4], 9)
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
        return []

    def get_user_history(self, username: str, limit: int = 10) -> List[Dict]:
        """Get user check history"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT timestamp, check_type, score, entropy, risk_level
                    FROM check_history
                    WHERE username = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (username, limit))

                results = cursor.fetchall()
                return [
                    {
                        'timestamp': r[0],
                        'type': r[1],
                        'score': r[2],
                        'entropy': r[3],
                        'risk_level': r[4]
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Failed to get history for {username}: {e}")
        return []

    def record_challenge(self, username: str, challenge_type: str, correct: bool):
        """Record challenge attempt"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO challenge_history (username, challenge_type, correct, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (username, challenge_type, correct, datetime.now().isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record challenge: {e}")

    def get_global_stats(self) -> Dict:
        """Get aggregated statistics"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()

                # Total users
                cursor.execute('SELECT COUNT(*) FROM users')
                total_users = cursor.fetchone()[0]

                # Average score
                cursor.execute('SELECT AVG(total_score) FROM users')
                avg_score = cursor.fetchone()[0] or 0

                # Total checks
                cursor.execute('SELECT COUNT(*) FROM check_history')
                total_checks = cursor.fetchone()[0]

                # Threats detected (score < 50)
                cursor.execute('SELECT COUNT(*) FROM check_history WHERE score < 50')
                threats_detected = cursor.fetchone()[0]

                # Total XP
                cursor.execute('SELECT SUM(xp_points) FROM users')
                total_xp = cursor.fetchone()[0] or 0

                return {
                    'total_users': total_users,
                    'avg_score': avg_score,
                    'total_checks': total_checks,
                    'threats_detected': threats_detected,
                    'total_xp': total_xp
                }
        except Exception as e:
            logger.error(f"Failed to get global stats: {e}")
        return {}

    def delete_user_data(self, username: str):
        """GDPR-compliant data deletion"""
        try:
            with closing(self.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM check_history WHERE username = ?', (username,))
                cursor.execute('DELETE FROM challenge_history WHERE username = ?', (username,))
                cursor.execute('DELETE FROM leaderboard WHERE username = ?', (username,))
                cursor.execute('DELETE FROM users WHERE username = ?', (username,))
                conn.commit()
                logger.info(f"User data deleted: {username}")
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")

    def backup_database(self) -> Optional[bytes]:
        """Create database backup"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
        return None

# ============================================================================
# ML MODEL (Phishing Detection)
# ============================================================================

@st.cache_resource
def load_ml_model():
    """Load or train ML model for phishing detection"""
    try:
        # Sample phishing dataset
        legitimate_urls = [
            'google.com', 'facebook.com', 'twitter.com', 'amazon.com', 'microsoft.com',
            'apple.com', 'linkedin.com', 'wikipedia.org', 'github.com', 'stackoverflow.com',
            'reddit.com', 'youtube.com', 'instagram.com', 'netflix.com', 'paypal.com'
        ]

        phishing_urls = [
            'secure-paypal-verify.com', 'login-microsoft-secure.com', 'amazon-account-update.net',
            'facebook-security-check.info', 'apple-id-verify.org', 'netflix-payment-failed.com',
            'bank-of-america-secure.info', 'chase-verify-account.net', 'google-security-alert.com',
            'instagram-verify-badge.org', 'twitter-blue-verify.info', 'linkedin-premium-offer.com'
        ]

        # Create dataset
        urls = legitimate_urls + phishing_urls
        labels = [0] * len(legitimate_urls) + [1] * len(phishing_urls)

        # Vectorize
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1000)
        X = vectorizer.fit_transform(urls)

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, labels)

        # Calculate metrics
        y_pred = model.predict(X)
        metrics = {
            'accuracy': accuracy_score(labels, y_pred),
            'precision': precision_score(labels, y_pred, zero_division=0),
            'recall': recall_score(labels, y_pred, zero_division=0),
            'f1_score': f1_score(labels, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, y_pred).tolist()
        }

        logger.info(f"ML Model trained - Accuracy: {metrics['accuracy']:.2%}")

        return {
            'model': model,
            'vectorizer': vectorizer,
            'feature_names': vectorizer.get_feature_names_out().tolist(),
            'metrics': metrics
        }

    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")
        return None

# ============================================================================
# SECURITY ANALYZER
# ============================================================================

class SecurityAnalyzer:
    """Advanced password and URL security analyzer"""

    def __init__(self, ml_components: Dict):
        self.model = ml_components['model'] if ml_components else None
        self.vectorizer = ml_components['vectorizer'] if ml_components else None
        self.feature_names = ml_components['feature_names'] if ml_components else []
        self.metrics = ml_components['metrics'] if ml_components else {}

        # Common phishing keywords
        self.phishing_keywords = [
            'verify', 'secure', 'account', 'update', 'confirm', 'login',
            'banking', 'alert', 'suspended', 'urgent', 'click', 'credential'
        ]

        # Suspicious TLDs
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.info', '.online']

    def calculate_entropy(self, password: str) -> float:
        """Calculate Shannon entropy"""
        if not password:
            return 0.0

        prob = [float(password.count(c)) / len(password) for c in set(password)]
        entropy = -sum(p * math.log2(p) for p in prob)
        return entropy

    def estimate_crack_time(self, password: str) -> str:
        """Estimate password crack time"""
        charset_size = 0
        if re.search(r'[a-z]', password):
            charset_size += 26
        if re.search(r'[A-Z]', password):
            charset_size += 26
        if re.search(r'\d', password):
            charset_size += 10
        if re.search(r'[^a-zA-Z0-9]', password):
            charset_size += 32

        if charset_size == 0:
            return "Instantly"

        combinations = charset_size ** len(password)
        # Assume 10 billion attempts per second
        seconds = combinations / (10 * 10**9)

        if seconds < 1:
            return "Instantly"
        elif seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            return f"{int(seconds/60)} minutes"
        elif seconds < 86400:
            return f"{int(seconds/3600)} hours"
        elif seconds < 31536000:
            return f"{int(seconds/86400)} days"
        else:
            years = int(seconds/31536000)
            return f"{years:,} years"

    def analyze_password(self, password: str) -> Dict:
        """Comprehensive password analysis"""
        if not password:
            return {
                'score': 0,
                'strength': 'Empty',
                'entropy': 0,
                'crack_time': 'Instantly',
                'feedback': ['Password cannot be empty'],
                'risk_level': 'critical'
            }

        score = 0
        feedback = []

        # Length check
        length = len(password)
        if length >= 12:
            score += 30
        elif length >= 8:
            score += 20
            feedback.append("Consider using 12+ characters")
        else:
            score += 10
            feedback.append("Password is too short (minimum 8 characters)")

        # Character variety
        has_lower = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_special = bool(re.search(r'[^a-zA-Z0-9]', password))

        variety_score = sum([has_lower, has_upper, has_digit, has_special])
        score += variety_score * 15

        if not has_upper:
            feedback.append("Add uppercase letters")
        if not has_digit:
            feedback.append("Add numbers")
        if not has_special:
            feedback.append("Add special characters (!@#$%)")

        # Common patterns
        if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
            score -= 10
            feedback.append("Avoid sequential numbers")

        if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):
            score -= 10
            feedback.append("Avoid sequential letters")

        common_passwords = ['password', '12345678', 'qwerty', 'abc123', 'admin']
        if password.lower() in common_passwords:
            score = 0
            feedback = ["This is a commonly used password - very weak!"]

        # Calculate entropy
        entropy = self.calculate_entropy(password)

        # Estimate crack time
        crack_time = self.estimate_crack_time(password)

        # Determine strength
        score = max(0, min(100, score))

        if score >= 80:
            strength = 'Excellent'
            risk_level = 'very_low'
        elif score >= 60:
            strength = 'Good'
            risk_level = 'low'
        elif score >= 40:
            strength = 'Fair'
            risk_level = 'medium'
        elif score >= 20:
            strength = 'Weak'
            risk_level = 'high'
        else:
            strength = 'Very Weak'
            risk_level = 'critical'

        if not feedback:
            feedback = ["Excellent password strength!"]

        return {
            'score': score,
            'strength': strength,
            'entropy': round(entropy, 2),
            'crack_time': crack_time,
            'feedback': feedback,
            'risk_level': risk_level,
            'has_lower': has_lower,
            'has_upper': has_upper,
            'has_digit': has_digit,
            'has_special': has_special
        }

    def analyze_url(self, url: str) -> Dict:
        """Comprehensive URL analysis with ML and rule-based checks"""
        if not url:
            return {
                'score': 0,
                'risk_level': 'critical',
                'ml_confidence': 0,
                'suspicious_patterns': ['URL cannot be empty'],
                'domain': '',
                'explanation': 'No URL provided'
            }

        try:
            # Clean URL
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split('/')[0]

            # Extract features
            extracted = tldextract.extract(url)

            score = 100
            suspicious_patterns = []
            ml_confidence = 0.5

            # ML prediction
            if self.model and self.vectorizer:
                try:
                    X = self.vectorizer.transform([domain])
                    prediction = self.model.predict(X)[0]
                    proba = self.model.predict_proba(X)[0]
                    ml_confidence = float(proba[1])  # Phishing probability

                    if prediction == 1:
                        score -= 40
                        suspicious_patterns.append(f"ML model detected phishing (confidence: {ml_confidence:.1%})")
                except Exception as e:
                    logger.error(f"ML prediction failed: {e}")

            # Rule-based checks
            if not url.startswith('https://'):
                score -= 20
                suspicious_patterns.append("No HTTPS encryption")

            # Check for IP address
            try:
                ip_address(domain.strip('[]'))
                score -= 30
                suspicious_patterns.append("Using IP address instead of domain")
            except:
                pass

            # Suspicious TLD
            if any(domain.endswith(tld) for tld in self.suspicious_tlds):
                score -= 15
                suspicious_patterns.append("Suspicious top-level domain")

            # Phishing keywords
            url_lower = url.lower()
            found_keywords = [kw for kw in self.phishing_keywords if kw in url_lower]
            if found_keywords:
                score -= len(found_keywords) * 5
                suspicious_patterns.append(f"Phishing keywords: {', '.join(found_keywords)}")

            # Excessive hyphens or dots
            if domain.count('-') > 2:
                score -= 10
                suspicious_patterns.append("Excessive hyphens in domain")

            if domain.count('.') > 3:
                score -= 10
                suspicious_patterns.append("Too many subdomains")

            # URL length
            if len(url) > 75:
                score -= 10
                suspicious_patterns.append("Unusually long URL")

            # @ symbol in URL
            if '@' in url:
                score -= 30
                suspicious_patterns.append("Contains @ symbol (phishing technique)")

            # Double slashes
            if '//' in parsed.path:
                score -= 15
                suspicious_patterns.append("Double slashes in path")

            score = max(0, min(100, score))

            # Determine risk level
            if score >= 80:
                risk_level = 'very_low'
                explanation = "This URL appears safe to visit."
            elif score >= 60:
                risk_level = 'low'
                explanation = "This URL is likely safe, but exercise caution."
            elif score >= 40:
                risk_level = 'medium'
                explanation = "This URL shows some suspicious characteristics."
            elif score >= 20:
                risk_level = 'high'
                explanation = "This URL has multiple red flags - avoid visiting."
            else:
                risk_level = 'critical'
                explanation = "This URL is highly suspicious - DO NOT visit!"

            if not suspicious_patterns:
                suspicious_patterns = ["No suspicious patterns detected"]

            return {
                'score': score,
                'risk_level': risk_level,
                'ml_confidence': ml_confidence,
                'suspicious_patterns': suspicious_patterns,
                'domain': domain,
                'explanation': explanation,
                'has_https': url.startswith('https://'),
                'subdomain': extracted.subdomain,
                'tld': extracted.suffix
            }

        except Exception as e:
            logger.error(f"URL analysis failed: {e}")
            return {
                'score': 0,
                'risk_level': 'critical',
                'ml_confidence': 0,
                'suspicious_patterns': [f'Error analyzing URL: {str(e)}'],
                'domain': '',
                'explanation': 'Analysis failed'
            }

# ============================================================================
# BLOCKCHAIN LEDGER
# ============================================================================

class BlockchainLedger:
    """Immutable audit trail for user actions"""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self.chain = []
        self.pending_events = []
        self.load_from_db()

    def calculate_hash(self, index: int, timestamp: str, events: List, previous_hash: str) -> str:
        """Calculate SHA-256 hash"""
        data = f"{index}{timestamp}{json.dumps(events)}{previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def create_genesis_block(self):
        """Create the first block"""
        genesis = {
            'index': 0,
            'timestamp': datetime.now().isoformat(),
            'events': [{'type': 'genesis', 'description': 'Genesis block created'}],
            'previous_hash': '0',
            'hash': ''
        }
        genesis['hash'] = self.calculate_hash(0, genesis['timestamp'], genesis['events'], '0')
        self.chain.append(genesis)
        self.save_block_to_db(genesis)

    def add_event(self, event_type: str, description: str, metadata: Dict = None):
        """Add event to pending queue"""
        event = {
            'type': event_type,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        if metadata:
            event['metadata'] = metadata
        self.pending_events.append(event)

    def commit_events(self) -> Optional[Dict]:
        """Commit pending events as new block"""
        if not self.pending_events:
            return None

        if not self.chain:
            self.create_genesis_block()

        previous_block = self.chain[-1]
        new_block = {
            'index': len(self.chain),
            'timestamp': datetime.now().isoformat(),
            'events': self.pending_events.copy(),
            'previous_hash': previous_block['hash'],
            'hash': ''
        }
        new_block['hash'] = self.calculate_hash(
            new_block['index'],
            new_block['timestamp'],
            new_block['events'],
            new_block['previous_hash']
        )

        self.chain.append(new_block)
        self.pending_events.clear()
        self.save_block_to_db(new_block)

        return new_block

    def verify_chain(self) -> Tuple[bool, Optional[int]]:
        """Verify blockchain integrity"""
        if not self.chain:
            return True, None

        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]

            # Verify hash
            calculated_hash = self.calculate_hash(
                current['index'],
                current['timestamp'],
                current['events'],
                current['previous_hash']
            )

            if current['hash'] != calculated_hash:
                return False, i

            # Verify link
            if current['previous_hash'] != previous['hash']:
                return False, i

        return True, None

    def save_block_to_db(self, block: Dict):
        """Persist block to database"""
        try:
            with closing(self.db.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO blockchain (block_index, timestamp, events, previous_hash, hash)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    block['index'],
                    block['timestamp'],
                    json.dumps(block['events']),
                    block['previous_hash'],
                    block['hash']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save block: {e}")

    def load_from_db(self):
        """Load blockchain from database"""
        try:
            with closing(self.db.get_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM blockchain ORDER BY block_index')
                results = cursor.fetchall()

                self.chain = []
                for row in results:
                    block = {
                        'index': row[0],
                        'timestamp': row[1],
                        'events': json.loads(row[2]),
                        'previous_hash': row[3],
                        'hash': row[4]
                    }
                    self.chain.append(block)

                if not self.chain:
                    self.create_genesis_block()

        except Exception as e:
            logger.error(f"Failed to load blockchain: {e}")
            self.create_genesis_block()

    def export_chain(self) -> str:
        """Export blockchain as JSON"""
        return json.dumps(self.chain, indent=2)

    def create_visual_graph(self):
        """Create visual blockchain graph using Plotly"""
        if not self.chain:
            return None

        # Prepare data for visualization
        x_pos = list(range(len(self.chain)))
        y_pos = [0] * len(self.chain)
        labels = [f"Block #{b['index']}" for b in self.chain]
        colors = ['#8b5cf6' if i == 0 else '#3b82f6' for i in range(len(self.chain))]

        # Create figure
        fig = go.Figure()

        # Add connections
        for i in range(1, len(self.chain)):
            fig.add_trace(go.Scatter(
                x=[x_pos[i-1], x_pos[i]],
                y=[0, 0],
                mode='lines',
                line=dict(color='rgba(139, 92, 246, 0.3)', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add blocks
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(
                size=50,
                color=colors,
                line=dict(color='white', width=2),
                symbol='square'
            ),
            text=labels,
            textposition='top center',
            textfont=dict(size=12, color='#1e293b', family='Inter'),
            hovertemplate='<b>%{text}</b><br>Hash: %{customdata}<extra></extra>',
            customdata=[b['hash'][:16] + '...' for b in self.chain],
            showlegend=False
        ))

        fig.update_layout(
            title='Blockchain Ledger Visualization',
            title_font=dict(size=20, color='#1e293b', family='Inter'),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-1, 1]
            ),
            plot_bgcolor='rgba(255, 255, 255, 0.95)',
            paper_bgcolor='rgba(255, 255, 255, 0.95)',
            hovermode='closest',
            height=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        return fig

# ============================================================================
# GAMIFICATION SYSTEM
# ============================================================================

class GamificationSystem:
    """XP, levels, and badges system"""

    def __init__(self):
        self.level_titles = {
            1: "🌱 Cyber Rookie",
            2: "🛡️ Security Apprentice",
            3: "🔍 Threat Hunter",
            4: "⚔️ Cyber Champion",
            5: "👑 Elite Guardian"
        }

        self.badges = {
            'first_check': {'name': '🎯 First Check', 'description': 'Completed first security check'},
            'password_guru': {'name': '🔐 Password Guardian', 'description': 'Analyzed a strong password'},
            'url_detective': {'name': '🔗 Link Detective', 'description': 'Checked first URL'},
            'safe_surfer': {'name': '✅ Safe Surfer', 'description': 'Found a safe URL'},
            'threat_hunter': {'name': '🚨 Threat Hunter', 'description': 'Detected a threat'},
            'high_scorer': {'name': '🏆 Elite Defender', 'description': 'Achieved score > 80'},
            'streak_3': {'name': '🔥 3-Day Streak', 'description': '3 days login streak'},
            'streak_7': {'name': '⚡ Week Warrior', 'description': '7 days login streak'},
            'simulation_master': {'name': '🎮 Simulation Master', 'description': 'Completed threat simulation'}
        }

    def calculate_level(self, xp: int) -> int:
        """Calculate level from XP"""
        return min(5, 1 + xp // 100)

    def get_level_title(self, level: int) -> str:
        """Get level title"""
        return self.level_titles.get(min(5, level), "👑 Elite Guardian")

    def xp_to_next_level(self, xp: int) -> int:
        """Calculate XP needed for next level"""
        current_level = self.calculate_level(xp)
        if current_level >= 5:
            return 0
        return (current_level * 100) - (xp % 100)

    def get_earned_badges(self, password_checked: bool, url_checked: bool,
                         url_safe: bool, score: float, streak: int,
                         simulation_done: bool) -> List[str]:
        """Get list of earned badges"""
        earned = []

        if password_checked or url_checked:
            earned.append('first_check')

        if password_checked:
            earned.append('password_guru')

        if url_checked:
            earned.append('url_detective')

        if url_safe:
            earned.append('safe_surfer')

        if score < 50:
            earned.append('threat_hunter')

        if score > 80:
            earned.append('high_scorer')

        if streak >= 3:
            earned.append('streak_3')

        if streak >= 7:
            earned.append('streak_7')

        if simulation_done:
            earned.append('simulation_master')

        return earned

    def award_xp(self, action: str) -> int:
        """Award XP for actions"""
        xp_rewards = {
            'password_check': 10,
            'url_check': 15,
            'simulation_correct': 20,
            'simulation_wrong': 5,
            'daily_login': 5
        }
        return xp_rewards.get(action, 0)

# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================

def generate_pdf_report(username: str, user_data: Dict, password_result: Dict,
                       url_result: Dict, badges: List, history: List) -> Optional[bytes]:
    """Generate security PDF report"""
    if not HAS_REPORTLAB:
        return None

    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = styles['Heading1']
        title_style.textColor = colors.HexColor('#8b5cf6')
        story.append(Paragraph("🛡️ Smart Secure Assist - Security Report", title_style))
        story.append(Spacer(1, 12))

        # User info
        story.append(Paragraph(f"<b>User:</b> {username}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>XP Points:</b> {user_data.get('xp_points', 0)}", styles['Normal']))
        story.append(Paragraph(f"<b>Level:</b> {user_data.get('level', 1)}", styles['Normal']))
        story.append(Paragraph(f"<b>Streak:</b> {user_data.get('streak_days', 0)} days", styles['Normal']))
        story.append(Spacer(1, 20))

        # Overall score
        story.append(Paragraph("<b>Overall Security Score</b>", styles['Heading2']))
        story.append(Paragraph(f"Score: {user_data.get('total_score', 0):.1f}/100", styles['Normal']))
        story.append(Spacer(1, 12))

        # Password analysis
        if password_result:
            story.append(Paragraph("<b>Password Strength Analysis</b>", styles['Heading2']))
            story.append(Paragraph(f"Strength: {password_result.get('strength', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"Score: {password_result.get('score', 0)}/100", styles['Normal']))
            story.append(Paragraph(f"Entropy: {password_result.get('entropy', 0):.2f} bits", styles['Normal']))
            story.append(Paragraph(f"Estimated Crack Time: {password_result.get('crack_time', 'Unknown')}", styles['Normal']))
            story.append(Spacer(1, 12))

        # URL analysis
        if url_result:
            story.append(Paragraph("<b>URL Safety Analysis</b>", styles['Heading2']))
            story.append(Paragraph(f"Safety Score: {url_result.get('score', 0)}/100", styles['Normal']))
            story.append(Paragraph(f"Risk Level: {url_result.get('risk_level', 'Unknown').upper()}", styles['Normal']))
            story.append(Paragraph(f"Domain: {url_result.get('domain', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 12))

        # Badges
        if badges:
            story.append(Paragraph("<b>Earned Badges</b>", styles['Heading2']))
            for badge_id in badges:
                badge = GamificationSystem().badges.get(badge_id)
                if badge:
                    story.append(Paragraph(f"• {badge['name']}: {badge['description']}", styles['Normal']))
            story.append(Spacer(1, 12))

        # Footer
        story.append(Spacer(1, 20))
        footer_style = styles['Normal']
        footer_style.textColor = colors.grey
        story.append(Paragraph("Generated by Smart Secure Assist v5.0", footer_style))
        story.append(Paragraph("Your AI Cyber Coach - Stay Secure, Stay Smart", footer_style))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_metric_card(label: str, value: str, delta: str = None, icon: str = "📊"):
    """Render a modern metric card"""
    if delta:
        html_content = f"""
        <div style="background: rgba(255, 255, 255, 0.95); border-radius: 16px; padding: 20px; box-shadow: 0 4px 6px rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.1); transition: all 0.3s ease;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="flex: 1;">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6; margin-top: 4px;">{value}</div>
                    <div style="color: #10b981; font-size: 0.9rem; margin-top: 4px;">{delta}</div>
                </div>
            </div>
        </div>
        """
    else:
        html_content = f"""
        <div style="background: rgba(255, 255, 255, 0.95); border-radius: 16px; padding: 20px; box-shadow: 0 4px 6px rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.1); transition: all 0.3s ease;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="flex: 1;">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #8b5cf6; margin-top: 4px;">{value}</div>
                </div>
            </div>
        </div>
        """
    
    st.markdown(html_content, unsafe_allow_html=True)

def render_risk_badge(risk_level: str) -> str:
    """Render risk level badge"""
    emoji_map = {
        'very_low': '✅',
        'low': '✅',
        'medium': '⚠️',
        'high': '⚠️',
        'critical': '🚨'
    }
    
    label_map = {
        'very_low': 'Very Low Risk',
        'low': 'Low Risk',
        'medium': 'Medium Risk',
        'high': 'High Risk',
        'critical': 'Critical Risk'
    }
    
    emoji = emoji_map.get(risk_level, '❓')
    label = label_map.get(risk_level, 'Unknown')
    
    return f"<span class='risk-badge risk-{risk_level}'>{emoji} {label}</span>"

def render_xp_progress(xp: int, level: int):
    """Render XP progress bar with level info"""
    gamification = GamificationSystem()
    current_level_xp = (level - 1) * 100
    next_level_xp = level * 100
    progress = ((xp - current_level_xp) / 100) * 100
    
    level_title = gamification.get_level_title(level)
    
    st.markdown(f"""
    <div class="stCard" style="background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%);">
        <div style="color: white;">
            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 8px;">Level {level}</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 12px;">{level_title}</div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 10px; overflow: hidden; margin-bottom: 8px;">
                <div style="background: white; height: 100%; width: {min(progress, 100)}%; transition: width 0.3s ease;"></div>
            </div>
            <div style="font-size: 0.85rem; opacity: 0.9;">{xp} XP • {max(0, next_level_xp - xp)} XP to next level</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_badge_showcase(badges_list: List[str], gamification: GamificationSystem):
    """Render earned badges in modern cards"""
    if not badges_list:
        st.info("Complete actions to earn badges! 🎯")
        return
    
    st.markdown("<div class='badge-container'>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, badge_id in enumerate(badges_list):
        badge = gamification.badges.get(badge_id)
        if badge:
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="badge-item">
                    <div style="font-size: 1.5rem; text-align: center; margin-bottom: 8px;">{badge['name'].split()[0]}</div>
                    <div style="font-weight: 600; text-align: center; color: #1e293b; margin-bottom: 4px;">{' '.join(badge['name'].split()[1:])}</div>
                    <div style="font-size: 0.8rem; text-align: center; color: #64748b;">{badge['description']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================================
# PAGE RENDERERS
# ============================================================================

def render_home():
    """Render home/dashboard page"""
    st.title("🏠 Dashboard")
    
    if not st.session_state.current_user:
        st.info("👋 Please enter your username in the sidebar to get started!")
        return
    
    user_data = st.session_state.db.get_user(st.session_state.current_user)
    
    if not user_data:
        st.warning("User data not found. Creating profile...")
        st.session_state.db.create_user(st.session_state.current_user)
        user_data = st.session_state.db.get_user(st.session_state.current_user)
    
    # Welcome message
    st.markdown(f"### Welcome back, **{st.session_state.current_user}**! 👋")
    st.caption("Your AI Cyber Coach - Stay Secure, Stay Smart")
    
    st.markdown("---")
    
    # XP Progress
    render_xp_progress(user_data['xp_points'], user_data['level'])
    
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card("Security Score", f"{user_data['total_score']:.0f}/100", icon="🛡️")
    
    with col2:
        render_metric_card("XP Points", f"{user_data['xp_points']}", icon="⭐")
    
    with col3:
        render_metric_card("Streak", f"{user_data['streak_days']} days", icon="🔥")
    
    with col4:
        badges = st.session_state.gamification.get_earned_badges(
            st.session_state.password_strength,
            st.session_state.url_checked,
            st.session_state.url_safe,
            st.session_state.overall_score,
            user_data['streak_days'],
            st.session_state.simulation_done
        )
        render_metric_card("Badges", f"{len(badges)}/9", icon="🏆")
    
    st.markdown("---")
    
    # Recent Activity
    st.markdown("### 📊 Recent Activity")
    
    history = st.session_state.db.get_user_history(st.session_state.current_user, 10)
    
    if history:
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df = df[['timestamp', 'type', 'score', 'risk_level']]
        df.columns = ['Time', 'Check Type', 'Score', 'Risk Level']
        
        # Create activity chart
        fig = px.line(
            df,
            x='Time',
            y='Score',
            markers=True,
            title='Security Score Trend',
            color_discrete_sequence=['#8b5cf6']
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.95)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            font={'color': '#1e293b', 'family': 'Inter'},
            height=300,
            xaxis_title='',
            yaxis_title='Score',
            yaxis_range=[0, 100]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No activity yet. Start by analyzing passwords or URLs! 🚀")
    
    st.markdown("---")
    
    # Earned Badges
    st.markdown("### 🏆 Your Badges")
    
    badges = st.session_state.gamification.get_earned_badges(
        st.session_state.password_strength,
        st.session_state.url_checked,
        st.session_state.url_safe,
        st.session_state.overall_score,
        user_data['streak_days'],
        st.session_state.simulation_done
    )
    
    render_badge_showcase(badges, st.session_state.gamification)

def render_analyzer():
    """Render password and URL analyzer page"""
    st.title("🔍 Security Analyzer")
    st.caption("Analyze passwords and URLs for security risks")
    
    tab1, tab2 = st.tabs(["🔐 Password Analyzer", "🔗 URL Analyzer"])
    
    with tab1:
        st.markdown("### Password Strength Checker")
        st.info("⚠️ **Privacy First:** Your password is never stored or logged!")
        
        password = st.text_input(
            "Enter password to analyze",
            type="password",
            key="password_input",
            help="Your password is analyzed locally and never saved"
        )
        
        if st.button("🔍 Analyze Password", key="analyze_pwd", use_container_width=True):
            if password:
                with st.spinner("Analyzing password security..."):
                    result = st.session_state.analyzer.analyze_password(password)
                    
                    # Update session state
                    st.session_state.password_strength = True
                    st.session_state.overall_score = result['score']
                    st.session_state.password_result = result
                    
                    # Add to blockchain
                    st.session_state.blockchain.add_event(
                        'password_check',
                        f"Password analyzed - Strength: {result['strength']}",
                        {'score': result['score'], 'risk_level': result['risk_level']}
                    )
                    
                    # Update user score
                    if st.session_state.current_user:
                        st.session_state.db.update_user_score(
                            st.session_state.current_user,
                            result['score'],
                            'password',
                            result['entropy'],
                            result['risk_level']
                        )
                        
                        # Award XP
                        xp = st.session_state.gamification.award_xp('password_check')
                        st.toast(f"✨ +{xp} XP earned!", icon="⭐")
                
                st.markdown("---")
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {result['strength']} Password")
                    st.markdown(render_risk_badge(result['risk_level']), unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(result['score'] / 100)
                    st.markdown(f"**Score:** {result['score']}/100")
                    
                    st.markdown("---")
                    
                    # Details
                    st.markdown("**📊 Analysis Details:**")
                    st.markdown(f"- **Entropy:** {result['entropy']} bits")
                    st.markdown(f"- **Estimated Crack Time:** {result['crack_time']}")
                    
                    st.markdown("---")
                    
                    # Character composition
                    st.markdown("**🔤 Character Composition:**")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Lowercase", "✅" if result['has_lower'] else "❌")
                    col_b.metric("Uppercase", "✅" if result['has_upper'] else "❌")
                    col_c.metric("Numbers", "✅" if result['has_digit'] else "❌")
                    col_d.metric("Special", "✅" if result['has_special'] else "❌")
                
                with col2:
                    # Visual gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Strength", 'font': {'size': 16, 'color': '#1e293b', 'family': 'Inter'}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1e293b"},
                            'bar': {'color': "#8b5cf6"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#e2e8f0",
                            'steps': [
                                {'range': [0, 20], 'color': '#fecaca'},
                                {'range': [20, 40], 'color': '#fed7aa'},
                                {'range': [40, 60], 'color': '#fde68a'},
                                {'range': [60, 80], 'color': '#d1fae5'},
                                {'range': [80, 100], 'color': '#a7f3d0'}
                            ],
                        }
                    ))
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(255, 255, 255, 0)',
                        font={'color': '#1e293b', 'family': 'Inter'},
                        height=250,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feedback
                if result['feedback']:
                    st.markdown("### 💡 Recommendations")
                    for feedback in result['feedback']:
                        if "excellent" in feedback.lower() or "good" in feedback.lower():
                            st.success(feedback)
                        else:
                            st.warning(feedback)
            else:
                st.warning("Please enter a password to analyze")
    
    with tab2:
        st.markdown("### URL Safety Checker")
        st.info("🔍 Check if a URL is safe before visiting")
        
        url = st.text_input(
            "Enter URL to analyze",
            key="url_input",
            placeholder="example.com or https://example.com",
            help="Enter any URL to check for phishing and security risks"
        )
        
        if st.button("🔍 Analyze URL", key="analyze_url", use_container_width=True):
            if url:
                with st.spinner("Analyzing URL safety..."):
                    result = st.session_state.analyzer.analyze_url(url)
                    
                    # Update session state
                    st.session_state.url_checked = True
                    st.session_state.url_safe = result['score'] > 60
                    st.session_state.overall_score = result['score']
                    st.session_state.url_result = result
                    
                    # Add to blockchain
                    st.session_state.blockchain.add_event(
                        'url_check',
                        f"URL analyzed - Domain: {result['domain']}",
                        {'score': result['score'], 'risk_level': result['risk_level']}
                    )
                    
                    # Update user score
                    if st.session_state.current_user:
                        st.session_state.db.update_user_score(
                            st.session_state.current_user,
                            result['score'],
                            'url',
                            0,
                            result['risk_level']
                        )
                        
                        # Award XP
                        xp = st.session_state.gamification.award_xp('url_check')
                        st.toast(f"✨ +{xp} XP earned!", icon="⭐")
                
                st.markdown("---")
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"### {result['explanation']}")
                    st.markdown(render_risk_badge(result['risk_level']), unsafe_allow_html=True)
                    
                    # Progress bar
                    st.progress(result['score'] / 100)
                    st.markdown(f"**Safety Score:** {result['score']}/100")
                    
                    st.markdown("---")
                    
                    # Domain info
                    st.markdown("**🌐 Domain Information:**")
                    st.markdown(f"- **Domain:** `{result['domain']}`")
                    if result.get('subdomain'):
                        st.markdown(f"- **Subdomain:** `{result['subdomain']}`")
                    if result.get('tld'):
                        st.markdown(f"- **TLD:** `{result['tld']}`")
                    st.markdown(f"- **HTTPS:** {'✅ Yes' if result['has_https'] else '❌ No'}")
                    
                    st.markdown("---")
                    
                    # ML Analysis
                    st.markdown("**🤖 ML Analysis:**")
                    st.markdown(f"- **Phishing Confidence:** {result['ml_confidence']:.1%}")
                    
                    if result['ml_confidence'] > 0.7:
                        st.error("⚠️ High probability of phishing detected by AI model")
                    elif result['ml_confidence'] > 0.4:
                        st.warning("⚠️ Moderate phishing indicators detected")
                    else:
                        st.success("✅ Low phishing probability")
                
                with col2:
                    # Visual gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result['score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Safety", 'font': {'size': 16, 'color': '#1e293b', 'family': 'Inter'}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1e293b"},
                            'bar': {'color': "#3b82f6"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#e2e8f0",
                            'steps': [
                                {'range': [0, 20], 'color': '#fecaca'},
                                {'range': [20, 40], 'color': '#fed7aa'},
                                {'range': [40, 60], 'color': '#fde68a'},
                                {'range': [60, 80], 'color': '#d1fae5'},
                                {'range': [80, 100], 'color': '#a7f3d0'}
                            ],
                        }
                    ))
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(255, 255, 255, 0)',
                        font={'color': '#1e293b', 'family': 'Inter'},
                        height=250,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Suspicious patterns
                if result['suspicious_patterns']:
                    st.markdown("### 🚨 Security Analysis")
                    
                    safe_patterns = [p for p in result['suspicious_patterns'] if 'no suspicious' in p.lower() or 'safe' in p.lower()]
                    warning_patterns = [p for p in result['suspicious_patterns'] if p not in safe_patterns]
                    
                    if warning_patterns:
                        for pattern in warning_patterns:
                            st.warning(f"⚠️ {pattern}")
                    
                    if safe_patterns:
                        for pattern in safe_patterns:
                            st.success(f"✅ {pattern}")
            else:
                st.warning("Please enter a URL to analyze")

def render_simulator():
    """Render threat simulator page"""
    st.title("🎮 Threat Simulator")
    st.caption("Test your cybersecurity knowledge!")
    
    if not st.session_state.current_user:
        st.warning("Please log in to use the simulator")
        return
    
    st.markdown("""
    <div class="stCard">
        <h3>🎯 How It Works</h3>
        <p>We'll show you URLs and passwords. Your job is to identify if they're safe or dangerous!</p>
        <p><strong>Rewards:</strong> Earn XP and badges for correct answers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Challenge types
    challenge_type = st.radio(
        "Select Challenge Type",
        ["🔗 URL Safety", "🔐 Password Strength"],
        horizontal=True
    )
    
    if challenge_type == "🔗 URL Safety":
        st.markdown("### 🔗 Is This URL Safe?")
        
        # Generate challenge
        if 'current_challenge' not in st.session_state or st.button("🎲 New Challenge"):
            safe_urls = ['google.com', 'github.com', 'microsoft.com', 'apple.com', 'amazon.com']
            phishing_urls = [
                'g00gle-verify.com', 'paypa1-secure.com', 'microso ft-login.net',
                'app1e-id-verify.org', 'amaz0n-security.com', 'netfIix-payment.com'
            ]
            
            is_safe = random.choice([True, False])
            
            if is_safe:
                url = random.choice(safe_urls)
                st.session_state.current_challenge = {'url': url, 'is_safe': True}
            else:
                url = random.choice(phishing_urls)
                st.session_state.current_challenge = {'url': url, 'is_safe': False}
        
        challenge = st.session_state.current_challenge
        
        st.markdown(f"""
        <div class="stCard" style="text-align: center; font-size: 1.5rem; padding: 40px;">
            <code>{challenge['url']}</code>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Safe", key="safe_btn", use_container_width=True):
                correct = challenge['is_safe']
                
                if correct:
                    st.success("🎉 Correct! This URL is safe.")
                    xp = st.session_state.gamification.award_xp('simulation_correct')
                    st.balloons()
                else:
                    st.error("❌ Wrong! This URL is suspicious.")
                    xp = st.session_state.gamification.award_xp('simulation_wrong')
                
                # Record challenge
                st.session_state.db.record_challenge(
                    st.session_state.current_user,
                    'url_safety',
                    correct
                )
                
                st.session_state.simulation_done = True
                st.toast(f"+{xp} XP earned!", icon="⭐")
                
                # Add to blockchain
                st.session_state.blockchain.add_event(
                    'simulation',
                    f"URL challenge completed - {'Correct' if correct else 'Incorrect'}",
                    {'url': challenge['url'], 'correct': correct}
                )
        
        with col2:
            if st.button("🚨 Dangerous", key="danger_btn", use_container_width=True):
                correct = not challenge['is_safe']
                
                if correct:
                    st.success("🎉 Correct! This URL is dangerous.")
                    xp = st.session_state.gamification.award_xp('simulation_correct')
                    st.balloons()
                else:
                    st.error("❌ Wrong! This URL is actually safe.")
                    xp = st.session_state.gamification.award_xp('simulation_wrong')
                
                # Record challenge
                st.session_state.db.record_challenge(
                    st.session_state.current_user,
                    'url_safety',
                    correct
                )
                
                st.session_state.simulation_done = True
                st.toast(f"+{xp} XP earned!", icon="⭐")
                
                # Add to blockchain
                st.session_state.blockchain.add_event(
                    'simulation',
                    f"URL challenge completed - {'Correct' if correct else 'Incorrect'}",
                    {'url': challenge['url'], 'correct': correct}
                )
    
    else:
        st.markdown("### 🔐 Is This Password Strong?")
        
        # Generate challenge
        if 'current_pwd_challenge' not in st.session_state or st.button("🎲 New Challenge"):
            strong_pwds = ['xK9$mP3#nQ7!vR2', 'Tr0ub4dor&3*Qx', 'P@ssw0rd!2024#Secure']
            weak_pwds = ['password123', '12345678', 'qwerty', 'abc123456']
            
            is_strong = random.choice([True, False])
            
            if is_strong:
                pwd = random.choice(strong_pwds)
                st.session_state.current_pwd_challenge = {'password': pwd, 'is_strong': True}
            else:
                pwd = random.choice(weak_pwds)
                st.session_state.current_pwd_challenge = {'password': pwd, 'is_strong': False}
        
        challenge = st.session_state.current_pwd_challenge
        
        st.markdown(f"""
        <div class="stCard" style="text-align: center; font-size: 1.5rem; padding: 40px;">
            <code>{challenge['password']}</code>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💪 Strong", key="strong_btn", use_container_width=True):
                correct = challenge['is_strong']
                
                if correct:
                    st.success("🎉 Correct! This password is strong.")
                    xp = st.session_state.gamification.award_xp('simulation_correct')
                    st.balloons()
                else:
                    st.error("❌ Wrong! This password is weak.")
                    xp = st.session_state.gamification.award_xp('simulation_wrong')
                
                # Record challenge
                st.session_state.db.record_challenge(
                    st.session_state.current_user,
                    'password_strength',
                    correct
                )
                
                st.session_state.simulation_done = True
                st.toast(f"+{xp} XP earned!", icon="⭐")
                
                # Add to blockchain
                st.session_state.blockchain.add_event(
                    'simulation',
                    f"Password challenge completed - {'Correct' if correct else 'Incorrect'}",
                    {'correct': correct}
                )
        
        with col2:
            if st.button("😰 Weak", key="weak_btn", use_container_width=True):
                correct = not challenge['is_strong']
                
                if correct:
                    st.success("🎉 Correct! This password is weak.")
                    xp = st.session_state.gamification.award_xp('simulation_correct')
                    st.balloons()
                else:
                    st.error("❌ Wrong! This password is actually strong.")
                    xp = st.session_state.gamification.award_xp('simulation_wrong')
                
                # Record challenge
                st.session_state.db.record_challenge(
                    st.session_state.current_user,
                    'password_strength',
                    correct
                )
                
                st.session_state.simulation_done = True
                st.toast(f"+{xp} XP earned!", icon="⭐")
                
                # Add to blockchain
                st.session_state.blockchain.add_event(
                    'simulation',
                    f"Password challenge completed - {'Correct' if correct else 'Incorrect'}",
                    {'correct': correct}
                )

def render_leaderboard():
    """Render leaderboard page"""
    st.title("🏆 Leaderboard")
    st.caption("Top cybersecurity champions")
    
    leaderboard = st.session_state.db.get_leaderboard(20)
    
    if leaderboard:
        # Top 3 podium
        if len(leaderboard) >= 3:
            st.markdown("### 🎖️ Top 3 Champions")
            
            col1, col2, col3 = st.columns(3)
            
            with col2:
                st.markdown(f"""
                <div class="stCard" style="background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); text-align: center; padding: 30px;">
                    <div style="font-size: 3rem;">🥇</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin: 12px 0;">{leaderboard[0]['username']}</div>
                    <div style="font-size: 1.2rem; color: #64748b;">Score: {leaderboard[0]['score']:.0f}/100</div>
                    <div style="font-size: 1rem; color: #64748b;">XP: {leaderboard[0]['xp']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col1:
                st.markdown(f"""
                <div class="stCard" style="background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%); text-align: center; padding: 25px; margin-top: 40px;">
                    <div style="font-size: 2.5rem;">🥈</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #1e293b; margin: 8px 0;">{leaderboard[1]['username']}</div>
                    <div style="font-size: 1rem; color: #64748b;">Score: {leaderboard[1]['score']:.0f}/100</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stCard" style="background: linear-gradient(135deg, #cd7f32 0%, #e6a57e 100%); text-align: center; padding: 25px; margin-top: 40px;">
                    <div style="font-size: 2.5rem;">🥉</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #1e293b; margin: 8px 0;">{leaderboard[2]['username']}</div>
                    <div style="font-size: 1rem; color: #64748b;">Score: {leaderboard[2]['score']:.0f}/100</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Full leaderboard
        st.markdown("### 📊 Complete Rankings")
        
        df = pd.DataFrame(leaderboard)
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'username', 'score', 'xp', 'level', 'badges']]
        df.columns = ['Rank', 'Username', 'Score', 'XP', 'Level', 'Badges']
        
        # Add medal emojis
        df['Rank'] = df['Rank'].apply(lambda x: f"🥇 {x}" if x == 1 else f"🥈 {x}" if x == 2 else f"🥉 {x}" if x == 3 else str(x))
        
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No leaderboard entries yet. Be the first! 🚀")

def render_blockchain():
    """Render blockchain audit trail page"""
    st.title("⛓️ Blockchain Audit Trail")
    st.caption("Immutable record of all security events")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Commit Events", use_container_width=True):
            block = st.session_state.blockchain.commit_events()
            if block:
                st.success(f"✅ Block #{block['index']} created")
                st.balloons()
            else:
                st.info("No pending events to commit")
    
    with col2:
        if st.button("🔄 Verify Chain", use_container_width=True):
            is_valid, error_idx = st.session_state.blockchain.verify_chain()
            if is_valid:
                st.success("✅ Chain integrity verified")
            else:
                st.error(f"🚨 Compromised at block {error_idx}")
    
    # Chain status
    is_valid, error_idx = st.session_state.blockchain.verify_chain()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_metric_card(
            "Total Blocks",
            str(len(st.session_state.blockchain.chain)),
            icon="⛓️"
        )
    
    with col2:
        render_metric_card(
            "Pending Events",
            str(len(st.session_state.blockchain.pending_events)),
            icon="⏳"
        )
    
    with col3:
        status = "✅ Valid" if is_valid else "🚨 Compromised"
        render_metric_card(
            "Chain Status",
            status,
            icon="🔒"
        )
    
    st.markdown("---")
    
    # Visualization
    if st.session_state.blockchain.chain:
        st.markdown("### 📊 Chain Visualization")
        fig = st.session_state.blockchain.create_visual_graph()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            chain_json = st.session_state.blockchain.export_chain()
            st.download_button(
                "📥 Export Blockchain JSON",
                data=chain_json,
                file_name=f"blockchain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            backup = st.session_state.db.backup_database()
            if backup:
                st.download_button(
                    "💾 Backup Database",
                    data=backup,
                    file_name=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Recent blocks
        st.markdown("### 📦 Recent Blocks")
        
        for block in reversed(st.session_state.blockchain.chain[-5:]):
            with st.expander(f"Block #{block['index']} - {block['timestamp'][:19]}"):
                st.code(f"Hash: {block['hash']}", language="text")
                st.code(f"Previous: {block['previous_hash']}", language="text")
                
                st.markdown("**Events:**")
                for event in block['events']:
                    st.markdown(f"- **{event['type']}**: {event['description']}")
                    if 'metadata' in event:
                        st.json(event['metadata'])
    else:
        st.info("No blocks in chain yet. Start using the app to create blocks! 🚀")

def render_admin():
    """Render admin analytics page"""
    st.title("👑 Admin Analytics")
    st.caption("Global system statistics and insights")
    
    st.warning("⚠️ This page shows aggregated statistics for all users")
    
    stats = st.session_state.db.get_global_stats()
    
    if stats:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_metric_card("Total Users", str(stats['total_users']), icon="👥")
        
        with col2:
            render_metric_card("Avg Score", f"{stats['avg_score']:.1f}/100", icon="📊")
        
        with col3:
            render_metric_card("Total Checks", str(stats['total_checks']), icon="🔍")
        
        with col4:
            render_metric_card("Threats Detected", str(stats['threats_detected']), icon="🚨")
        
        st.markdown("---")
        
        # ML Model Performance
        st.markdown("### 🤖 ML Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metrics = st.session_state.analyzer.metrics
            if metrics:
                st.markdown(f"""
                <div class="stCard">
                    <h4 style="margin-top: 0;">Model Metrics</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 20px;">
                        <div>
                            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 4px;">Accuracy</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #8b5cf6;">{metrics['accuracy']*100:.1f}%</div>
                        </div>
                        <div>
                            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 4px;">Precision</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #3b82f6;">{metrics['precision']*100:.1f}%</div>
                        </div>
                        <div>
                            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 4px;">Recall</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{metrics['recall']*100:.1f}%</div>
                        </div>
                        <div>
                            <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 4px;">F1 Score</div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">{metrics['f1_score']*100:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            if metrics and 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Legitimate', 'Phishing'],
                    y=['Legitimate', 'Phishing'],
                    color_continuous_scale='Purples',
                    text_auto=True
                )
                fig.update_layout(
                    title="Confusion Matrix",
                    paper_bgcolor='rgba(255, 255, 255, 0.95)',
                    plot_bgcolor='rgba(255, 255, 255, 0.95)',
                    font={'color': '#1e293b', 'family': 'Inter'},
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature Importance
        st.markdown("### 🔬 Top Phishing Indicators")
        
        if hasattr(st.session_state.analyzer.model, 'coef_'):
            coef = st.session_state.analyzer.model.coef_[0]
            feature_names = st.session_state.analyzer.feature_names
            
            # Get top 10 features
            top_indices = np.argsort(np.abs(coef))[-10:]
            top_features = [(feature_names[i], float(coef[i])) for i in top_indices]
            top_features.reverse()
            
            df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
            
            fig = px.bar(
                df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Most Important Features for Phishing Detection',
                color='Importance',
                color_continuous_scale=['#10b981', '#8b5cf6', '#ef4444']
            )
            fig.update_layout(
                paper_bgcolor='rgba(255, 255, 255, 0.95)',
                plot_bgcolor='rgba(255, 255, 255, 0.95)',
                font={'color': '#1e293b', 'family': 'Inter'},
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # System Info
        st.markdown("### ⚙️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Database:** SQLite with connection pooling  
            **ML Model:** Logistic Regression (cached)  
            **Blockchain Blocks:** {len(st.session_state.blockchain.chain)}  
            **Total XP Earned:** {stats['total_xp']:,}
            """)
        
        with col2:
            st.success("""
            **Security Features:**  
            ✅ No password storage  
            ✅ SSRF protection  
            ✅ URL validation  
            ✅ Blockchain integrity  
            ✅ GDPR compliance
            """)

def render_reports():
    """Render PDF reports page"""
    st.title("📄 Security Reports")
    st.caption("Generate comprehensive security reports")
    
    if not st.session_state.current_user:
        st.warning("Please log in to generate reports")
        return
    
    if not HAS_REPORTLAB:
        st.error("ReportLab not installed. Please run: pip install reportlab")
        return
    
    user_data = st.session_state.db.get_user(st.session_state.current_user)
    
    if not user_data:
        st.warning("No user data found")
        return
    
    st.markdown("""
    <div class="stCard">
        <h3>📊 Report Preview</h3>
        <p>Your comprehensive security report includes:</p>
        <ul>
            <li>✅ Overall security score and XP</li>
            <li>🔐 Password strength analysis</li>
            <li>🔗 URL safety evaluation</li>
            <li>🏆 Earned badges and achievements</li>
            <li>📈 Recent activity history</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Report Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Report Details")
        st.markdown(f"**Username:** {st.session_state.current_user}")
        st.markdown(f"**XP Points:** {user_data['xp_points']}")
        st.markdown(f"**Level:** {user_data['level']}")
        st.markdown(f"**Security Score:** {user_data['total_score']:.1f}/100")
        st.markdown(f"**Streak:** {user_data['streak_days']} days")
    
    with col2:
        st.markdown("### 🏆 Achievements")
        badges = st.session_state.gamification.get_earned_badges(
            st.session_state.password_strength,
            st.session_state.url_checked,
            st.session_state.url_safe,
            st.session_state.overall_score,
            user_data['streak_days'],
            st.session_state.simulation_done
        )
        st.markdown(f"**Badges Earned:** {len(badges)}/9")
        
        for badge_id in badges[:5]:  # Show first 5
            badge = st.session_state.gamification.badges.get(badge_id)
            if badge:
                st.markdown(f"• {badge['name']}")
    
    st.markdown("---")
    
    # Generate Report
    if st.button("📥 Generate PDF Report", use_container_width=True, type="primary"):
        with st.spinner("Generating your security report..."):
            history = st.session_state.db.get_user_history(st.session_state.current_user, 10)
            
            pdf_data = generate_pdf_report(
                st.session_state.current_user,
                user_data,
                st.session_state.get('password_result', {}),
                st.session_state.get('url_result', {}),
                badges,
                history
            )
            
            if pdf_data:
                st.success("✅ Report generated successfully!")
                
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"security_report_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.balloons()
                
                # Add to blockchain
                st.session_state.blockchain.add_event(
                    'report_generated',
                    f"PDF report generated for {st.session_state.current_user}",
                    {'timestamp': datetime.now().isoformat()}
                )
            else:
                st.error("Failed to generate report")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_user = None
        st.session_state.overall_score = 0
        st.session_state.password_strength = False
        st.session_state.url_checked = False
        st.session_state.url_safe = False
        st.session_state.simulation_done = False
        st.session_state.password_result = {}
        st.session_state.url_result = {}
        
        # Initialize components
        st.session_state.db = DatabaseManager()
        ml_components = load_ml_model()
        st.session_state.analyzer = SecurityAnalyzer(ml_components)
        st.session_state.blockchain = BlockchainLedger(st.session_state.db)
        st.session_state.gamification = GamificationSystem()

def render_sidebar():
    """Render sidebar navigation and user info"""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 3rem;">🛡️</div>
            <h2 style="margin: 12px 0 4px 0; background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Smart Secure Assist</h2>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">Your AI Cyber Coach</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # User login
        st.markdown("### 👤 User Profile")
        username = st.text_input(
            "Username",
            value=st.session_state.current_user or "",
            key="username_input",
            placeholder="Enter your username"
        )
        
        if username and username != st.session_state.current_user:
            st.session_state.current_user = username
            st.session_state.db.create_user(username)
            st.session_state.db.update_streak(username)
            st.rerun()
        
        # Quick stats
        if st.session_state.current_user:
            user_data = st.session_state.db.get_user(st.session_state.current_user)
            if user_data:
                st.markdown("---")
                st.markdown("### 📊 Quick Stats")
                
                st.metric("Score", f"{user_data['total_score']:.0f}/100")
                st.metric("XP", f"{user_data['xp_points']}")
                st.metric("Level", f"{user_data['level']}")
                st.metric("Streak", f"{user_data['streak_days']} days")
                
                # Progress bar
                st.markdown("**XP Progress**")
                xp_progress = (user_data['xp_points'] % 100) / 100
                st.progress(xp_progress)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select Page",
            [
                "🏠 Home",
                "🔍 Analyzer",
                "🎮 Simulator",
                "🏆 Leaderboard",
                "⛓️ Blockchain",
                "👑 Admin",
                "📄 Reports"
            ],
            label_visibility="collapsed"
        )
        
        st.session_state.current_page = page
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; opacity: 0.7; font-size: 0.75rem; padding: 20px 0;">
            <strong>Smart Secure Assist v5.0</strong><br>
            🏆 MODERN EDITION<br>
            🔒 Privacy-First • 🛡️ Secure<br>
            <br>
            <em>Stay Secure, Stay Smart</em>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    # Page config
    st.set_page_config(
        page_title="Smart Secure Assist - AI Cyber Coach",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply theme
    apply_modern_theme()
    
    # Initialize
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Route to pages
    page = st.session_state.get('current_page', '🏠 Home')
    
    if page == '🏠 Home':
        render_home()
    elif page == '🔍 Analyzer':
        render_analyzer()
    elif page == '🎮 Simulator':
        render_simulator()
    elif page == '🏆 Leaderboard':
        render_leaderboard()
    elif page == '⛓️ Blockchain':
        render_blockchain()
    elif page == '👑 Admin':
        render_admin()
    elif page == '📄 Reports':
        render_reports()

if __name__ == "__main__":
    main()
