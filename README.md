# 🛡️ Smart Secure Assist – AI Cybersecurity Companion

### 💡 "Your AI Cyber Coach – Stay Secure, Stay Smart."

Smart Secure Assist is an **AI-powered cybersecurity assistant** built with **Python and Streamlit**, designed to help users analyze passwords and URLs, detect potential phishing threats, visualize blockchain audit trails, and gamify cybersecurity learning — all through a sleek, modern, light-themed UI.

---

## 🚀 Key Features

### 🔍 **Security Analyzer**
- **Password Strength Detection**  
  - Calculates entropy, strength score, and estimated crack time
  - Provides actionable improvement feedback
  
- **URL Phishing Detection**  
  - Combines ML model prediction (Logistic Regression + CountVectorizer) with rule-based heuristics
  - Detects suspicious patterns, keywords, and insecure configurations

### 📊 **Analytics Dashboard**
Personalized dashboard displaying:
- User's XP, badges, security scores, and streaks
- Recent activity charts and performance trends
- Global leaderboard and admin metrics

### 🎮 **Gamification System**
- Earn XP and unlock badges for analyzing threats, maintaining streaks, or completing simulations
- Level up from **Cyber Rookie → Elite Guardian** through consistent engagement
- Simulated threat challenges to test cybersecurity instincts

### ⛓️ **Blockchain Ledger**
- Immutable record of all user actions (without storing sensitive data)
- SHA-256-based block hashing with visualization through **Plotly**
- Verifiable chain integrity and optional JSON export

### 🧠 **AI Model**
- Logistic Regression–based phishing classifier
- Evaluates real and fake URLs using n-gram vectorization
- Shows accuracy, precision, recall, and confusion matrix

### 🧩 **Database (SQLite with Connection Pooling)**
- Manages users, history, leaderboard, and blockchain data
- GDPR-compliant data deletion and safe backup creation
- Designed for multi-session scalability

### 📄 **PDF Report Generator**
Generates a detailed, professional-grade PDF summarizing:
- Password and URL security analysis
- XP, levels, badges, and history
- Recommendations for better digital hygiene

---

## 🎨 Modern UI Design

- Built with **Inter font**, **pastel gradients**, and **soft glassmorphic cards**
- Minimal, responsive, and hackathon-ready layout
- Color-coded risk indicators:
  - ✅ Very Low Risk
  - ⚠️ Medium Risk
  - 🚨 Critical Risk
- Sidebar navigation for all features:
  - Home | Analyzer | Simulator | Dashboard | Blockchain | Reports | Admin

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit (Custom CSS & HTML Integration) |
| **Backend** | Python |
| **Database** | SQLite |
| **ML Model** | Logistic Regression + CountVectorizer |
| **Visualization** | Plotly, ReportLab |
| **Logging** | Python Logging |
| **Security** | SHA-256 Hashing, Local Processing (No password storage) |

---

## ⚙️ Installation & Setup

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/smart-secure-assist.git
cd smart-secure-assist
```

### **Step 2: Install Dependencies**
```bash
pip install streamlit pandas numpy scikit-learn plotly tldextract joblib reportlab
```

### **Step 3: Run the Application**
```bash
streamlit run smart_secure_assist_modern.py
```

### **Step 4: Access Locally**
Open your browser at `http://localhost:8501`

---

## 🧭 Navigation Overview

| Section | Description |
|---------|-------------|
| **Home Dashboard** | Overview of XP, score, streak, badges |
| **Analyzer** | Password & URL strength checker |
| **Simulator** | Gamified phishing/strength quiz |
| **Blockchain** | View and verify ledger integrity |
| **Leaderboard** | Global user rankings |
| **Reports** | Generate and download PDF summary |
| **Admin Panel** | Aggregate statistics and system insights |

---

## 🔒 Privacy & Security

- ✅ **No sensitive data stored** (no passwords saved)
- ✅ **Local-only execution** — no API calls to third parties
- ✅ **Blockchain ensures tamper-proof audit trail**
- ✅ **Fully GDPR-compliant** — users can delete their data anytime

---

## 🏆 Ideal For

- Hackathons & cybersecurity competitions
- Demonstrations of applied AI + Blockchain in security
- Educational use for password hygiene and phishing awareness

---

## 📸 Screenshots

> Add screenshots of your application here to showcase the UI

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Smart Secure Assist Team**

Enhanced for Modern UI – Smart Secure Assist v5.0

---

## 🙏 Acknowledgments

- Built with Streamlit for rapid prototyping
- Inspired by the need for accessible cybersecurity education
- Special thanks to the open-source community

---

**"Empower users to stay safe — not scared. Let AI be their cyber coach."** 🧠🛡️

⭐ Star this repo if you find it helpful!
