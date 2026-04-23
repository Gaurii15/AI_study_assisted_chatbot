# 📚 StudyBuddyAI – AI-Powered Study Assistant

> **Learn smarter, not harder 🚀**

StudyBuddyAI is an AI-powered web application that helps students interact with their study materials (PDFs). It provides features like chatbot-based Q&A, notes generation, quiz creation, and answer evaluation — all in one place.

---

## 🌐 Live Demo
👉 **[Click here to use StudyBuddyAI](https://studybuddyai-by-gauriborse-x4pagzvtaucurrgjehzqwn.streamlit.app/)**

---

## 🖼️ Application Preview

![StudyBuddyAI Screenshot](overview_img.png)

---

## 💡 Features

- 💬 AI Chatbot (Ask questions from PDF)
- 📄 Upload PDF & extract text
- 🧠 Smart Summarization (5 key points)
- 📝 Structured Notes Generation
- ❓ Quiz Generation (Auto questions)
- ✅ Answer Evaluation with feedback
- 📊 Score tracking system
- 📥 Download notes as PDF
- 📂 Download chat history

---

## 🛠️ Tech Stack

- **Frontend & Backend:** Streamlit  
- **AI Model:** Groq API (LLM)  
- **Programming Language:** Python  
- **PDF Processing:** PyPDF2  
- **PDF Generation:** ReportLab  

---

## ⚙️ How It Works

1. User uploads a PDF 📄  
2. Text is extracted using PyPDF2  
3. AI processes queries using Groq API  
4. Responses are generated based on PDF content  
5. Additional features like notes, quiz, and summary are provided  

---

## 🧠 AI Integration

- Uses **Groq API** for fast AI responses  
- Implements **Prompt Engineering**:
  - Normal Mode  
  - Explain Simply Mode  
  - Exam Answer Mode  
- Context-aware responses using uploaded PDF  

---

## 🎯 Use Cases

- 📚 Student Learning Assistant  
- 📝 Exam Preparation Tool  
- 🔁 Quick Revision Tool  
- 🧠 Self-Assessment System  
- 📑 Digital Notes Generator  

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPO_LINK
cd ai-study-chatbot
2. Install dependencies
pip install -r requirements.txt
3. Add API Key

Create a .streamlit/secrets.toml file and add:

GROQ_API_KEY = "your_api_key_here"
4. Run the application
streamlit run app.py
⚠️ Limitations
Requires internet connection
API usage limits (Groq free tier)
Cannot process scanned PDFs without OCR
🔮 Future Enhancements
🎤 Voice interaction (speech-to-text)
📷 OCR support for scanned PDFs
🌍 Multi-language support
🔐 User authentication system
📊 Data analytics feature
👩‍💻 Author

Gauri Borse
✨ Crafted with ❤️

⭐ Support

If you like this project:

⭐ Star this repository
🍴 Fork it
📢 Share with others
