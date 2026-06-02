# 🤖 AI Expert Chatbot

A conversational AI chatbot built with **LangChain** + **Groq** + **Streamlit** that answers Machine Learning and AI questions in beginner-friendly language.

---

## Features

- 💬 Chat interface with persistent conversation memory
- 👤 Dynamic session management per user
- 🔒 Restricted to ML/AI topics only
- 🗑️ Clear memory and switch user from the sidebar
- ⚡ Powered by Groq's fast inference API

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `LangChain` | Chain + memory management |
| `Groq` | LLM inference (qwen3-32b) |
| `Streamlit` | Web UI |
| `python-dotenv` | API key management |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Abhineetsahay/AI-ChatBot
cd AI-ChatBot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com).

### 4. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Project Structure

```
ml-chatbot/
├── app.py            # Main Streamlit app
├── .env              # API keys (never commit this)
├── requirements.txt  # Dependencies
└── README.md
```

---

## Requirements

```
streamlit
langchain
langchain-core
langchain-groq
python-dotenv
```

## Usage

1. Enter your name to start a session
2. Ask any ML/AI question in plain English
3. Use **Clear Memory** in the sidebar to reset the conversation
4. Use **Switch User** to start a new session

> Questions outside ML/AI (anime, sports, etc.) will be politely declined.