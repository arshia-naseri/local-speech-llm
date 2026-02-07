# 🗣️ Local Voice Assistant

A fully offline voice assistant that chains speech-to-text, a local LLM, and text-to-speech into a seamless conversational pipeline. No internet connection required — all inference runs locally on your machine.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-Apple_Silicon-000000?logo=apple&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2-blue)
![Offline](https://img.shields.io/badge/100%25-Offline-brightgreen)

## 🏗️ Architecture

```
🎙️ Microphone → Whisper (STT) → Ollama / Llama 3.2 (LLM) → Piper (TTS) → 🔊 Speaker
```

The final implementation (`voice_assistant.py`) wraps this pipeline in a tkinter GUI with initialize, talk, and stop controls.

## 🧩 Components

| Module | Technology | Purpose |
|---|---|---|
| `My_transcriber.py` | OpenAI Whisper (`large-v3-turbo`) | Real-time speech-to-text with voice activity detection |
| `My_LLM.py` | Ollama (`llama3.2`) | Local LLM chat with conversation history |
| `My_tts.py` | Piper TTS (ONNX) | Neural text-to-speech via `ffplay` streaming |
| `voice_assistant.py` | tkinter | GUI that orchestrates the full pipeline |

### 🎤 Speech-to-Text (Whisper)

- Model: `large-v3-turbo` on MPS (Apple Silicon GPU)
- Built-in voice activity detection (VAD) using RMS energy thresholds
- Configurable silence duration, speech minimum, and chunk size
- FP16 precision for faster inference

### 🧠 LLM (Ollama)

- Connects to a local Ollama server (`localhost:11434`)
- Streaming responses for low latency
- Maintains conversation history across turns
- Optional web search via DuckDuckGo for queries about weather, news, prices, etc.

### 🔊 Text-to-Speech (Piper)

- Offline neural TTS with ONNX voice models
- Voices included: `joe-medium` (default), `lessac-high`
- Real-time audio streaming through `ffplay`
- Stoppable playback for interruption support

## 📋 Prerequisites

- **macOS** with Apple Silicon (uses MPS acceleration)
- **Python 3.10+**
- **[Ollama](https://ollama.com/)** installed with a model pulled (default: `llama3.2`)
- **[FFmpeg](https://ffmpeg.org/)** installed (`ffplay` is used for audio playback)
- **[Piper TTS](https://github.com/rhasspy/piper)** binary available on PATH

## ⚙️ Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd local-speech-llm
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install openai-whisper torch sounddevice numpy requests piper-tts duckduckgo-search
   ```

3. Download models:
   - **Whisper**: automatically downloads on first run into `models/`
   - **Piper voices**: place `.onnx` and `.onnx.json` files in `voices/`
     (e.g. `en_US-joe-medium.onnx` from the [Piper voices repository](https://github.com/rhasspy/piper/blob/master/VOICES.md))

4. Start the Ollama server:
   ```bash
   ollama serve
   # or simply:
   ./start.sh
   ```

5. Pull a model (if not already done):
   ```bash
   ollama pull llama3.2
   ```

## 🚀 Usage

### Voice Assistant (GUI)

```bash
python voice_assistant.py
```

1. Click **Initialize** to load all models
2. Click **Talk** to start speaking
3. Wait for the assistant to transcribe, think, and respond
4. Click **Stop** at any time to interrupt

### Other Scripts

| Script | Description |
|---|---|
| `local_llm_chatbot.py` | Text-only chatbot in the terminal |
| `live_transcribe.py` | Continuous live transcription to console |
| `text_to_speech.py` | TTS demo — speaks a sample text |
| `whisper_opensource.py` | Whisper benchmark on audio files |

### 📓 Jupyter Notebooks

- `lcoal_llm_text_only.ipynb` — LLM experimentation (prompting, streaming, web search)
- `text_to_audio.ipynb` — TTS engine comparison (pyttsx3 vs gTTS vs Piper)

## 📁 Project Structure

```
local-speech-llm/
├── voice_assistant.py      # Main GUI application
├── My_transcriber.py       # Whisper STT module
├── My_LLM.py               # Ollama LLM module
├── My_tts.py               # Piper TTS module
├── live_transcribe.py       # Continuous transcription script
├── local_llm_chatbot.py     # Terminal chatbot
├── text_to_speech.py        # TTS demo
├── whisper_opensource.py     # Whisper benchmark
├── start.sh                 # Launches Ollama server
├── models/                  # Whisper model files (git-ignored)
├── voices/                  # Piper ONNX voice models (git-ignored)
└── audios/                  # Test audio files
```

## 🔄 How It Works

1. **Listening** — `sounddevice` captures microphone audio in 0.1s chunks. A simple VAD monitors RMS energy; once 1.5 seconds of silence follows detected speech, the audio buffer is sent to Whisper.
2. **Transcription** — Whisper processes the audio on the MPS device and returns text.
3. **LLM Response** — The transcribed text (optionally augmented with web search results) is sent to Ollama. The response streams back token-by-token.
4. **Speech** — Piper converts the LLM response to audio, piped directly to `ffplay` for near-instant playback.

The GUI runs each pipeline step in a background thread so the interface stays responsive, and the stop button can interrupt at any stage.
