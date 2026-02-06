import numpy as np
import sounddevice as sd
import whisper
import torch
import gc
import threading
import queue

# Configuration
DEVICE = "mps"                # "cpu" | "cuda" | "mps"
MODEL_DIR = "./models"
MODEL_NAME = "large-v3-turbo"
SAMPLE_RATE = 16000           # Whisper expects 16kHz audio
SILENCE_THRESHOLD = 0.01      # RMS energy threshold (adjust for your environment)
SILENCE_DURATION = 1.5        # Seconds of silence before triggering transcription
CHUNK_DURATION = 0.1          # Seconds per audio chunk
MIN_SPEECH_DURATION = 0.5     # Minimum speech duration to bother transcribing


def main():
    print("Loading whisper model...")
    model = whisper.load_model(MODEL_NAME, device=DEVICE, download_root=MODEL_DIR)
    print("Model loaded!\n")

    while True:
        user_input = input("Type 'start' to begin recording: ").strip().lower()
        if user_input == "start":
            break

    print("\nRecording... Type 'stop' + Enter to end.\n")

    audio_q = queue.Queue()
    transcribe_q = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Audio: {status}]")
        audio_q.put(indata.copy())

    def stop_listener():
        while not stop_event.is_set():
            try:
                if input().strip().lower() == "stop":
                    stop_event.set()
            except EOFError:
                break

    def transcriber():
        while True:
            try:
                audio_data = transcribe_q.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue
            result = model.transcribe(
                audio_data, language="en", fp16=True, verbose=None
            )
            text = result["text"].strip()
            if text:
                print(f">> {text}")
            transcribe_q.task_done()

    threading.Thread(target=stop_listener, daemon=True).start()
    threading.Thread(target=transcriber, daemon=True).start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
        callback=audio_callback,
    )
    stream.start()

    audio_buffer = []
    is_speaking = False
    silence_chunks = 0
    silence_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    min_speech_chunks = int(MIN_SPEECH_DURATION / CHUNK_DURATION)

    while not stop_event.is_set():
        try:
            chunk = audio_q.get(timeout=0.5)
        except queue.Empty:
            continue

        energy = np.sqrt(np.mean(chunk ** 2))

        if energy > SILENCE_THRESHOLD:
            is_speaking = True
            silence_chunks = 0
            audio_buffer.append(chunk)
        elif is_speaking:
            audio_buffer.append(chunk)
            silence_chunks += 1

            if silence_chunks >= silence_needed:
                speech_chunks = len(audio_buffer) - silence_chunks
                if speech_chunks >= min_speech_chunks:
                    audio_data = np.concatenate(audio_buffer, axis=0).flatten()
                    transcribe_q.put(audio_data)
                audio_buffer.clear()
                is_speaking = False
                silence_chunks = 0

    stream.stop()
    stream.close()

    # Transcribe any remaining audio in the buffer
    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0).flatten()
        transcribe_q.put(audio_data)

    transcribe_q.join()
    print("\nRecording stopped.")

    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
