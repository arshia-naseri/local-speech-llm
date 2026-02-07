import numpy as np
import sounddevice as sd
import whisper
import torch
import gc


class MyTranscriber:
    DEVICE = "mps"
    MODEL_DIR = "./models"
    MODEL_NAME = "large-v3-turbo"
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION = 1.5
    CHUNK_DURATION = 0.1
    MIN_SPEECH_DURATION = 0.5

    def __init__(self, model_name: str = None, device: str = None):
        if model_name:
            self.MODEL_NAME = model_name
        if device:
            self.DEVICE = device
        self.model = None
        self._stopped = False

    def init_model(self):
        print("[Transcriber] Loading whisper model...")
        self.model = whisper.load_model(
            self.MODEL_NAME, device=self.DEVICE, download_root=self.MODEL_DIR
        )
        print("[Transcriber] Model loaded!")

    def stop(self):
        self._stopped = True

    def listen_and_transcribe(self) -> str:
        if self.model is None:
            raise RuntimeError("Model not initialized. Call init_model() first.")

        self._stopped = False
        audio_buffer = []
        is_speaking = False
        silence_chunks = 0
        silence_needed = int(self.SILENCE_DURATION / self.CHUNK_DURATION)
        min_speech_chunks = int(self.MIN_SPEECH_DURATION / self.CHUNK_DURATION)
        done = False
        captured_audio = None

        def audio_callback(indata, frames, time_info, status):
            nonlocal is_speaking, silence_chunks, done, captured_audio

            if done or self._stopped:
                done = True
                return

            chunk = indata.copy()
            energy = np.sqrt(np.mean(chunk ** 2))

            if energy > self.SILENCE_THRESHOLD:
                is_speaking = True
                silence_chunks = 0
                audio_buffer.append(chunk)
            elif is_speaking:
                audio_buffer.append(chunk)
                silence_chunks += 1

                if silence_chunks >= silence_needed:
                    speech_chunks = len(audio_buffer) - silence_chunks
                    if speech_chunks >= min_speech_chunks:
                        captured_audio = np.concatenate(audio_buffer, axis=0).flatten()
                    done = True

        stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(self.SAMPLE_RATE * self.CHUNK_DURATION),
            callback=audio_callback,
        )

        stream.start()
        while not done:
            sd.sleep(100)
        stream.stop()
        stream.close()

        if self._stopped or captured_audio is None:
            return ""

        print("[Transcriber] Transcribing...")
        result = self.model.transcribe(captured_audio, language="en", fp16=True, verbose=None)
        text = result["text"].strip()
        return text

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if self.DEVICE == "mps":
                torch.mps.empty_cache()
            print("[Transcriber] Model cleaned up.")
