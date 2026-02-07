import tkinter as tk
import threading
from My_LLM import MyLlm
from My_tts import MyTTS
from My_transcriber import MyTranscriber


class VoiceAssistant:
    def __init__(self):
        self.transcriber = None
        self.llm = None
        self.tts = None
        self.is_running = False
        self._stop_requested = False

        self.root = tk.Tk()
        self.root.title("Voice Assistant")
        self.root.geometry("300x300")
        self.root.resizable(False, False)

        # Init button at top
        self.init_btn = tk.Button(
            self.root, text="Initialize", command=self._on_init, width=20, height=2
        )
        self.init_btn.pack(pady=(20, 10))

        # Primary listen button in centre
        self.listen_btn = tk.Button(
            self.root,
            text="🎤 Talk",
            command=self._on_listen,
            width=20,
            height=3,
            state=tk.DISABLED,
        )
        self.listen_btn.pack(pady=10)

        # Stop button
        self.stop_btn = tk.Button(
            self.root, text="⏹ Stop", command=self._on_stop, width=20, height=2,
            state=tk.DISABLED, fg="red",
        )
        self.stop_btn.pack(pady=5)

        # Status label
        self.status_var = tk.StringVar(value="Not initialized")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var, font=("Helvetica", 12)
        )
        self.status_label.pack(pady=10)

    def _set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _on_init(self):
        self.init_btn.config(state=tk.DISABLED, text="Initializing...")
        self._set_status("Loading models...")

        def do_init():
            self.transcriber = MyTranscriber()
            self.transcriber.init_model()

            self.llm = MyLlm()
            self.tts = MyTTS("joe-medium")

            self.root.after(0, self._init_done)

        threading.Thread(target=do_init, daemon=True).start()

    def _init_done(self):
        self.init_btn.config(text="Initialized ✓")
        self.listen_btn.config(state=tk.NORMAL)
        self._set_status("Ready")
        print("[System] All models initialized.")

    def _on_listen(self):
        if self.is_running:
            return
        self.is_running = True
        self._stop_requested = False
        self.listen_btn.config(state=tk.DISABLED, text="Listening...")
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("Listening...")

        threading.Thread(target=self._pipeline, daemon=True).start()

    def _on_stop(self):
        self._stop_requested = True
        self.stop_btn.config(state=tk.DISABLED)
        self._set_status("Stopping...")
        print("[System] Stop requested.")

        # Stop listening
        if self.transcriber:
            self.transcriber.stop()
        # Stop speaking
        if self.tts:
            self.tts.stop()

    def _pipeline(self):
        try:
            # 1. Listen and transcribe
            text = self.transcriber.listen_and_transcribe()
            if self._stop_requested or not text:
                if not self._stop_requested:
                    print("[System] No speech detected.")
                self.root.after(0, self._pipeline_done)
                return

            print(f"\nYou: {text}")
            self.root.after(0, lambda: self._set_status("Thinking..."))

            # 2. Send to LLM
            if self._stop_requested:
                self.root.after(0, self._pipeline_done)
                return
            response = self.llm.chat(text, print_output=False)

            if self._stop_requested:
                self.root.after(0, self._pipeline_done)
                return
            print(f"LLM: {response}")
            self.root.after(0, lambda: self._set_status("Speaking..."))

            # 3. Speak the response
            self.tts.process_play(response)

        except Exception as e:
            if not self._stop_requested:
                print(f"[Error] {e}")
        finally:
            self.root.after(0, self._pipeline_done)

    def _pipeline_done(self):
        self.is_running = False
        self._stop_requested = False
        self.listen_btn.config(state=tk.NORMAL, text="🎤 Talk")
        self.stop_btn.config(state=tk.DISABLED)
        self._set_status("Ready")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceAssistant()
    app.run()
