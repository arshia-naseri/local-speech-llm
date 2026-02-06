import subprocess
from typing import Literal

VoiceName = Literal["joe-medium", "lessac-high"]


class MyTTS:
    __VOICE_DIR__ = "./voices"
    __VOICES__: dict[VoiceName, str] = {
        "joe-medium": f"{__VOICE_DIR__}/en_US-joe-medium.onnx",
        "lessac-high": f"{__VOICE_DIR__}/en_US-lessac-high.onnx",
    }
    __System__ = "mac"

    @classmethod
    def list_voices(cls):
        return list(cls.__VOICES__.keys())

    def __init__(
        self, voiceName: VoiceName, slowness: float = 1, sentence_silence: float = 0
    ):
        self.voice = self.__VOICES__[voiceName]
        self.slowness = slowness
        self.sentence_silence = sentence_silence

    def process_play(self, text: str):
        piper_proc = subprocess.Popen(
            [
                "piper",
                "--model",
                self.voice,
                "--output-raw",
                "--length-scale",
                f"{self.slowness:.1f}",
                "--sentence-silence",
                f"{self.sentence_silence:.1f}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        play_proc = subprocess.Popen(
            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-f",
                "s16le",  # raw 16-bit signed little-endian
                "-ar",
                "22050",  # piper's default sample rate (check your model)
                "-i",
                "pipe:0",
            ],
            stdin=piper_proc.stdout,
        )

        piper_proc.stdin.write(text.encode("utf-8"))
        piper_proc.stdin.close()
        play_proc.wait()

    def process_save(self, text: str, output_file: str, play: bool = False):
        piper_proc = subprocess.Popen(
            [
                "piper",
                "--model",
                self.voice,
                "--output_file",
                output_file,
                "--length-scale",
                f"{self.slowness:.1f}",
                "--sentence-silence",
                f"{self.sentence_silence:.1f}",
            ],
            stdin=subprocess.PIPE,
        )

        piper_proc.stdin.write(text.encode("utf-8"))
        piper_proc.stdin.close()
        piper_proc.wait()

        if play:
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", output_file]
            )
