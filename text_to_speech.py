from My_tts import MyTTS

text = "To be, or not to be, that is the question; whether ’tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles and by opposing end them."

tts = MyTTS("joe-medium", 1.1, 0.3)
# print(tts.list_voices())
# tts.process_play(text)
tts.process_save(text, "hello.wav")
