import numpy as np
import sounddevice as sd

def record_audio(duration, fs, channels):
    return sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float32')

def play_audio(audio_data, fs):
    sd.play(audio_data, fs)
    sd.wait()

def chunk_generator(buffer, chunk_size):
    for i in range(0, len(buffer), chunk_size):
        yield buffer[i:i + chunk_size]