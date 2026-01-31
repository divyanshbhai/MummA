import sounddevice as sd
import numpy as np
from fastrtc import get_stt_model, get_tts_model
from ollama import chat

# ----------------------------
# Initialize models
# ----------------------------
print("INFO: Warming up STT model...")
stt_model = get_stt_model()  # offline STT
tts_model = get_tts_model()  # offline TTS
print("INFO: Models warmed up.")

# ----------------------------
# Record audio
# ----------------------------
def record(duration=5, fs=16000):
    print("🟢 Listening...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return fs, audio.flatten()

# ----------------------------
# Play audio
# ----------------------------
def play_audio(chunks):
    for sr, chunk in chunks:
        audio = np.array(chunk, dtype=np.float32)
        sd.play(audio, sr)
        sd.wait()

# ----------------------------
# Main loop
# ----------------------------
def main():
    last_ai_reply = ""
    while True:
        try:
            sr, audio = record(duration=4)  # listen for 4 seconds
            transcript = stt_model.stt((sr, audio)).strip()
            if not transcript:
                continue
            # Avoid AI listening to itself
            if last_ai_reply and transcript.lower() in last_ai_reply:
                continue

            print(f"👤: {transcript}")

            try:
                response = chat(
                    model="qwen:0.5b",
                    messages=[{"role": "user", "content": transcript}]
                )
                reply = response["message"]["content"].strip()
            except Exception:
                reply = "Oops! I couldn't understand that."

            last_ai_reply = reply.lower()
            print(f"🤖: {reply}")

            # Play TTS
            play_audio(tts_model.stream_tts_sync(reply))

        except KeyboardInterrupt:
            print("\n👋 Exiting... Goodbye!")
            break

if __name__ == "__main__":
    main()