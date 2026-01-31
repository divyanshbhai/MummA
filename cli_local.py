import sounddevice as sd
import numpy as np
from fastrtc import get_stt_model, get_tts_model
from ollama import chat
import threading
import queue
import time
import sys

# ----------------------------
# 1. Configuration (M1 Optimized)
# ----------------------------
FS = 16000
# Threshold ko thoda badha diya hai taaki background hum ignore ho
MANUAL_THRESHOLD = 0.035 
SILENCE_LIMIT = 0.8
MAX_SPEECH_DURATION = 20.0 # Ek baar mein 10s se zyada record nahi karega

print("INFO: Initializing ManoMitra Engines...")
stt_model = get_stt_model()
tts_model = get_tts_model()
audio_queue = queue.Queue()
is_speaking = threading.Event()

# ----------------------------
# 2. Audio Player Thread
# ----------------------------
def player():
    while True:
        item = audio_queue.get()
        if item is None: break
        is_speaking.set()
        try:
            sd.play(np.array(item[1], dtype=np.float32), item[0])
            sd.wait()
        except: pass
        audio_queue.task_done()
        if audio_queue.empty():
            time.sleep(0.5) 
            is_speaking.clear()

threading.Thread(target=player, daemon=True).start()

# ----------------------------
# 3. Secure Voice Capture
# ----------------------------
def get_voice_input():
    print(f"\r🟢 Ready (Threshold: {MANUAL_THRESHOLD})", end="")
    chunks = []
    has_spoken = False
    silence_start = None
    recording_start_time = None
    
    def callback(indata, frames, time_info, status):
        nonlocal has_spoken, silence_start, recording_start_time
        # RMS volume calculation
        volume = np.sqrt(np.mean(indata**2))
        chunks.append(indata.copy())

        if volume > MANUAL_THRESHOLD:
            if not has_spoken:
                has_spoken = True
                recording_start_time = time.time()
            silence_start = None # Reset silence timer
        elif has_spoken:
            if silence_start is None:
                silence_start = time.time()

        # Meter UI
        bar = int(min(volume * 200, 30))
        indicator = "🔴" if has_spoken else "🟢"
        sys.stdout.write(f"\r{indicator} Vol: {volume:.4f} [{'|'*bar}{' '*(30-bar)}] ")
        sys.stdout.flush()

    try:
        with sd.InputStream(samplerate=FS, channels=1, callback=callback, blocksize=1024):
            start_loop = time.time()
            while True:
                now = time.time()
                
                # Agar 15 seconds tak kuch nahi bola toh reset
                if not has_spoken and (now - start_loop) > 15:
                    return None
                
                # Agar bolna shuru kiya
                if has_spoken:
                    # Case A: User chup ho gaya
                    if silence_start and (now - silence_start) > SILENCE_LIMIT:
                        break
                    # Case B: Safety Timeout (Bahut lamba bol raha hai)
                    if (now - recording_start_time) > MAX_SPEECH_DURATION:
                        break
                time.sleep(0.05)
    except Exception as e:
        print(f"\n[Mic Error]: {e}")
        return None

    print("\n[Processing Text...]")
    return np.concatenate(chunks).flatten()

# ----------------------------
# 4. Main Process
# ----------------------------
def main():
    last_reply = ""
    while True:
        try:
            if is_speaking.is_set():
                time.sleep(0.1)
                continue

            audio_data = get_voice_input()
            if audio_data is None: continue

            # STT call with error handling
            try:
                transcript = stt_model.stt((FS, audio_data)).strip()
            except Exception as stt_err:
                print(f"STT Hang or Error: {stt_err}")
                continue

            if not transcript or len(transcript) < 2:
                continue

            # Skip duplicate/echo
            if last_reply.lower() in transcript.lower() and len(transcript) < len(last_reply) + 10:
                continue

            print(f"\n👤: {transcript}")
            print("🤖: ", end="", flush=True)

            # LLM Logic
            response = chat(
                model="gemma3:1b",
                messages=[
                    {"role": "system", "content": "You are ManoMitra. Be a friendly AI. Reply in short Hinglish sentences."},
                    {"role": "user", "content": transcript}
                ],
                stream=True
            )

            full_text = ""
            sentence = ""
            for chunk in response:
                word = chunk['message']['content']
                print(word, end="", flush=True)
                full_text += word
                sentence += word

                if any(p in word for p in [".", "?", "!", "\n"]):
                    if sentence.strip():
                        audio_gen = tts_model.stream_tts_sync(sentence.strip())
                        for sr, wav in audio_gen:
                            audio_queue.put((sr, wav))
                    sentence = ""

            if sentence.strip():
                audio_gen = tts_model.stream_tts_sync(sentence.strip())
                for sr, wav in audio_gen:
                    audio_queue.put((sr, wav))
            
            last_reply = full_text

        except KeyboardInterrupt:
            print("\n👋 Stopping ManoMitra...")
            break
        except Exception as e:
            print(f"\nSystem Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()