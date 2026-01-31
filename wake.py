import sounddevice as sd
import numpy as np
from fastrtc import get_stt_model, get_tts_model
from ollama import chat
import threading
import queue
import time
import sys
from collections import deque

# ----------------------------
# 1. Configuration
# ----------------------------
FS = 16000
WAKE_WORD = "mitra"  # Recommended: "Suno" or "Mitra"
# Rolling buffer size (1.5 seconds of audio)
BUFFER_SIZE = int(FS * 1.5) 
audio_buffer = deque(maxlen=BUFFER_SIZE)

print("INFO: Initializing High-Speed Wake Engine...")
stt_model = get_stt_model()
tts_model = get_tts_model()
audio_queue = queue.Queue()
is_speaking = threading.Event()
wake_detected = threading.Event()

# ----------------------------
# 2. Background Audio Player
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
# 3. Continuous Rolling Listener
# ----------------------------
def stream_callback(indata, frames, time_info, status):
    """Continuously feeds the rolling buffer."""
    audio_buffer.extend(indata.flatten())

def wake_word_monitor():
    """Background thread that scans the rolling buffer for the wake word."""
    print(f"☁️  Always Listening for '{WAKE_WORD}'...")
    while not wake_detected.is_set():
        if len(audio_buffer) < BUFFER_SIZE:
            time.sleep(0.1)
            continue
        
        # Take a snapshot of the current 1.5s buffer
        current_audio = np.array(list(audio_buffer))
        
        # Check volume - only run STT if there is actual sound
        volume = np.sqrt(np.mean(current_audio**2))
        if volume > 0.02:  # Threshold
            text = stt_model.stt((FS, current_audio)).lower().strip()
            if WAKE_WORD in text:
                wake_detected.set()
        
        # Scan every 0.5 seconds for maximum speed
        time.sleep(0.5)

# ----------------------------
# 4. Command Capture (Active Mode)
# ----------------------------
def get_command_vad():
    print(f"\n✨ Listening! Boliye...")
    # Beep logic can be added here
    chunks = []
    silence_start = None
    has_spoken = False
    
    def callback(indata, frames, time_info, status):
        nonlocal silence_start, has_spoken
        vol = np.sqrt(np.mean(indata**2))
        chunks.append(indata.copy())
        if vol > 0.03:
            has_spoken = True
            silence_start = None
        elif has_spoken:
            if silence_start is None: silence_start = time.time()

    with sd.InputStream(samplerate=FS, channels=1, callback=callback):
        while True:
            if has_spoken and silence_start and (time.time() - silence_start) > 0.8:
                break
            time.sleep(0.1)
    return np.concatenate(chunks).flatten()

# ----------------------------
# 5. Main Loop
# ----------------------------
def main():
    while True:
        # Start passive streaming
        wake_detected.clear()
        with sd.InputStream(samplerate=FS, channels=1, callback=stream_callback):
            monitor_thread = threading.Thread(target=wake_word_monitor, daemon=True)
            monitor_thread.start()
            
            # Wait until the background monitor finds the word
            while not wake_detected.is_set():
                time.sleep(0.1)
        
        # WAKE WORD DETECTED -> MOVE TO ACTIVE MODE
        cmd_audio = get_command_vad()
        if cmd_audio is not None:
            transcript = stt_model.stt((FS, cmd_audio)).strip()
            if transcript:
                print(f"👤: {transcript}")
                print("🤖: ", end="", flush=True)

                response = chat(
                    model="qwen:0.5b",
                    messages=[
                        {"role": "system", "content": "You are ManoMitra. Reply in short Hinglish sentences."},
                        {"role": "user", "content": transcript}
                    ],
                    stream=True
                )

                full_reply = ""
                sentence = ""
                for chunk in response:
                    word = chunk['message']['content']
                    print(word, end="", flush=True)
                    full_reply += word
                    sentence += word
                    if any(p in word for p in [".", "?", "!", "\n"]):
                        if sentence.strip():
                            tts_gen = tts_model.stream_tts_sync(sentence.strip())
                            for sr, wav in tts_gen: audio_queue.put((sr, wav))
                        sentence = ""
                
                # Final TTS flush
                if sentence.strip():
                    tts_gen = tts_model.stream_tts_sync(sentence.strip())
                    for sr, wav in tts_gen: audio_queue.put((sr, wav))
                
                # Wait for AI to finish speaking before going back to passive mode
                while is_speaking.is_set():
                    time.sleep(0.1)
                print(f"\n--- Back to Passive Mode ---")

if __name__ == "__main__":
    main()