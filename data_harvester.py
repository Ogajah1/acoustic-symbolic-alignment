import os
import soundfile as sf
from datasets import load_dataset

# --- CONFIGURATION ---
DATASET_NAME = "google/fleurs"

# FINAL TARGET LIST (6 Languages)
# Dropped Portuguese to maintain European consistency.
TARGET_LANGS = {
    "es_419": "Spanish",    # LatAm Spanish (Close proxy for ES)
    "ca_es":  "Catalan",
    "en_us":  "English",
    "hu_hu":  "Hungarian",  # The Isolate Control
    "it_it":  "Italian",
    "de_de":  "German"
}

SAMPLES_PER_LANG = 30
OUTPUT_DIR = "data"

def harvest_audio():
    print(f"--- STARTING HARVEST ({len(TARGET_LANGS)} Languages) ---")
    
    for lang_code, lang_name in TARGET_LANGS.items():
        print(f"\n[Stream] Connecting to Google FLEURS ({lang_name} - {lang_code})...")
        
        try:
            # Load Dataset (Streaming)
            ds = load_dataset(
                DATASET_NAME, 
                lang_code, 
                split="train", 
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"   [Error] Could not connect to {lang_name}: {e}")
            continue

        # Create Directory
        simple_code = lang_code.split("_")[0] 
        lang_dir = os.path.join(OUTPUT_DIR, simple_code)
        os.makedirs(lang_dir, exist_ok=True)

        print(f"   -> Downloading {SAMPLES_PER_LANG} clips to '{lang_dir}'...")
        count = 0
        
        try:
            for sample in ds:
                if count >= SAMPLES_PER_LANG:
                    break
                
                audio_array = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]
                
                filename = os.path.join(lang_dir, f"{simple_code}_clip_{count}.wav")
                sf.write(filename, audio_array, sample_rate)
                
                if count % 10 == 0 and count > 0:
                    print(f"      ...saved {count} clips")
                count += 1
                
            print(f"   [Done] Successfully saved {count} clips for {lang_name}")
            
        except Exception as e:
             print(f"   [Error] Failed while downloading {lang_name}: {e}")

if __name__ == "__main__":
    harvest_audio()