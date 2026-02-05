import os
import soundfile as sf
from datasets import load_dataset

# --- CONFIGURATION ---
DATASET_NAME = "google/fleurs"

# 6 Languages (Romance, Germanic, Isolate)
TARGET_LANGS = {
    "es_419": "Spanish",   
    "ca_es":  "Catalan",
    "en_us":  "English",
    "hu_hu":  "Hungarian",  # The Control (Uralic)
    "it_it":  "Italian",
    "de_de":  "German"
}

# of audio samples to harvest per language 
SAMPLES_PER_LANG = 100
OUTPUT_DIR = "data"

def harvest_audio():
    print(f"--- STARTING ROBUST HARVEST (N={SAMPLES_PER_LANG} per language) ---")
    
    for lang_code, lang_name in TARGET_LANGS.items():
        print(f"\n[Stream] Connecting to {lang_name} ({lang_code})...")
        
        try:
            ds = load_dataset(
                DATASET_NAME, 
                lang_code, 
                split="train", 
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"   [Error] Connection failed: {e}")
            continue

        # Setup Directory
        simple_code = lang_code.split("_")[0] 
        # Fix for Hungarian code consistency
        if simple_code == "hu": simple_code = "hu"
        
        lang_dir = os.path.join(OUTPUT_DIR, simple_code)
        os.makedirs(lang_dir, exist_ok=True)

        print(f"   -> Downloading clips to '{lang_dir}'...")
        count = 0
        
        try:
            for sample in ds:
                if count >= SAMPLES_PER_LANG:
                    break
                
                audio_array = sample["audio"]["array"]
                sample_rate = sample["audio"]["sampling_rate"]
                
                filename = os.path.join(lang_dir, f"{simple_code}_clip_{count}.wav")
                sf.write(filename, audio_array, sample_rate)
                
                # Log progress
                if count % 20 == 0 and count > 0:
                    print(f"      ...saved {count}/{SAMPLES_PER_LANG}")
                count += 1
                
            print(f"   [Done] {lang_name} complete.")
            
        except Exception as e:
             print(f"   [Error] Interrupted: {e}")

if __name__ == "__main__":
    harvest_audio()