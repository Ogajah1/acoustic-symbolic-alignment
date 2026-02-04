import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import numpy as np
import os
import glob
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
TARGET_SAMPLE_RATE = 16000 
USE_REAL_FILES = True 

# --- THEORETICAL GROUND TRUTH (ASJP/LDND Distance) ---
# Source: ASJP Database (Automated Similarity Judgment Program)
# Metric: LDND (Levenshtein Distance Normalized Divided)
# Range: 0 (Identical) to ~100 (Completely Unrelated)
THEORETICAL_DISTANCES = {
    # ROMANCE INTERNAL (Very Close)
    ('Spanish', 'Catalan'): 23.4,  
    ('Spanish', 'Italian'): 26.8,  
    ('Catalan', 'Italian'): 25.1,

    # ROMANCE VS GERMANIC (Distant)
    ('Spanish', 'English'): 92.1,  
    ('Spanish', 'German'):  94.5,
    ('Catalan', 'English'): 91.8,
    ('Italian', 'German'):  93.2,

    # GERMANIC INTERNAL (Medium Close)
    ('English', 'German'):  58.7,  

    # HUNGARIAN (The Outlier - Uralic Family)
    # Hungarian is linguistically distant from ALL Indo-European languages
    ('Spanish', 'Hungarian'): 98.2, 
    ('English', 'Hungarian'): 99.1,
    ('German', 'Hungarian'):  97.5,
    ('Italian', 'Hungarian'): 98.0
}

class SemioticAlignmentAnalyzer:
    def __init__(self):
        print(f"Loading Model: {MODEL_NAME}...")
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
            self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
            self.model.eval()
            print("Model loaded successfully.")
        except OSError:
            print("[Error] Could not load model. Check internet connection.")

    def preprocess_audio(self, file_path):
        """
        ROBUST LOADING: Uses soundfile directly to bypass Windows torchaudio bugs.
        """
        # 1. Read file with soundfile
        audio_array, sample_rate = sf.read(file_path)
        
        # 2. Convert to Torch Tensor
        waveform = torch.from_numpy(audio_array).float()
        
        # 3. Standardize dimensions
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0) 
        elif waveform.ndim == 2: waveform = waveform.t() 

        # 4. Resample
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # 5. Mono conversion
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform.squeeze()

    def get_embedding(self, audio_tensor, layer_index=-1):
        """
        Extracts embedding from a specific transformer layer.
        layer_index=-1 (Last layer, most semantic/abstract)
        """
        inputs = self.feature_extractor(
            audio_tensor, 
            sampling_rate=TARGET_SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Select specific layer
        hidden_states = outputs.hidden_states[layer_index]
        
        # Mean Pooling
        embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()
        return embedding

    def calculate_scientific_correlation(self, embeddings, labels):
        """
        Calculates Pearson r between 'Acoustic Distance' (Model) 
        and 'Theoretical Distance' (ASJP Linguistics).
        
        CRITICAL FIX: Centers the data (Anisotropy Correction) before 
        calculating Cosine Distance.
        """
        # 1. CENTER THE DATA (Remove the common "Speech" vector)
        global_mean = np.mean(embeddings, axis=0)
        centered_embeddings = embeddings - global_mean
        
        unique_langs = np.unique(labels)
        centroids = {}
        # Calculate centroids using the CENTERED embeddings
        for lang in unique_langs:
            idxs = [i for i, l in enumerate(labels) if l == lang]
            centroids[lang] = np.mean(centered_embeddings[idxs], axis=0)

        model_distances = []
        theory_distances = []
        
        print("\n--- Correlation Study: Acoustic vs. Symbolic (Centered) ---")
        print(f"{'Language Pair':<25} | {'Model (Cos)':<12} | {'ASJP (LDND)':<12}")
        print("-" * 55)

        for l1, l2 in THEORETICAL_DISTANCES.keys():
            if l1 in centroids and l2 in centroids:
                # Calculate Model Distance (Cosine Distance = 1 - Similarity)
                vec_a = centroids[l1].reshape(1, -1)
                vec_b = centroids[l2].reshape(1, -1)
                acoustic_dist = cosine_distances(vec_a, vec_b)[0][0]
                
                theory_dist = THEORETICAL_DISTANCES[(l1, l2)]
                
                model_distances.append(acoustic_dist)
                theory_distances.append(theory_dist)
                
                print(f"{l1}-{l2:<16} | {acoustic_dist:.4f}       | {theory_dist:.1f}")

        if len(model_distances) > 2:
            r, p_value = pearsonr(model_distances, theory_distances)
            print("-" * 55)
            print(f"Pearson Correlation (r): {r:.4f}")
            print(f"P-Value: {p_value:.4f}")
            print("-" * 55)
            if r > 0.6:
                print("✅ RESULT: Strong scientific validation (Anisotropy Corrected).")
            elif r > 0.4:
                print("⚠️ RESULT: Moderate alignment. (Common in self-supervised models).")
            else:
                print("❌ RESULT: Weak alignment.")
        else:
            print("[Warning] Not enough matching pairs for correlation.")

    def plot_confusion(self, embeddings, labels):
        print("\nComputing Confusion Matrix...")
        # Increased K for larger N
        clf = KNeighborsClassifier(n_neighbors=5) 
        clf.fit(embeddings, labels)
        y_pred = clf.predict(embeddings)
        
        cm = confusion_matrix(labels, y_pred, labels=np.unique(labels))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
        plt.title(f"Linguistic Confusion Matrix (N={len(labels)})")
        plt.tight_layout()
        plt.show()

    def plot_latent_space(self, embeddings, labels):
        print("\nComputing t-SNE Visualization...")
        perp = min(50, len(labels)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        reduced_vecs = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=reduced_vecs[:,0], y=reduced_vecs[:,1], hue=labels, style=labels, s=60, alpha=0.8)
        plt.title(f"The Geometry of Language: Wav2Vec 2.0 Latent Space")
        plt.xlabel("Latent Dim 1")
        plt.ylabel("Latent Dim 2")
        plt.legend(title="Language", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyzer = SemioticAlignmentAnalyzer()
    
    if USE_REAL_FILES:
        embeddings = []
        labels = []
        
        # 6-Language Setup (Matching the N=100 Harvester)
        path_map = {
            'Spanish': 'data/es/*.wav',
            'Catalan': 'data/ca/*.wav',
            'Italian': 'data/it/*.wav',
            'English': 'data/en/*.wav',
            'German': 'data/de/*.wav',
            'Hungarian': 'data/hu/*.wav'
        }
        
        for lang_name, path_pattern in path_map.items():
            files = glob.glob(path_pattern)[:100] # Cap at 100 per language
            if not files:
                print(f"[Warning] No files found for {lang_name}. Check 'data' folder.")
                continue
                
            print(f"Processing {lang_name} ({len(files)} clips)...")
            for f in files:
                try:
                    wf = analyzer.preprocess_audio(f)
                    # Extracting from the final layer (-1)
                    emb = analyzer.get_embedding(wf, layer_index=-1) 
                    embeddings.append(emb)
                    labels.append(lang_name)
                except Exception as e:
                    pass
        
        if len(embeddings) > 0:
            embeddings = np.array(embeddings)
            analyzer.calculate_scientific_correlation(embeddings, labels)
            analyzer.plot_confusion(embeddings, labels)
            analyzer.plot_latent_space(embeddings, labels)
        else:
            print("[Error] No embeddings generated. Run data_harvester.py first.")