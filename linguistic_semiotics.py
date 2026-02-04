import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import numpy as np
import os
import glob
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

class SemioticAlignmentAnalyzer:
    def __init__(self):
        print(f"Loading Model: {MODEL_NAME}...")
        try:
            # Using FeatureExtractor to avoid missing vocab.json errors
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
            self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
            self.model.eval()
            print("Model loaded successfully.")
        except OSError:
            print("[Error] Could not load model. Check your internet connection.")

    def preprocess_audio(self, file_path):
        """
        ROBUST LOADING: Uses soundfile directly to avoid Windows-specific torchaudio 
        decoding bugs (TorchCodec).
        """
        # 1. Read file with soundfile
        audio_array, sample_rate = sf.read(file_path)
        
        # 2. Convert to Torch Tensor
        waveform = torch.from_numpy(audio_array).float()
        
        # 3. Standardize dimensions to (Channels, Time)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) 
        elif waveform.ndim == 2:
            waveform = waveform.t() 

        # 4. Resample to 16kHz for Wav2Vec2
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # 5. Downmix to Mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform.squeeze()

    def get_embedding(self, audio_tensor):
        """Extracts the contextual latent representation from the model's last hidden state."""
        inputs = self.feature_extractor(
            audio_tensor, 
            sampling_rate=TARGET_SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean Pooling: Collapses the sequence length dimension into a single vector
        last_hidden_states = outputs.last_hidden_state
        embedding = torch.mean(last_hidden_states, dim=1).squeeze().numpy()
        return embedding

    def analyze_semiotic_distance(self, embeddings, labels):
        """Quantifies the 'Acoustic Distance' between linguistic centroids."""
        unique_langs = np.unique(labels)
        centroids = {}
        for lang in unique_langs:
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            centroids[lang] = np.mean(embeddings[lang_indices], axis=0)
            
        print("\n--- Latent Acoustic Distances (1 - Cosine Similarity) ---")
        if 'Spanish' in centroids and 'Catalan' in centroids:
            dist = cosine_distances(centroids['Spanish'].reshape(1,-1), centroids['Catalan'].reshape(1,-1))[0][0]
            print(f"Linguistic Target Check: Spanish <-> Catalan Distance: {dist:.4f}")

        print("\nFull Pairwise Matrix:")
        for i in range(len(unique_langs)):
            for j in range(i + 1, len(unique_langs)):
                l_a, l_b = unique_langs[i], unique_langs[j]
                d = cosine_distances(centroids[l_a].reshape(1,-1), centroids[l_b].reshape(1,-1))[0][0]
                print(f"   {l_a} <-> {l_b}: {d:.4f}")

    def plot_linguistic_confusion(self, embeddings, labels):
        """Visualizes how often the model 'confuses' similar linguistic signifiers."""
        print("\nGenerating Confusion Matrix...")
        X, y = embeddings, labels
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        
        unique_labels = np.unique(y)
        cm = confusion_matrix(y, y_pred, labels=unique_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
        plt.title("Semiotic Confusion: Do linguistically close languages overlap?")
        plt.tight_layout()
        plt.show()

    def plot_latent_space(self, embeddings, labels):
        """Projects high-dimensional embeddings into 2D for cluster analysis."""
        print("\nComputing t-SNE projection...")
        perp = min(30, len(labels)-1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        reduced_vecs = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=reduced_vecs[:,0], y=reduced_vecs[:,1], hue=labels, style=labels, s=100)
        plt.title("The Geometry of Language: Wav2Vec 2.0 Latent Space")
        plt.xlabel("Latent Component 1")
        plt.ylabel("Latent Component 2")
        plt.legend(title="Language", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyzer = SemioticAlignmentAnalyzer()
    
    if USE_REAL_FILES:
        embeddings, labels = [], []
        
        # Consistent mapping for the 6-language European experiment
        path_map = {
            'Spanish': 'data/es/*.wav',
            'Catalan': 'data/ca/*.wav',
            'Italian': 'data/it/*.wav',
            'English': 'data/en/*.wav',
            'German': 'data/de/*.wav',
            'Hungarian': 'data/hu/*.wav'
        }
        
        for lang_name, path_pattern in path_map.items():
            files = glob.glob(path_pattern)[:30]
            if not files:
                print(f"[Warning] No files found for {lang_name}. Check your 'data' folder.")
                continue
                
            print(f"Processing {lang_name} ({len(files)} clips)...")
            for f in files:
                try:
                    wf = analyzer.preprocess_audio(f)
                    emb = analyzer.get_embedding(wf)
                    embeddings.append(emb)
                    labels.append(lang_name)
                except Exception as e:
                    print(f"Skipped {f}: {e}")
        
        if len(embeddings) > 0:
            embeddings = np.array(embeddings)
            analyzer.analyze_semiotic_distance(embeddings, labels)
            analyzer.plot_linguistic_confusion(embeddings, labels)
            analyzer.plot_latent_space(embeddings, labels)
        else:
            print("[Error] Empty dataset. Run data_harvester.py first.")