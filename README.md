# Acoustic vs. Symbolic: Benchmarking Linguistic Distance in Self-Supervised Audio Embeddings

### 🔭 Research Abstract
This project investigates the alignment between **Symbolic Linguistic Distance** (Levenshtein/ASJP) and **Latent Acoustic Distance** in state-of-the-art audio models. Using **Meta AI’s Wav2Vec 2.0**, I analyzed whether self-supervised models implicitly learn the "Romance-Germanic-Uralic" typological divide without supervision.

### 🔬 Key Findings
Using a dataset of **600 audio samples** (N=100 per language) across 6 languages, the experiment yielded a significant dissociation between local and global representations:

1.  **High Local Separability (The "Isolate" Hypothesis)**:
    * The model achieved **76% accuracy** in distinguishing **Hungarian** (a Uralic isolate) from Indo-European languages using a simple KNN classifier.
    * **Catalan** showed high distinctiveness (**81% accuracy**), validating its phonological identity despite its lexical proximity to Spanish.

2.  **The "Romance Continuum"**:
    * As predicted by linguistic theory, the model exhibited the highest "confusion" between **Spanish, Catalan, and Italian**. This acoustic overlap mirrors the **Dialect Continuum** found in Romance linguistics.

3.  **Manifold Geometry**:
    * While local clusters were distinct (High KNN Accuracy), global linear correlations with Levenshtein distance were weak. This suggests the Wav2Vec 2.0 latent space is **topologically consistent** (neighbors are correct) but **globally anisotropic** (directions are non-linear).

![Linguistic Confusion Matrix](Figures/Confusion-Matrix.png)

### 🛠️ Technical Pipeline
* **Data**: Automated ETL pipeline streaming **Google FLEURS** (16kHz).
* **Model**: Wav2Vec 2.0 (XLSR-53) Feature Extractor (Layer -1).
* **Metrics**: Dimensionality Reduction (t-SNE), Pearson Correlation, and KNN Classification.

### 📁 Repository Structure
* `data_harvester.py`: Robust N=100 streaming ingestion script.
* `linguistic_semiotics.py`: Analysis engine with Anisotropy Correction.
* `Confusion-Matrix.png`: Visualization of semiotic overlap.
* `Geometric-landscape.png`: t-SNE projection of the latent space.

---
*A research study bridging Labor Economics (Linguistic distances, Migration Costs, etc.) and Audio Deep Learning.*