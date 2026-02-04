![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Data Source](https://img.shields.io/badge/dataset-Google_FLEURS-yellow)
![Model](https://img.shields.io/badge/model-Wav2Vec2_XLSR-red)

# Acoustic vs. Symbolic: Benchmarking Linguistic Distance in Self-Supervised Audio Embeddings

### 🔭 Research Abstract
This project bridges the gap between **Labor Economics** and **Audio Deep Learning** by investigating a fundamental question: *Does the "acoustic distance" between languages learned by neural networks correlate with the "symbolic distance" used in economic literature?*

In quantitative economics, migration and labor market integration studies typically rely on text-based proxies (e.g., Levenshtein Distance/ASJP) to quantify the difficulty of learning a new language. This repository implements a computational pipeline to benchmark these traditional metrics against **Latent Acoustic Distance**—a data-driven metric derived from the embedding space of **Meta AI's Wav2Vec 2.0 (XLSR-53)**.

### 🛠️ Technical Pipeline
The project utilizes a modular Python pipeline to transform raw audio into semiotic analysis:

1. **Data Engineering**: Automated ETL pipeline streaming **Google FLEURS** (16kHz) to harvest balanced samples across Romance, Germanic, and Uralic language families.
2. **Representation Learning**: Utilization of **Wav2Vec 2.0 (XLSR-53)** as a feature extractor, processing the model's last hidden states via Mean Pooling.
3. **Analysis**: Multidimensional scaling and manifold visualization (t-SNE) combined with **Pearson Correlation** ($r$) against ASJP ground truth.

### 🔬 Key Findings
Using a dataset of **600 audio samples** (N=100 per language), the experiment yielded a significant dissociation between local and global representations:

1. **High Local Separability**: 
  * The model achieved **76% accuracy** in distinguishing **Hungarian** (a Uralic isolate) from Indo-European languages.
  * **Catalan** showed high distinctiveness (**81% accuracy**), validating its phonological identity despite its lexical proximity to Spanish.
2. **The "Romance Continuum"**: As predicted by linguistic theory, the model exhibited high acoustic overlap between **Spanish, Catalan, and Italian**, mirroring the **Dialect Continuum** found in Romance linguistics.
3. **Manifold Geometry**: While local neighborhoods are semantically meaningful (High KNN accuracy), global linear correlations were weak ($r \approx -0.21$), suggesting the latent space is **topologically consistent** but **globally anisotropic**.

![Linguistic Confusion Matrix](Figures/Confusion-Matrix.png)

### 💻 Installation & Usage

**1. Clone the Repository**
```bash
git clone https://github.com/Ogajah1/acoustic-symbolic-alignment.git
cd acoustic-symbolic-alignment
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Pipeline**

Step A: Harvest Data (Streaming ingestion from Hugging Face)
```bash
python data_harvester.py
```

Step B: Run Analysis (Feature extraction and visualization)
```bash
python linguistic_semiotics.py
```

### 📁 Repository Structure
* `data_harvester.py`: Robust N=100 streaming ingestion script.
* `linguistic_semiotics.py`: Analysis engine with Anisotropy Correction.
* `Confusion-Matrix.png`: Visualization of semiotic overlap.
* `Geometric-landscape.png`: t-SNE projection of the latent space.

### ⚠️ Limitations
While the confusion matrix demonstrates strong local separability, several constraints frame the interpretation of these results:

1. **Representation Anisotropy**: The weak linear correlation ($r \approx -0.21$) between Cosine Distance and Levenshtein Distance indicates that the model's latent space is highly anisotropic (the "Representation Cone" effect). While local neighborhoods are semantically meaningful (high KNN accuracy), global distances are distorted by the common dominant vector of human speech.
2. **Dataset Domain**: This study utilizes **Google FLEURS**, which consists of read speech. The clear articulation in read speech may inflate the model's performance compared to spontaneous, noisy speech found in real-world economic scenarios.
3. **Sample Size**: With $N=600$ (100 clips per language), the study is statistically significant for exploratory analysis but effectively a "low-resource" scenario in the context of Deep Learning.

### 🚀 Future Research Directions

**1. Layer-Wise Probing of Linguistic Hierarchy**
Investigate the "Depth of Representation" by extracting embeddings from intermediate Transformer layers. The hypothesis is that early layers (1-12) cluster by **phonetic similarity** (acoustic surface form), while deeper layers (24+) cluster by **typological family** (prosodic/syntactic structure).

**2. Disentanglement Analysis (Speaker vs. Language)**
Apply t-SNE visualization colored by `Speaker_ID` rather than `Language_ID` to quantify the model's invariance to speaker identity. This measures the disentanglement of **linguistic content** from **paralinguistic features**. 

**3. Economic Gravity Model Integration**
Utilize the derived "Acoustic Distance" as a novel instrumental variable in **Gravity Models of Trade**, testing if acoustic friction predicts labor market integration better than traditional text-based proxies.

### 📝 Citation
If you find this analysis useful, please cite this repository:

```bibtex
@misc{acoustic-symbolic-alignment,
  author = {Thompson, O.},
  title = {Acoustic vs. Symbolic: Benchmarking Linguistic Distance in Self-Supervised Audio Embeddings},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Ogajah1/acoustic-symbolic-alignment}}
}
```

---

*Developed at the intersection of Econometrics and Audio Machine Learning.*
