# EEG Analysis Report: Insights for Model Training

**Dataset:** Physionet EEG Motor Movement/Imagery (S001–S109)  
**Objective:** Prepare data for feature extraction and machine learning.

---

## 1. Key Preprocessing Findings

**Raw Data Limitations**  
The original signals contained significant DC offset (drifting baselines) and high-frequency noise (sharp spikes). This noise stems from muscle activity (EMG) and power line interference (50/60 Hz).

**Filter Impact (1–40 Hz)**  
Applying a band-pass filter transformed *sharp/jagged* signals into *smooth/rounded* waves. This removes non-brain artifacts while preserving the physiological rhythms necessary for classification.

---

## 2. Feature Selection Strategy

The following **features** were identified as the most critical inputs for the algorithm:

### A. Frequency Domain (The “What”)

- **Alpha Band (8–13 Hz)**  
  Essential for detecting relaxation and *Event-Related Desynchronization* during motor tasks.

- **Beta Band (13–30 Hz)**  
  The primary indicator of active motor cortical engagement.

- **Metric**  
  Power Spectral Density (PSD) will be used to calculate the relative power of these bands per channel.

### B. Spatial Domain (The “Where”)

- **Motor Cortex Focus**  
  While the data includes 64 channels, the algorithm should prioritize the central electrodes **C3, Cz, and C4**. These are located over the motor strip, where motor imagery signals are strongest.

---

## 3. Conclusion for Pipeline

The input to the machine learning model will **not** be the raw voltage signal, but a **feature vector** consisting of:

- Mean power in the Alpha and Beta bands  
- Ratios between bands (e.g., Alpha/Beta ratio)  
- Channel selection focused on the motor cortex to reduce dimensionality and avoid overfitting on noise from frontal or occipital leads
