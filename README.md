# A neuromorphic approach to secure data processing is achieved through optimised spiking neural network encoding and decoding

This repository contains the official codebase accompanying the paper:

> **Privacy-Preserving Spiking Neural Networks: A Deep Dive into Encryption Parameter Optimisation**  
> *Mahitha Pulivathi, Ana Fontes Rodrigues, Isibor Kennedy Ihianle, Andreas Oikonomou,  
> Srinivas Boppu, Pedro Machado (2025)*  
> **arXiv:** https://arxiv.org/abs/2510.19537

BioEncryptSNN introduces a biologically inspired, spike-based encryption and decryption framework leveraging **Spiking Neural Networks (SNNs)**.  
The system converts classical ciphertext into spike trains and uses temporal neural dynamics to perform secure transformations.  

The framework provides:

- **Privacy-preserving computation** enabled by neuromorphic spike-based processing.  
- **Encryption parameter optimisation**, including:  
  - key lengths  
  - spike timing windows  
  - synaptic connectivity regimes  
- **Benchmarking against classical cryptographic algorithms** such as AES-128, RSA-2048, and DES.  
- **Latency and throughput improvements** when compared with conventional software encryption pipelines.  
- **Noise and perturbation robustness analysis** in the spike domain.  

This repository contains the reference implementation, experiments, evaluation scripts, and configuration files used in the paper.

---

## 🔧 Repository Structure

configs/ # Experiment configs (SNN architectures, crypto parameters)
models/ # Spiking neural network architectures and neuron models
enc_schemes/ # Classical encryption algorithms (AES, RSA, DES)
data/ # Data generation, formatting, and encoding scripts
train/ # Training routines for BioEncryptSNN
eval/ # Evaluation scripts for latency, throughput, robustness
notebooks/ # Jupyter notebooks for visualisation & reproduction of figures
utils/ # Logging, plotting, spike encoders, helper functions

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Data Preparation

You may either generate synthetic plaintext–ciphertext datasets or load existing ones.

### Option A — Generate synthetic encryption pairs

Example for AES-128:

```bash
python data/generate_pairs.py     --scheme AES     --key-length 128     --num-samples 10000     --output data/aes_128_pairs.npz
```

Common parameters include:

    --scheme: AES, RSA, or DES

    --key-length: numerical key size

    --num-samples: number of samples

    --noise-level: optional artificial noise injection

### Option B — Use pre-generated datasets

Modify the relevant configuration file:

data:
  dataset_path: "data/aes_128_pairs.npz"
  batch_size: 64

---

## 🧠 Running BioEncryptSNN Training

To train an SNN for AES-128 decryption reconstruction:

```bash
python train/train_bioencryptsnn.py     --config configs/aes_128_snn.yaml     --output-dir runs/aes_128_snn
```

Configuration files typically define:

    neuron model (LIF, SLIF, etc.)

    temporal resolution (timesteps)

    spike encoding method

    synaptic connection topology

    optimiser, epochs, learning rate

    encryption scheme parameters

---

## 📊 Benchmarking Against AES, RSA, and DES

1. Run classical crypto baselines

```bash
python eval/benchmark_crypto_baselines.py     --schemes AES RSA DES     --key-lengths 128 2048 56     --num-samples 10000     --output results/baselines.json
```

2. Evaluate BioEncryptSNN

```bash
python eval/eval_bioencryptsnn.py     --checkpoint runs/aes_128_snn/best_model.pt     --data data/aes_128_pairs.npz     --output results/bioencryptsnn_aes128.json
```

3. Compare cryptographic performance

```bash
python eval/compare_results.py     --baselines results/baselines.json     --snn results/bioencryptsnn_aes128.json
```

Metrics include:

    latency and throughput

    reconstruction accuracy / bit error rate

    spike-domain noise sensitivity

    computational efficiency

---

## 🧪 Encryption Parameter Optimisation

BioEncryptSNN includes grid search or Bayesian hyperparameter optimisation for:

    key length

    spike timing windows

    synaptic connectivity sparsity/density

    training hyperparameters

Run an optimisation experiment:

```bash
python train/optimize_params.py     --config configs/optimisation_aes.yaml     --search-space configs/search_spaces/aes_key_spike_connectivity.yaml     --max-trials 50     --output-dir runs/optim_aes
```

---

## 🔁 Reproducing Paper Results

Each figure and table in the paper corresponds to a configuration under configs/paper/.
Example: Main AES-128 experiment

```bash
python train/train_bioencryptsnn.py     --config configs/paper/aes_128_main.yaml     --output-dir runs/paper_aes_128
```

Throughput & latency analysis

```bash
python eval/eval_throughput_latency.py     --config configs/paper/aes_128_main.yaml
```

Noise robustness study

```bash
python eval/eval_noise_robustness.py     --config configs/paper/noise_study.yaml
```

Notebooks under notebooks/ reproduce figures from the paper.

📚 Citation

If you use this work, please cite the following:
```
@misc{pulivathi2025privacypreservingspikingneuralnetworks,
      title        = {Privacy-Preserving Spiking Neural Networks: A Deep Dive into Encryption Parameter Optimisation},
      author       = {Mahitha Pulivathi and Ana Fontes Rodrigues and Isibor Kennedy Ihianle and Andreas Oikonomou and Srinivas Boppu and Pedro Machado},
      year         = {2025},
      eprint       = {2510.19537},
      archivePrefix= {arXiv},
      primaryClass = {cs.CR},
      url          = {https://arxiv.org/abs/2510.19537}
}
```
