# Ciphertext Transport through Spiking Neural Networks using BioEncryptSNN

This repository contains the reference implementation and experimental artefacts accompanying the paper:

> **Ciphertext Transport through Spiking Neural Networks using BioEncryptSNN**  
> *Mahitha Pulivathi, Ana Fontes Rodrigues, Isibor Kennedy Ihianle, Andreas Oikonomou, Srinivas Boppu, Pedro Machado*  
> **arXiv:** https://arxiv.org/abs/2510.19537

## Overview

BioEncryptSNN is a proof-of-concept framework for investigating the representation and transport of conventionally generated cryptograms through a Spiking Neural Network (SNN).

Cryptographic operations are performed outside the spiking network using established cryptographic algorithms. The resulting cryptogram, including any metadata required for subsequent decryption, is serialised into a binary representation and converted into dual-rail spike activity for transport through the SNN.

The SNN does not perform encryption, decryption, key generation, or cryptographic authentication. Its role is to encode, propagate, and reconstruct encrypted binary information through spiking neural dynamics.

The experimental evaluation considers six cryptographic configurations:

- Simplified DES (S-DES), used as a pedagogical configuration
- DES in CBC mode
- AES-128 in CBC mode
- ChaCha20
- RSA-2048 with OAEP-SHA256
- ECIES using P-256 ECDH, HKDF-SHA256, and AES-128-GCM

The experiments evaluate whether a common teacher-independent spiking transport pathway can represent and reconstruct cryptograms produced by different cryptographic constructions.

The principal analyses include:

- bit-level cryptogram transport through an SNN;
- Bit Error Rate (BER);
- exact cryptogram recovery;
- exact plaintext recovery after conventional decryption;
- CMA-ES optimisation of the SNN transport operating point;
- static versus Triplet-STDP transport conditions;
- Teaching Layer ablations;
- excitatory, inhibitory, and balanced Poisson perturbation experiments;
- spike-train distance analysis;
- stage-level execution latency;
- payload-size scaling;
- downstream classification experiments performed independently of the SNN transport task; and
- an experimental evidence audit summarising which scientific criteria are supported by the generated results.

BioEncryptSNN should be interpreted as an experimental framework for neuromorphic ciphertext transport. It is not introduced as a new cryptographic primitive, and the SNN does not provide additional confidentiality, integrity, authentication, or formal cryptographic security guarantees.

## Repository Structure

The repository is organised as follows:

```text
.
├── data/
├── results_es_full_noise_6_ciphers/
├── .gitignore
├── BioEncryptSNN_CMA_ES_FULL_NOISE_6_CIPHERS.ipynb
├── LICENSE
├── README.md
└── requirements.txt
```

### `BioEncryptSNN_CMA_ES_FULL_NOISE_6_CIPHERS.ipynb`

The Jupyter notebook contains the main executable experimental workflow, including:

- environment and dependency reporting;
- cryptographic configuration and cryptogram generation;
- dual-rail binary-to-spike encoding;
- NEST network construction;
- CMA-ES optimisation;
- static and Triplet-STDP transport experiments;
- Teaching Layer ablations;
- exact-recovery analysis;
- perturbation experiments;
- spike-train analysis;
- stage-level latency measurement;
- payload-size scaling;
- downstream classification;
- statistical analysis; and
- experimental evidence auditing.

The notebook should be executed from the first cell so that cryptographic material, random seeds, model state, and result tables are generated consistently within the same experimental session.

### `data/`

Contains the data required by the notebook and/or generated input artefacts used by the experiments.

### `results_es_full_noise_6_ciphers/`

Contains generated outputs from the six-cipher experimental workflow, including result tables and supporting artefacts produced by the notebook.

### `environment.yml`

Conda environment specification for reproducing the software environment.

### `requirements.txt`

Python package requirements for installations that do not use the Conda environment.

### `environment.json`

Recorded environment and dependency metadata used to document the execution environment associated with the experiments.

## Installation

The experiments require Python, Jupyter, NEST Simulator, and the Python packages listed in the repository.

### Option 1: Conda environment

Create the environment from:

```bash
conda env create -f environment.yml
```

Then activate the environment using the environment name defined in `environment.yml`.

### Option 2: Python virtual environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

A compatible NEST Simulator installation is also required. The experiments reported in the study use NEST 3.9.

## Running the Experiments

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Open:

```text
BioEncryptSNN_CMA_ES_FULL_NOISE_6_CIPHERS.ipynb
```

Run the notebook sequentially from the first cell.

The notebook provides the complete executable experimental pipeline rather than relying on separate training or evaluation scripts.

## BioEncryptSNN Processing Pipeline

The end-to-end workflow is:

```text
Plaintext
   |
   v
Conventional encryption
   |
   v
Complete cryptogram
   |
   v
Binary serialisation
   |
   v
Dual-rail spike encoding
   |
   v
Layer 1 -> Layer 2 -> Layer 3
   |
   v
Spike readout
   |
   v
Binary reconstruction
   |
   v
Recovered cryptogram
   |
   v
Conventional decryption
   |
   v
Recovered plaintext
```

The transported representation includes the encrypted payload and any metadata required for subsequent decryption. Depending on the cryptographic configuration, this can include ciphertext, initialisation vectors, nonces, authentication tags, and ephemeral public-key material.

No error-correction mechanism is applied to the recovered cryptogram.

## Cryptographic Configurations

| Algorithm | Configuration | Transported representation |
| --- | --- | --- |
| S-DES | Pedagogical 10-bit-key Simplified DES | Ciphertext |
| DES | 56-bit effective key, CBC, PKCS#7 | IV + ciphertext |
| AES-128 | 128-bit key, CBC, PKCS#7 | IV + ciphertext |
| ChaCha20 | 256-bit key, 96-bit nonce | Nonce + ciphertext |
| RSA-2048 | OAEP-SHA256 | RSA-OAEP ciphertext blocks |
| ECIES | P-256 ECDH, HKDF-SHA256, AES-128-GCM | Ephemeral public key + nonce + authentication tag + ciphertext |

These configurations are not compared in terms of cryptographic security strength. They provide structurally different encrypted representations for evaluating a common SNN transport pathway.

## Spiking Transport Architecture

BioEncryptSNN is implemented in NEST using `iaf_psc_alpha` neurons and a dual-rail representation.

Each binary bit is represented by two complementary neural channels:

```text
bit = 0 -> logical-0 rail active
bit = 1 -> logical-1 rail active
```

An eight-bit cryptogram byte is represented by 16 Layer 1 neurons.

The network populations are:

```text
Layer 1: 16 neurons
Layer 2: 32 neurons
Layer 3: 16 neurons
Teaching Layer: 16 neurons, used only for ablation experiments
```

Layer 3 contains eight complementary output pairs. Reciprocal inhibition between the two rails associated with each bit introduces competition between the logical-0 and logical-1 representations.

Each cryptogram byte is presented for 20 ms followed by a 5 ms inter-symbol interval. Payload length therefore changes the number of sequential symbol presentations rather than changing network size.

## CMA-ES Optimisation

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is used to determine the operating point of the teacher-independent SNN transport pathway.

The optimisation searches five transport parameters:

- Layer 1 to Layer 2 synaptic weight;
- initial Layer 2 to Layer 3 synaptic weight;
- Layer 2 to Layer 3 maximum weight;
- active dual-rail input current; and
- reciprocal Layer 3 inhibitory weight.

CMA-ES uses a fixed balanced calibration sequence. The calibration procedure does not use held-out cryptograms, cryptographic keys, downstream classification labels, or the Teaching Layer.

The selected operating point is subsequently evaluated across all six cryptographic configurations without cipher-specific optimisation.

## Triplet-STDP Evaluation

Triplet Spike-Timing-Dependent Plasticity is evaluated separately from CMA-ES initialisation.

The Layer 2 to Layer 3 projection is compared under conditions including:

```text
Static CMA-ES configuration
```

and:

```text
CMA-ES initialisation + Triplet-STDP
```

The comparison separates the effect of parameter initialisation from subsequent activity-dependent synaptic adaptation.

Changes in synaptic weights demonstrate that plasticity is active, but synaptic change alone is not treated as evidence of improved transport fidelity. Transport performance is assessed using BER, exact recovery, and associated quantitative metrics.

## Teaching Layer Ablation

The Teaching Layer is not part of the primary teacher-independent transport pathway.

When enabled, it provides a direct one-to-one reference projection to Layer 3 and is used only as an experimental ablation control. The Teaching Layer is disabled during CMA-ES optimisation and during the primary teacher-independent transport evaluation.

Ablation experiments distinguish performance attributable to:

```text
Layer 1 -> Layer 2 -> Layer 3
```

from performance associated with the direct Teaching Layer projection.

## Evaluation Metrics

### Bit Error Rate

Bit Error Rate measures the proportion of reconstructed bits that differ from the transmitted cryptogram. A low BER represents approximate reconstruction but does not imply exact recovery.

### Exact Cryptogram Recovery

Exact cryptogram recovery requires equality between the complete transmitted and reconstructed cryptograms. For configurations requiring IVs, nonces, authentication tags, or ephemeral public-key material, these fields form part of the equality test.

### Exact Plaintext Recovery

The reconstructed cryptogram is supplied to the corresponding conventional decryption procedure. Exact plaintext recovery requires the resulting plaintext to match the original plaintext.

### Decoder Ties

The dual-rail decoder compares the spike counts of the logical-0 and logical-1 output rails. Equal spike counts are recorded as decoder ties.

### Spike-Train Distance

Temporal similarity between spike trains is evaluated using spike-domain distance measures including Victor-Purpura distance and van Rossum distance.

## Noise Sensitivity

BioEncryptSNN includes controlled perturbation experiments designed to characterise the sensitivity of the SNN transport pathway.

Three perturbation conditions are evaluated:

- excitatory Poisson perturbation;
- inhibitory Poisson perturbation; and
- balanced excitatory/inhibitory Poisson perturbation.

The evaluated perturbation rates are:

```text
0 Hz
5 Hz
10 Hz
25 Hz
50 Hz
100 Hz
```

Perturbations are introduced at Layer 2. The perturbation amplitude is defined relative to the Layer 1 to Layer 2 transport weight to provide a consistent perturbation-to-signal relationship across different operating points.

These experiments characterise noise sensitivity rather than establishing a general noise-robust operating region.

## Stage-Level Latency

Execution time is measured separately for the principal stages of the pipeline:

```text
encryption
binary/spike encoding
network construction
NEST simulation
spike readout
binary reconstruction
decryption
```

The measurements characterise software execution latency in the NEST implementation. They are not evidence of neuromorphic hardware acceleration or physical energy efficiency.

## Payload-Size Scaling

Payload-size experiments evaluate how execution time and reconstruction behaviour change as the transported cryptogram becomes larger.

The BioEncryptSNN topology remains fixed as payload size increases. Larger cryptograms therefore increase the number of sequential symbol presentations rather than the number of neurons or synapses.

Scalability conclusions should be restricted to payload sizes measured experimentally. Computational scaling and functional scalability are treated separately: a payload may remain computationally processable even when exact reconstruction is not achieved.

## Downstream Classification

The repository also contains a downstream classification experiment using the JSVulnerabilityDataSet.

This experiment is independent of the SNN ciphertext-transport task. The SNN does not perform classification. Instead, the workflow is:

```text
SNN transport
   |
   v
Recovered cryptogram
   |
   v
Conventional decryption
   |
   v
Plaintext/source-derived tabular features
   |
   v
Conventional classifier
```

The evaluated classifiers include:

- Random Forest;
- HistGradientBoosting;
- Decision Tree;
- K-Nearest Neighbours;
- Support Vector Machine;
- Logistic Regression; and
- Gaussian Naive Bayes.

A label-shuffle control provides a negative classification baseline. Classification metrics are application-level measures and are not interpreted as measures of cryptographic security or SNN transport fidelity.

## Experimental Evidence Audit

The notebook includes an experimental evidence audit that distinguishes whether an experiment was performed from whether its associated scientific criterion was satisfied.

The audit covers:

- Triplet-STDP weight traces;
- classification and label-shuffle controls;
- teacher-independent transport;
- CMA-ES initialisation control;
- quantitative noise sensitivity;
- stage-level latency;
- scalability;
- DES/S-DES terminology;
- reproducibility and statistical analysis; and
- security and energy scope.

A failed criterion does not indicate that the corresponding experiment was omitted. It indicates that the generated evidence did not support that criterion.

## Reproducibility

For reproducibility:

1. use the supplied `environment.yml` or `requirements.txt`;
2. confirm that NEST 3.9 is available;
3. run the notebook from the first cell;
4. retain the generated environment metadata;
5. retain the CMA-ES optimisation history and selected operating point;
6. retain all generated result tables and statistical outputs; and
7. keep random seeds unchanged when reproducing the reported experiments.

Deterministic seeds are used for experimental reproducibility only. They should not be interpreted as a recommendation for production cryptographic key, IV, or nonce generation.

## Scope and Limitations

BioEncryptSNN is a research prototype for investigating the transport of encrypted binary representations through spiking neural dynamics.

The study does not claim that:

- the SNN provides cryptographic confidentiality;
- the SNN constitutes a new encryption algorithm;
- the SNN improves the security strength of the underlying cipher;
- the implementation provides formal resistance to cryptanalytic attack;
- low BER implies exact cryptogram recovery;
- the transport pathway is universally lossless;
- the perturbation experiments establish general noise robustness;
- NEST execution demonstrates neuromorphic hardware energy efficiency; or
- software execution time demonstrates an advantage over conventional cryptographic implementations.

Cryptographic security remains attributable to the underlying cryptographic algorithms and their implementations.

## Citation

If you use this repository or the associated work, please cite:

```bibtex
@misc{pulivathi2025bioencryptsnn,
      title         = {Ciphertext Transport through Spiking Neural Networks using BioEncryptSNN},
      author        = {Mahitha Pulivathi and Ana Fontes Rodrigues and Isibor Kennedy Ihianle and Andreas Oikonomou and Srinivas Boppu and Pedro Machado},
      year          = {2025},
      eprint        = {2510.19537},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CR},
      url           = {https://arxiv.org/abs/2510.19537}
}
```

## Licence

See [LICENSE](LICENSE) for the repository licence.
