# Continuous-Thought Machines for NLP (CTM-NLP)

**An adaptation of the [Continuous-Thought Machines](https://github.com/SakanaAI/continuous-thought-machines) architecture by Sakana AI for Natural Language Processing (NLP) tasks.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Original Paper](https://img.shields.io/badge/Original_Paper-Sakana_AI-orange)](https://pub.sakana.ai/ctm/)

This repository contains the code and resources for applying the Continuous-Thought Machines (CTM) architecture to NLP tasks. The original CTM was introduced as a versatile architecture capable of solving problems across different domains, from image classification to maze solving [1]. We adapt its core principles for deep text understanding and generation.

## Core Concepts

The key idea behind CTM is that "thought takes time, and reasoning is a process" [1]. Unlike traditional models like Transformers with a fixed number of layers, CTM introduces an internal time axis. This allows the model to dynamically "think" about the input data for as long as needed to solve a given task.

For NLP, this translates to:

*   **Dynamic Processing Depth**: Simple sentences can be processed quickly, while complex syntactic and semantic structures can trigger more "thinking" iterations.
*   **Improved Handling of Long-Range Dependencies**: Thanks to neurons that process the history of incoming signals, the model can potentially capture context better in long documents.
*   **Neural Synchronization as an Attention Mechanism**: Instead of classic attention mechanisms, information is encoded in the firing times of neurons, which could be a more efficient way to highlight important parts of the text.

## Supported Tasks

This architecture has been adapted and tested for the following NLP tasks:

*   **Text Classification** (e.g., sentiment analysis)
*   **(In progress) Semantic Similarity Analysis**
*   **(In progress) Text Generation**

## Installation

It is recommended to use `conda` to set up the environment.

```bash
# Create and activate the environment
conda create --name=ctm_nlp python=3.10
conda activate ctm_nlp

# Install dependencies
pip install -r requirements.txt
```

Ensure you have a compatible version of PyTorch installed.

## Repository Structure

```
├── models/             # Source code for the adapted CTM-NLP architecture
├── ag_news.ipynb       # Starter
└── requirements.txt    # Project dependencies
```

## Acknowledgements

This work is a direct adaptation and extension of the ideas presented by the **Sakana AI** team. Huge thanks to the authors for their groundbreaking work and for open-sourcing their code.

*   **Original Repository**: [SakanaAI/continuous-thought-machines](https://github.com/SakanaAI/continuous-thought-machines)
*   **Technical Report**: [pub.sakana.ai/ctm/](https://pub.sakana.ai/ctm/)

## License

This project is licensed under the Apache 2.0 License, the same as the original CTM repository.
