# teachBers (Teacher of Numbers)

**teachBers** is an educational Python project designed to support children in improving their digit handwriting. It uses computer vision techniques to analyze how numbers are written, and integrates with a local language model (Gemma 2B IT) via LM Studio to provide personalized feedback and suggestions.

## 🔍 Purpose

The goal of this project is to combine image processing with AI-assisted analysis to:

- Encourage better number writing habits.
- Provide real-time or batch feedback on handwritten digits.
- Engage children in a fun, interactive, and technology-enhanced learning process.

## 🧠 Powered by Local AI

The feedback system is built on:

- [LM Studio](https://lmstudio.ai/) (offline AI model runner)
- `gemma-2-2b-it-gguf` (Italian language model in GGUF format)

> ⚠️ Make sure LM Studio is running and configured with `gemma-2-2b-it-gguf` for full functionality.

## 🗂 Project Structure
teachBers/ 
- ├── teachbers.py # Main Python script
- ├── images/ # Input images
- ├── fotoCifre/ # Extracted digit images
- ├── risultati/ # Processed output and analysis
- ├── modello/ # The Machine Learning model and Jupyter notebook that create the machine learning model


## 🧰 Requirements

Install dependencies with:
- library:
  - opencv-python
  - numpy
  - matplotlib
- LM Studio
- Model: gemma-2-2b-it-gguf (download and configure inside LM Studio)
