# TeachBers - Teacher of Numbers.

**TeachBers** is an educational project in Python designed to help children improve number writing. It uses computer vision to analyze handwritten numbers and integrates with a local language model (Gemma 2B IT) via LM Studio to provide personalized feedback and suggestions for improvement.

## 🎯 Purpose.

The goal of this project is to combine image processing with artificial intelligence-based analysis to:

- Encourage correct handwriting habits for numbers.
- Provide feedback on number writing.
- Engage children in a fun, interactive, technology-based learning experience.

## 🧠 Powered by local artificial intelligence.

The feedback system is executed locally using:

- [LM Studio](https://lmstudio.ai/) (offline artificial intelligence model runner).
- Model: `gemma-2-2b-it-gguf` (an Italian language model in GGUF format).

> ⚠️ Make sure LM Studio is running and configured with `gemma-2-2b-it-gguf` for the system to work properly.

## 🗂 Project structure.
```
teachBers/
├── teachbers. py # Main Python script
├── images/ # Images saved in the program execution
├── photoFigures/ # Images of digits extracted in input
├── results/ # Output processed for feedback
├── model/ # ML model and Jupyter notebook for training
├── standard/ # Images of the standard digits for the comparison
```

## 🧰 Requirements

Install:
- [LM Studio](https://lmstudio.ai/)
- Model: `gemma-2-2b-it-gguf` (download and configure in LM Studio)

Install dependencies with:
```bash
pip install openai joblib numpy opencv-python pillow matplotlib
```
