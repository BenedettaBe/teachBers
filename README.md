# TeachBers – Teacher of Numbers

**TeachBers** is an educational Python project designed to help children improve their digit handwriting. It uses computer vision to analyze handwritten numbers and integrates with a local language model (Gemma 2B IT) via LM Studio to provide personalized feedback and improvement tips.

## 🎯 Purpose

The goal of this project is to combine image processing with AI-powered analysis to:

- Encourage proper handwriting habits for numbers.
- Provide real-time or batch feedback on digit writing.
- Engage children in a fun, interactive, and tech-enhanced learning experience.

## 🧠 Powered by Local AI

The feedback system runs locally using:

- [LM Studio](https://lmstudio.ai/) (offline AI model runner)
- Model: `gemma-2-2b-it-gguf` (an Italian language model in GGUF format)

> ⚠️ Make sure LM Studio is running and configured with `gemma-2-2b-it-gguf` for the system to work correctly.

## 🗂 Project Structure

```
teachBers/
├── teachbers.py         # Main Python script
├── images/              # Input images
├── fotoCifre/           # Extracted digit images
├── risultati/           # Processed output and feedback
├── modello/             # ML model and Jupyter notebook for training
```

## 🧰 Requirements

Install dependencies with:

```bash
pip install openai joblib numpy opencv-python pillow matplotlib
```

Additional requirements:

- [LM Studio](https://lmstudio.ai/)
- Model: `gemma-2-2b-it-gguf` (download and configure in LM Studio)

---

## 🚀 Future Improvements (optional)

Some possible future features include:

- A child-friendly graphical user interface
- Age-based digit evaluation
- Voice feedback for greater engagement
- Progress tracking and performance statistics over time
