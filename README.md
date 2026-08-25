# AI Product Feedback Categorizer

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

A zero-shot Natural Language Processing (NLP) dashboard designed to help Product Managers and Engineering Leads extract actionable business tickets from raw user sentiment. 

Instead of relying on basic positive/negative sentiment analysis, this tool scrapes live app reviews from **Google Play Store**, and categorizes them into actionable business buckets (e.g., Bug Reports, Feature Requests, Pricing Complaints) using a local, open-source AI model - meaning no tokens and no long-term costs to running the program.

## ✨ Key Features
* **Multi-Platform Scraping:** Dynamically fetches the latest reviews from Android storefronts without requiring paid API keys.
* **Zero-Shot Classification:** Utilizes Hugging Face's `distilbert-base-uncased-mnli` to categorize unstructured text on the fly without fine-tuning or expensive OpenAI API calls.
* **Stateful UI:** Built with a custom cyberpunk-inspired CSS theme (just for fun!). 
* **Export to CSV:** CSV file export to implement data directly into Excel and Google Sheets.

## 🏗️ Tech Stack
* **Frontend:** Streamlit (with injected custom CSS)
* **Backend/Data:** Python, Pandas
* **Machine Learning:** Hugging Face `transformers`, PyTorch
* **Data Visualization:** Plotly
* **Scraping:** `google-play-scraper`

## 🚀 Quick Start (Local Development)

To run this project locally without a cloud GPU:

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/ai-feedback-categorizer.git](https://github.com/yourusername/ai-feedback-categorizer.git)
   cd ai-feedback-categorizer