# 🌍 Climate Credit Risk Analyzer

> **Gemma 4 Good Hackathon Submission** | Global Resilience Track

A financial intelligence tool that democratizes institutional-grade climate risk assessment using open data and Gemma 4, enabling financial institutions worldwide to make informed, climate-conscious lending decisions — at zero cost.

---

## 🚀 Live Demo

**Hugging Face Space:** https://huggingface.co/spaces/SubramaniMokkala/climate-credit-risk-analyzer

---

## 🎯 Problem Statement

Most small and mid-size financial institutions have no systematic way to assess climate risk in their lending decisions. They either ignore it entirely or rely on expensive third-party ESG data providers. This means:
- Loans flow to high-carbon companies without proper risk pricing
- Banks are exposed to stranded asset risk they don't know about
- Capital keeps flowing to fossil fuel industries unchecked

The UN's *Financing for Sustainable Development Report 2026* warns that developing countries face shrinking fiscal space and uneven access to climate finance tools — widening the gap between climate commitments and action.

---

## 💡 Solution

**Climate Credit Risk Analyzer** combines machine learning and Gemma 4 to provide:

1. **📊 Risk Scoring** — XGBoost model scores any S&P 500 company across 5 climate risk dimensions
2. **📄 AI Report Generation** — Gemma 4 generates a professional 3-paragraph analyst memo
3. **💬 Interactive Chat** — Analysts can ask follow-up questions about the risk assessment

---

## 🏗️ Architecture

```
User Input (Company / Sector)
        ↓
Data Pipeline
  ├── S&P 500 Companies + Financials
  ├── ESG Risk Ratings
  ├── ND-GAIN Country Index 2026
  └── Feature Engineering (14 features)
        ↓
XGBoost Risk Scoring Model
  ├── Physical Risk Score
  ├── Transition Risk Score
  └── Climate Credit Risk Tier (Low / Medium / High / Critical)
        ↓
Gemma 4 (via Google AI Studio API)
  ├── Professional analyst report generation
  └── Interactive Q&A chat interface
        ↓
Streamlit App (Deployed on HF Spaces)
```

---

## 📊 Datasets

| Dataset | Source | Purpose |
|---|---|---|
| S&P 500 Companies & Stocks | Kaggle (andrewmvd) | Company financials, sector, market cap |
| S&P 500 ESG Risk Ratings | Kaggle (pritish509) | ESG scores, environmental risk |
| ND-GAIN Country Index 2026 | Notre Dame Global | Country climate vulnerability & readiness |
| World Bank Climate Change Data | Kaggle (World Bank) | Supporting climate indicators |

---

## 🤖 Model Details

**XGBoost Classifier**
- 421 S&P 500 companies
- 14 engineered features including climate gap, carbon sector flag, ESG per market cap
- Custom Climate Credit Risk Score (weighted composite of 5 dimensions)
- 4-class output: Low / Medium / High / Critical
- 59% accuracy on balanced 4-class problem | 90% precision on Critical tier

**Risk Score Composition:**
| Component | Weight |
|---|---|
| ESG Risk Score | 30% |
| Country Climate Vulnerability | 25% |
| High Carbon Sector | 20% |
| Low Climate Readiness | 15% |
| Low Profitability | 10% |

**Gemma 4** (gemma-4-26b-a4b-it via Google AI Studio)
- Report generation: structured 3-paragraph analyst memo
- Chat interface: context-aware Q&A using company risk profile

---

## 🌍 Impact & Good

This tool directly addresses the **Global Resilience** track by:
- **Democratizing** institutional-grade climate risk analysis (previously only available to large banks)
- **Accelerating green finance** by helping lenders identify and price climate risk
- **Zero cost** — built entirely on open data and free APIs
- **Accessible globally** — any analyst at any institution can use it

---

## 🛠️ Tech Stack

- **ML:** XGBoost, Scikit-learn, SHAP
- **LLM:** Gemma 4 (gemma-4-31b-it) via Google AI Studio
- **App:** Streamlit
- **Deployment:** Hugging Face Spaces (Docker)
- **Data:** Pandas, NumPy
- **Visualization:** Plotly

---

## 📁 Repository Structure
climate-credit-risk-analyzer/
├── app.py                          # Main Streamlit application
├── climate_credit_risk_model.pkl   # Trained XGBoost model
├── climate_credit_risk_data.csv    # Processed dataset (421 companies)
├── model_metadata.json             # Feature columns and mappings
├── requirements.txt                # Python dependencies
├── Dockerfile                      # HF Spaces deployment config
└── README.md                       # This file

---

## 🚀 Run Locally

```bash
git clone https://github.com/SubramaniMokkala/climate-credit-risk-analyzer
cd climate-credit-risk-analyzer
pip install -r requirements.txt
export GOOGLE_API_KEY=your_api_key_here
streamlit run app.py
```

---

## 👤 Author

**Subramani Mokkala**
B.Tech Computer Science (Data Science) | KGRCET Hyderabad
[GitHub](https://github.com/SubramaniMokkala) | [Hugging Face](https://huggingface.co/SubramaniMokkala)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for the Gemma 4 Good Hackathon by Kaggle & Google DeepMind*
