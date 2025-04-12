# Automated A/B Testing & Statistical Significance Tool

A lightweight Python tool for analyzing A/B test results and determining statistical significance.

## Features

- Statistical significance analysis (p-values, confidence intervals)
- Bayesian A/B testing support
- Multiple output formats (PDF/HTML/API)
- Streamlit UI for interactive analysis
- FastAPI for programmatic access
- Jupyter notebook integration
- Google Analytics and Facebook Ads API integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ab-testing-tool.git
cd ab-testing-tool
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Streamlit UI
```bash
streamlit run src/app.py
```

### FastAPI Server
```bash
uvicorn src.api:app --reload
```

### Jupyter Notebook
```bash
jupyter notebook
```

## Project Structure

```
ab-testing-tool/
├── src/
│   ├── __init__.py
│   ├── app.py              # Streamlit UI
│   ├── api.py              # FastAPI endpoints
│   ├── analysis.py         # Core statistical analysis
│   ├── bayesian.py         # Bayesian analysis
│   └── integrations/       # API integrations
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test files
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

"# Automated-A-B-tester" 
