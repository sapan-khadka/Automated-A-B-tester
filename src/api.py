from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from analysis import ABTestAnalyzer
from bayesian import BayesianABTest

app = FastAPI(
    title="A/B Testing API",
    description="API for analyzing A/B test results and determining statistical significance",
    version="1.0.0"
)

class ABTestRequest(BaseModel):
    control_successes: int
    control_trials: int
    treatment_successes: int
    treatment_trials: int
    analysis_type: Optional[str] = "both"  # "frequentist", "bayesian", or "both"

class CSVUploadRequest(BaseModel):
    csv_data: str  # Base64 encoded CSV data
    analysis_type: Optional[str] = "both"

# Initialize analyzers
frequentist_analyzer = ABTestAnalyzer()
bayesian_analyzer = BayesianABTest()

@app.post("/analyze")
async def analyze_test(request: ABTestRequest):
    """
    Analyze A/B test results using either frequentist or Bayesian methods.
    """
    try:
        results = {}
        
        # Calculate conversion rates
        control_rate = request.control_successes / request.control_trials if request.control_trials > 0 else 0
        treatment_rate = request.treatment_successes / request.treatment_trials if request.treatment_trials > 0 else 0
        
        results["summary"] = {
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "relative_improvement": (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        }
        
        if request.analysis_type in ["frequentist", "both"]:
            frequentist_results = frequentist_analyzer.analyze_test(
                request.control_successes, request.control_trials,
                request.treatment_successes, request.treatment_trials
            )
            
            results["frequentist"] = {
                "p_value": frequentist_results["p_value"],
                "is_significant": frequentist_results["is_significant"],
                "control_ci": frequentist_results["control_ci"],
                "treatment_ci": frequentist_results["treatment_ci"],
                "required_sample_size": frequentist_analyzer.calculate_sample_size(
                    control_rate,
                    abs(treatment_rate - control_rate)
                )
            }
        
        if request.analysis_type in ["bayesian", "both"]:
            bayesian_results = bayesian_analyzer.analyze_test(
                request.control_successes, request.control_trials,
                request.treatment_successes, request.treatment_trials
            )
            
            results["bayesian"] = {
                "probability_treatment_better": bayesian_results["probability_treatment_better"],
                "expected_loss": bayesian_results["expected_loss"],
                "control_mean": bayesian_results["control_mean"],
                "treatment_mean": bayesian_results["treatment_mean"]
            }
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze-csv")
async def analyze_csv(request: CSVUploadRequest):
    """
    Analyze A/B test results from uploaded CSV data.
    """
    try:
        # Decode and parse CSV data
        df = pd.read_csv(pd.compat.StringIO(request.csv_data))
        
        # Assuming CSV has columns: group, successes, trials
        control_data = df[df['group'] == 'control']
        treatment_data = df[df['group'] == 'treatment']
        
        control_successes = control_data['successes'].sum()
        control_trials = control_data['trials'].sum()
        treatment_successes = treatment_data['successes'].sum()
        treatment_trials = treatment_data['trials'].sum()
        
        # Create request object
        ab_request = ABTestRequest(
            control_successes=control_successes,
            control_trials=control_trials,
            treatment_successes=treatment_successes,
            treatment_trials=treatment_trials,
            analysis_type=request.analysis_type
        )
        
        # Use existing analyze endpoint
        return await analyze_test(ab_request)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint returning API information.
    """
    return {
        "name": "A/B Testing API",
        "version": "1.0.0",
        "description": "API for analyzing A/B test results and determining statistical significance"
    } 