import pytest
from src.analysis import ABTestAnalyzer
from src.bayesian import BayesianABTest

def test_conversion_rate_calculation():
    analyzer = ABTestAnalyzer()
    assert analyzer.calculate_conversion_rate(50, 100) == 0.5
    assert analyzer.calculate_conversion_rate(0, 100) == 0.0
    assert analyzer.calculate_conversion_rate(100, 100) == 1.0

def test_p_value_calculation():
    analyzer = ABTestAnalyzer()
    # Test with significant difference
    p_value = analyzer.calculate_p_value(100, 1000, 150, 1000)
    assert p_value < 0.05
    
    # Test with no significant difference
    p_value = analyzer.calculate_p_value(100, 1000, 105, 1000)
    assert p_value > 0.05

def test_confidence_interval():
    analyzer = ABTestAnalyzer()
    ci = analyzer.calculate_confidence_interval(50, 100)
    assert len(ci) == 2
    assert 0 <= ci[0] <= ci[1] <= 1

def test_sample_size_calculation():
    analyzer = ABTestAnalyzer()
    sample_size = analyzer.calculate_sample_size(0.1, 0.02)
    assert isinstance(sample_size, int)
    assert sample_size > 0

def test_bayesian_probability():
    analyzer = BayesianABTest()
    prob = analyzer.calculate_probability_better(100, 1000, 120, 1000)
    assert 0 <= prob <= 1

def test_bayesian_expected_loss():
    analyzer = BayesianABTest()
    loss = analyzer.calculate_expected_loss(100, 1000, 120, 1000)
    assert loss >= 0

def test_complete_analysis():
    frequentist = ABTestAnalyzer()
    bayesian = BayesianABTest()
    
    # Test data
    control_successes = 100
    control_trials = 1000
    treatment_successes = 120
    treatment_trials = 1000
    
    # Frequentist analysis
    freq_results = frequentist.analyze_test(
        control_successes, control_trials,
        treatment_successes, treatment_trials
    )
    assert 'p_value' in freq_results
    assert 'is_significant' in freq_results
    assert 'control_ci' in freq_results
    assert 'treatment_ci' in freq_results
    
    # Bayesian analysis
    bayes_results = bayesian.analyze_test(
        control_successes, control_trials,
        treatment_successes, treatment_trials
    )
    assert 'probability_treatment_better' in bayes_results
    assert 'expected_loss' in bayes_results
    assert 'control_mean' in bayes_results
    assert 'treatment_mean' in bayes_results 