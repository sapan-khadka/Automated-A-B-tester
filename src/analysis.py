import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import tt_ind_solve_power
from typing import Tuple, Dict, Union
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

class ABTestAnalyzer:
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the AB Test Analyzer.
        
        Args:
            alpha (float): Significance level (default: 0.05)
        """
        self.alpha = alpha
        
    def calculate_conversion_rate(self, successes: int, trials: int) -> float:
        """
        Calculate conversion rate.
        
        Args:
            successes (int): Number of successful events
            trials (int): Total number of trials
            
        Returns:
            float: Conversion rate
        """
        return successes / trials if trials > 0 else 0.0
    
    def calculate_p_value(self, 
                         control_successes: int, 
                         control_trials: int,
                         treatment_successes: int,
                         treatment_trials: int) -> float:
        """
        Calculate p-value using chi-square test.
        
        Args:
            control_successes: Number of successes in control group
            control_trials: Total number of trials in control group
            treatment_successes: Number of successes in treatment group
            treatment_trials: Total number of trials in treatment group
            
        Returns:
            float: p-value
        """
        # Validate inputs
        if control_successes < 0 or treatment_successes < 0:
            raise ValueError("Number of successes cannot be negative")
        if control_successes > control_trials or treatment_successes > treatment_trials:
            raise ValueError("Number of successes cannot exceed number of trials")
            
        # Handle edge cases
        if control_successes == 0 and treatment_successes == 0:
            # If there are no successes in either group, the test is not meaningful
            return 1.0
        if control_successes == control_trials and treatment_successes == treatment_trials:
            # If all observations are successes, the test is not meaningful
            return 1.0
        if control_trials == 0 or treatment_trials == 0:
            # If either group has no trials, the test is not meaningful
            return 1.0
            
        control_failures = control_trials - control_successes
        treatment_failures = treatment_trials - treatment_successes
        
        # Ensure failures are non-negative
        control_failures = max(0, control_failures)
        treatment_failures = max(0, treatment_failures)
        
        # Add small constant to avoid zero expected frequencies
        epsilon = 0.5
        contingency_table = np.array([
            [control_successes + epsilon, control_failures + epsilon],
            [treatment_successes + epsilon, treatment_failures + epsilon]
        ])
        
        try:
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            return p_value
        except ValueError as e:
            if "zero element" in str(e):
                # If we still get a zero element error, return a conservative p-value
                return 1.0
            raise
    
    def calculate_confidence_interval(self,
                                    successes: int,
                                    trials: int,
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for conversion rate.
        
        Args:
            successes (int): Number of successful events
            trials (int): Total number of trials
            confidence_level (float): Confidence level (default: 0.95)
            
        Returns:
            Tuple[float, float]: Lower and upper bounds of confidence interval
        """
        if trials == 0:
            return (0.0, 0.0)
            
        p_hat = successes / trials
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        margin = z * np.sqrt((p_hat * (1 - p_hat)) / trials)
        
        return (max(0, p_hat - margin), min(1, p_hat + margin))
    
    def calculate_sample_size(self,
                            baseline_rate: float,
                            minimum_detectable_effect: float,
                            power: float = 0.8) -> int:
        """
        Calculate required sample size for A/B test.
        
        Args:
            baseline_rate (float): Expected conversion rate in control group
            minimum_detectable_effect (float): Minimum effect size to detect
            power (float): Statistical power (default: 0.8)
            
        Returns:
            int: Required sample size per group
            
        Raises:
            ValueError: If baseline_rate is not between 0 and 1
            ValueError: If minimum_detectable_effect is not positive
            ValueError: If power is not between 0 and 1
        """
        # Validate inputs
        if not 0 <= baseline_rate <= 1:
            raise ValueError("Baseline rate must be between 0 and 1")
        if minimum_detectable_effect <= 0:
            raise ValueError("Minimum detectable effect must be positive")
        if not 0 < power < 1:
            raise ValueError("Power must be between 0 and 1")
            
        # Calculate effect size
        try:
            effect_size = proportion_effectsize(
                baseline_rate,
                baseline_rate + minimum_detectable_effect
            )
            
            # Handle very small effect sizes
            if abs(effect_size) < 1e-10:
                raise ValueError(
                    "Effect size is too small to detect. Please increase the minimum detectable effect."
                )
                
            sample_size = tt_ind_solve_power(
                effect_size=effect_size,
                alpha=self.alpha,
                power=power,
                ratio=1.0
            )
            return int(np.ceil(sample_size))
            
        except ValueError as e:
            if "Cannot detect an effect-size of 0" in str(e):
                raise ValueError(
                    "The minimum detectable effect is too small relative to the baseline rate. "
                    "Please increase the minimum detectable effect or choose a different baseline rate."
                ) from e
            raise
    
    def analyze_test(self,
                    control_successes: int,
                    control_trials: int,
                    treatment_successes: int,
                    treatment_trials: int) -> Dict[str, Union[float, Tuple[float, float], bool]]:
        """
        Perform complete A/B test analysis.
        
        Args:
            control_successes: Number of successes in control group
            control_trials: Total number of trials in control group
            treatment_successes: Number of successes in treatment group
            treatment_trials: Total number of trials in treatment group
            
        Returns:
            Dict containing analysis results
        """
        # Validate inputs
        if control_trials <= 0 or treatment_trials <= 0:
            raise ValueError("Number of trials must be positive")
        if control_successes < 0 or treatment_successes < 0:
            raise ValueError("Number of successes cannot be negative")
        if control_successes > control_trials or treatment_successes > treatment_trials:
            raise ValueError("Number of successes cannot exceed number of trials")
            
        control_rate = self.calculate_conversion_rate(control_successes, control_trials)
        treatment_rate = self.calculate_conversion_rate(treatment_successes, treatment_trials)
        
        p_value = self.calculate_p_value(
            control_successes, control_trials,
            treatment_successes, treatment_trials
        )
        
        control_ci = self.calculate_confidence_interval(control_successes, control_trials)
        treatment_ci = self.calculate_confidence_interval(treatment_successes, treatment_trials)
        
        relative_improvement = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'relative_improvement': relative_improvement,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'control_ci': control_ci,
            'treatment_ci': treatment_ci
        }
    
    def plot_results(self,
                    control_successes: int,
                    control_trials: int,
                    treatment_successes: int,
                    treatment_trials: int,
                    save_path: str = None):
        """
        Create visualization of A/B test results.
        
        Args:
            control_successes (int): Successes in control group
            control_trials (int): Total trials in control group
            treatment_successes (int): Successes in treatment group
            treatment_trials (int): Total trials in treatment group
            save_path (str, optional): Path to save the plot
        """
        if not matplotlib_available:
            raise ImportError("Matplotlib is not available. Visualization is disabled.")
            
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        control_rate = self.calculate_conversion_rate(control_successes, control_trials)
        treatment_rate = self.calculate_conversion_rate(treatment_successes, treatment_trials)
        control_ci = self.calculate_confidence_interval(control_successes, control_trials)
        treatment_ci = self.calculate_confidence_interval(treatment_successes, treatment_trials)
        
        # Create bar plot
        groups = ['Control', 'Treatment']
        rates = [control_rate, treatment_rate]
        ci_lower = [control_ci[0], treatment_ci[0]]
        ci_upper = [control_ci[1], treatment_ci[1]]
        
        plt.bar(groups, rates, yerr=[rates[i] - ci_lower[i] for i in range(2)], capsize=10)
        plt.ylabel('Conversion Rate')
        plt.title('A/B Test Results')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 