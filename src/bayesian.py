import numpy as np
from scipy import stats
from typing import Tuple, Dict
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False

class BayesianABTest:
    def __init__(self, alpha_prior: float = 1, beta_prior: float = 1):
        """
        Initialize Bayesian A/B Test analyzer.
        
        Args:
            alpha_prior (float): Alpha parameter for Beta prior (default: 1)
            beta_prior (float): Beta parameter for Beta prior (default: 1)
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
    
    def update_posterior(self, successes: int, trials: int) -> Tuple[float, float]:
        """
        Update Beta posterior distribution parameters.
        
        Args:
            successes (int): Number of successful events
            trials (int): Total number of trials
            
        Returns:
            Tuple[float, float]: Updated alpha and beta parameters
        """
        alpha_posterior = self.alpha_prior + successes
        beta_posterior = self.beta_prior + (trials - successes)
        return alpha_posterior, beta_posterior
    
    def calculate_probability_better(self,
                                   control_successes: int,
                                   control_trials: int,
                                   treatment_successes: int,
                                   treatment_trials: int,
                                   n_samples: int = 100000) -> float:
        """
        Calculate probability that treatment is better than control.
        
        Args:
            control_successes (int): Successes in control group
            control_trials (int): Total trials in control group
            treatment_successes (int): Successes in treatment group
            treatment_trials (int): Total trials in treatment group
            n_samples (int): Number of Monte Carlo samples (default: 100000)
            
        Returns:
            float: Probability that treatment is better than control
        """
        # Update posteriors
        alpha_control, beta_control = self.update_posterior(control_successes, control_trials)
        alpha_treatment, beta_treatment = self.update_posterior(treatment_successes, treatment_trials)
        
        # Sample from posteriors
        control_samples = np.random.beta(alpha_control, beta_control, n_samples)
        treatment_samples = np.random.beta(alpha_treatment, beta_treatment, n_samples)
        
        # Calculate probability
        return np.mean(treatment_samples > control_samples)
    
    def calculate_expected_loss(self,
                              control_successes: int,
                              control_trials: int,
                              treatment_successes: int,
                              treatment_trials: int,
                              n_samples: int = 100000) -> float:
        """
        Calculate expected loss if we choose the wrong variant.
        
        Args:
            control_successes (int): Successes in control group
            control_trials (int): Total trials in control group
            treatment_successes (int): Successes in treatment group
            treatment_trials (int): Total trials in treatment group
            n_samples (int): Number of Monte Carlo samples (default: 100000)
            
        Returns:
            float: Expected loss
        """
        # Update posteriors
        alpha_control, beta_control = self.update_posterior(control_successes, control_trials)
        alpha_treatment, beta_treatment = self.update_posterior(treatment_successes, treatment_trials)
        
        # Sample from posteriors
        control_samples = np.random.beta(alpha_control, beta_control, n_samples)
        treatment_samples = np.random.beta(alpha_treatment, beta_treatment, n_samples)
        
        # Calculate expected loss
        loss_control = np.mean(np.maximum(treatment_samples - control_samples, 0))
        loss_treatment = np.mean(np.maximum(control_samples - treatment_samples, 0))
        
        return min(loss_control, loss_treatment)
    
    def plot_posterior_distributions(self,
                                   control_successes: int,
                                   control_trials: int,
                                   treatment_successes: int,
                                   treatment_trials: int,
                                   save_path: str = None):
        """
        Plot posterior distributions for control and treatment groups.
        
        Args:
            control_successes (int): Successes in control group
            control_trials (int): Total trials in control group
            treatment_successes (int): Successes in treatment group
            treatment_trials (int): Total trials in treatment group
            save_path (str, optional): Path to save the plot
        """
        if not matplotlib_available:
            raise ImportError("Matplotlib is not available. Visualization is disabled.")
            
        # Update posteriors
        alpha_control, beta_control = self.update_posterior(control_successes, control_trials)
        alpha_treatment, beta_treatment = self.update_posterior(treatment_successes, treatment_trials)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 1, 1000)
        
        plt.plot(x, stats.beta.pdf(x, alpha_control, beta_control),
                label='Control', color='blue')
        plt.plot(x, stats.beta.pdf(x, alpha_treatment, beta_treatment),
                label='Treatment', color='red')
        
        plt.xlabel('Conversion Rate')
        plt.ylabel('Probability Density')
        plt.title('Posterior Distributions')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def analyze_test(self,
                    control_successes: int,
                    control_trials: int,
                    treatment_successes: int,
                    treatment_trials: int) -> Dict[str, float]:
        """
        Perform complete Bayesian A/B test analysis.
        
        Args:
            control_successes (int): Successes in control group
            control_trials (int): Total trials in control group
            treatment_successes (int): Successes in treatment group
            treatment_trials (int): Total trials in treatment group
            
        Returns:
            Dict containing analysis results
        """
        prob_better = self.calculate_probability_better(
            control_successes, control_trials,
            treatment_successes, treatment_trials
        )
        
        expected_loss = self.calculate_expected_loss(
            control_successes, control_trials,
            treatment_successes, treatment_trials
        )
        
        # Update posteriors for mean calculations
        alpha_control, beta_control = self.update_posterior(control_successes, control_trials)
        alpha_treatment, beta_treatment = self.update_posterior(treatment_successes, treatment_trials)
        
        control_mean = alpha_control / (alpha_control + beta_control)
        treatment_mean = alpha_treatment / (alpha_treatment + beta_treatment)
        
        return {
            'probability_treatment_better': prob_better,
            'expected_loss': expected_loss,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'relative_improvement': (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
        } 