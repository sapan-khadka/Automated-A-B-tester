import streamlit as st
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    st.warning("Matplotlib is not available. Some visualizations will be disabled.")
from analysis import ABTestAnalyzer
from bayesian import BayesianABTest

st.set_page_config(
    page_title="A/B Testing Analysis Tool",
    page_icon="📊",
    layout="wide"
)

st.title("📊 A/B Testing Analysis Tool")

# Sidebar for input parameters
st.sidebar.header("Test Parameters")

# Input method selection
input_method = st.sidebar.radio(
    "Input Method",
    ["Manual Input", "Upload CSV"]
)

if input_method == "Manual Input":
    st.sidebar.subheader("Control Group")
    control_successes = st.sidebar.number_input(
        "Control Successes",
        min_value=0,
        value=100
    )
    control_trials = st.sidebar.number_input(
        "Control Trials",
        min_value=0,
        value=1000
    )

    st.sidebar.subheader("Treatment Group")
    treatment_successes = st.sidebar.number_input(
        "Treatment Successes",
        min_value=0,
        value=120
    )
    treatment_trials = st.sidebar.number_input(
        "Treatment Trials",
        min_value=0,
        value=1000
    )
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("Preview of uploaded data:")
        st.sidebar.write(df.head())
        
        # Get column names
        available_columns = df.columns.tolist()
        
        # Let user select columns
        group_col = st.sidebar.selectbox(
            "Select Group Column",
            available_columns,
            index=available_columns.index('group') if 'group' in available_columns else 0
        )
        
        successes_col = st.sidebar.selectbox(
            "Select Successes Column",
            available_columns,
            index=available_columns.index('successes') if 'successes' in available_columns else 0
        )
        
        trials_col = st.sidebar.selectbox(
            "Select Trials Column",
            available_columns,
            index=available_columns.index('trials') if 'trials' in available_columns else 0
        )
        
        # Get unique group values
        unique_groups = df[group_col].unique()
        if len(unique_groups) != 2:
            st.error(f"Error: Expected exactly 2 groups, found {len(unique_groups)}")
            st.stop()
            
        control_group = st.sidebar.selectbox(
            "Select Control Group",
            unique_groups
        )
        treatment_group = st.sidebar.selectbox(
            "Select Treatment Group",
            [g for g in unique_groups if g != control_group]
        )
        
        # Process data
        control_data = df[df[group_col] == control_group]
        treatment_data = df[df[group_col] == treatment_group]
        
        control_successes = control_data[successes_col].sum()
        control_trials = control_data[trials_col].sum()
        treatment_successes = treatment_data[successes_col].sum()
        treatment_trials = treatment_data[trials_col].sum()

# Analysis type selection
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["Frequentist", "Bayesian", "Both"]
)

# Initialize analyzers
frequentist_analyzer = ABTestAnalyzer()
bayesian_analyzer = BayesianABTest()

# Main content area
if 'control_successes' in locals() and 'control_trials' in locals() and \
   'treatment_successes' in locals() and 'treatment_trials' in locals():
    
    # Calculate conversion rates
    control_rate = control_successes / control_trials if control_trials > 0 else 0
    treatment_rate = treatment_successes / treatment_trials if treatment_trials > 0 else 0
    
    # Display summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Control Conversion Rate",
            f"{control_rate:.2%}",
            f"({control_successes}/{control_trials})"
        )
    
    with col2:
        st.metric(
            "Treatment Conversion Rate",
            f"{treatment_rate:.2%}",
            f"({treatment_successes}/{treatment_trials})"
        )
    
    # Perform analysis based on selected type
    if analysis_type in ["Frequentist", "Both"]:
        st.header("Frequentist Analysis")
        
        # Run frequentist analysis
        frequentist_results = frequentist_analyzer.analyze_test(
            control_successes, control_trials,
            treatment_successes, treatment_trials
        )
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "P-value",
                f"{frequentist_results['p_value']:.4f}",
                "Significant" if frequentist_results['is_significant'] else "Not Significant"
            )
        
        with col2:
            st.metric(
                "Relative Improvement",
                f"{frequentist_results['relative_improvement']:.2%}"
            )
        
        with col3:
            # Calculate minimum detectable effect
            min_effect = abs(treatment_rate - control_rate)
            # Ensure minimum detectable effect is at least 0.01 (1%)
            min_effect = max(min_effect, 0.01)
            
            st.metric(
                "Required Sample Size",
                str(frequentist_analyzer.calculate_sample_size(
                    control_rate,
                    min_effect
                ))
            )
        
        # Plot results if matplotlib is available
        if matplotlib_available:
            st.subheader("Frequentist Results Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            frequentist_analyzer.plot_results(
                control_successes, control_trials,
                treatment_successes, treatment_trials
            )
            st.pyplot(fig)
    
    if analysis_type in ["Bayesian", "Both"]:
        st.header("Bayesian Analysis")
        
        # Run Bayesian analysis
        bayesian_results = bayesian_analyzer.analyze_test(
            control_successes, control_trials,
            treatment_successes, treatment_trials
        )
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Probability Treatment is Better",
                f"{bayesian_results['probability_treatment_better']:.2%}"
            )
        
        with col2:
            st.metric(
                "Expected Loss",
                f"{bayesian_results['expected_loss']:.4f}"
            )
        
        with col3:
            st.metric(
                "Relative Improvement",
                f"{bayesian_results['relative_improvement']:.2%}"
            )
        
        # Plot posterior distributions if matplotlib is available
        if matplotlib_available:
            st.subheader("Posterior Distributions")
            fig, ax = plt.subplots(figsize=(10, 6))
            bayesian_analyzer.plot_posterior_distributions(
                control_successes, control_trials,
                treatment_successes, treatment_trials
            )
            st.pyplot(fig)
    
    # Add download button for results
    if st.button("Download Results"):
        results = {
            "Control Rate": control_rate,
            "Treatment Rate": treatment_rate,
            "Relative Improvement": (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
        }
        
        if analysis_type in ["Frequentist", "Both"]:
            results.update({
                "P-value": frequentist_results['p_value'],
                "Is Significant": frequentist_results['is_significant'],
                "Required Sample Size": frequentist_analyzer.calculate_sample_size(
                    control_rate,
                    abs(treatment_rate - control_rate)
                )
            })
        
        if analysis_type in ["Bayesian", "Both"]:
            results.update({
                "Probability Treatment is Better": bayesian_results['probability_treatment_better'],
                "Expected Loss": bayesian_results['expected_loss']
            })
        
        results_df = pd.DataFrame([results])
        st.download_button(
            label="Download Results as CSV",
            data=results_df.to_csv(index=False),
            file_name="ab_test_results.csv",
            mime="text/csv"
        )
else:
    st.info("Please enter test parameters in the sidebar.") 