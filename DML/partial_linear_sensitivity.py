#from DOWHY
import logging
import scipy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
#from tqdm import tqdm #bootsratp progress
import matplotlib.pyplot as plt
from DML.utils import get_regression_r2

#Turn nonparametricsensitivityanalyzer into a simpler function - adaptatyion of the code from DoWhy package to perform sensitivity analysis for causal inference. The code is modified to work with a sliding window approach to estimate CATE over time, capturing non-stationary effects in the data.

def sensitivity_analysis(num_splits, shuffle_data, shuffle_random_seed, benchmark_common_causes, significance_level, frac_strength_treatment, frac_strength_outcome, 
                         theta_s, t_resid, agg_resid, base_t, base_y, observed_common_causes, outcome, treatment, algo, params_t=None, params_y=None, plot=True, benchmarking=True):
    """
        Function to perform sensitivity analysis.
        The following formulae are used to obtain the upper and lower bound respectively.
        θ+ = θ_s + S * C_g * C_α
        θ- = θ_s - S * C_g * C_α
        where θ_s is the obtained estimate,  S^2 = E[Y - gs]^2  * E[α_s]^ 2
        S is obtained by debiased machine learning.
        θ_s = E[m(W, gs) + (Y - gs) * α_s]
        σ² = E[Y - gs]^2
        ν^2 = 2 * E[m(W, α_s )] - E[α_s ^ 2]

        :param plot: plot = True generates a plot of lower confidence bound of the estimate for different variations of unobserved confounding.
                     plot = False overrides the setting

        :returns: RV, RV_alpha, results, r2yu_tw, r2tu_w, r2yu_tw, r2tu_w
    """
   
    logger = logging.getLogger(__name__)

    # features are the observed confounders
    features = observed_common_causes.copy()
   
    #Convert to numpy arrays for regression
    W = features.to_numpy() #observed confounders
    T =  treatment.copy().values.ravel()

    Y = outcome.copy()
    Y = Y.values.ravel()

    t_resid = np.asarray(t_resid).ravel()
    agg_resid = np.asarray(agg_resid).ravel()
    base_t = np.asarray(base_t).ravel()
    base_y = np.asarray(base_y).ravel()

    # Set up cross-validation for regression and treatment estimation
    #May be able to merge with fit_residualized_model function to avoid duplication of code?
    cv = KFold(n_splits=num_splits, shuffle=shuffle_data, random_state=shuffle_random_seed)

    split_indices = list(cv.split(W))
    # Y - g_s = (Y - E[Y|W]) - theta_s * (T - E[T|W]) = Y_res - theta_s * T_res
    residualized_outcome_second_stage = agg_resid - theta_s * t_resid

    

    nu_2 = np.mean(t_resid**2)  # E[T_res^2]  (== E[alpha_s^-1], since alpha_s = T_res/nu_2)
    sigma_2 = np.mean(residualized_outcome_second_stage**2)  # E[(Y_res - theta_s*T_res)^2]
    # S2 = sigma_2 * E[alpha_s^2] = sigma_2 * E[(T_res/nu_2)^2] = sigma_2 / nu_2
    S2 = sigma_2 / nu_2
    S = np.sqrt(S2)
 
    neyman_orthogonal_score_outcome = residualized_outcome_second_stage**2 - sigma_2
    neyman_orthogonal_score_treatment = t_resid**2 - nu_2
    neyman_orthogonal_score_theta = residualized_outcome_second_stage * t_resid / nu_2

    # Now code for benchmarking using covariates begins
    # R^2 of outcome with observed common causes and treatment
    r2t_w = np.var(base_t) / np.var(T)
    # R^2 of outcome explained by treatment + observed confounders:
    # g_s = Y - (Y_res - theta_s*T_res) = base_y + theta_s * t_resid
    g_s = base_y + theta_s * t_resid
    r2y_tw = np.var(g_s) / np.var(Y)

    if benchmarking:
        delta_r2_y_wj, delta_r2t_wj  = compute_r2diff_benchmarking_covariates(features=features,
            T=T,
            Y=Y,
            benchmark_common_causes=benchmark_common_causes,
            split_indices=split_indices,
            algo=algo,
            params_t=params_t,
            params_y=params_y,
            r2t_w=r2t_w,
            r2y_tw=r2y_tw
            )

        # Partial R^2 of outcome after regressing over unobserved confounder, observed common causes and treatment
        # Assuming that the difference in R2 is the same for wj and new unobserved confounder
        delta_r2y_u = frac_strength_outcome * delta_r2_y_wj
        delta_r2t_u = frac_strength_treatment * delta_r2t_wj

        r2yu_tw = delta_r2y_u / (1 - r2y_tw)  # partial R2 for outcome
        r2tu_w = delta_r2t_u / (1 - r2t_w)

        # wrt unobserved confounders in partial-linear models
        #r2tu_w = frac_strength_treatment * (1 - ratio_var_alpha_wj)
        if r2tu_w < 0:
            r2tu_w = 1e-8
            logger.warning(
                "Warning: r2tu_w computed as negative (likely due to estimation noise in var_alpha_wj/var_alpha_s "
                "on a small sample). Setting r2tu_w to 0."
            )

        if r2yu_tw >= 1:
            r2yu_tw = 1 - 1e-8
            logger.warning(
                "Warning: r2yu_tw can not be > 1. Try a lower effect_fraction_on_outcome. Setting r2yu_tw to 1"
            )
        if r2tu_w >= 1:
            r2tu_w = 1 - 1e-8
            logger.warning("r2tu_w > 1 (try a lower frac_strength_treatment); clipped to 1.")

        if r2yu_tw < 0:
            r2yu_tw = 1e-8
            logger.warning(
                "Warning: r2yu_tw computed as negative (delta_r2_y_wj < 0, i.e. the reduced-feature outcome "
                "model out-performed the full model out-of-fold). Setting r2yu_tw to 0."
            )
        #no check for r2tu_w if <0
    else:
         r2yu_tw = frac_strength_outcome
         r2tu_w = frac_strength_treatment

    benchmarking_results = perform_benchmarking(
         r2yu_tw=r2yu_tw,
         r2tu_w=r2tu_w,
         significance_level=significance_level,
         S2= S2, 
         S=S,
         theta_s = theta_s,
         neyman_orthogonal_score_theta = neyman_orthogonal_score_theta, 
         sigma_2 = sigma_2, 
         neyman_orthogonal_score_treatment= neyman_orthogonal_score_treatment, 
         nu_2 = nu_2
    )
    results = pd.DataFrame(benchmarking_results, index=[0])
    RV, plot_type = calculate_robustness_value(alpha=None, S2=S2, S=S, theta_s=theta_s, neyman_orthogonal_score_theta = neyman_orthogonal_score_theta, 
                sigma_2=sigma_2, neyman_orthogonal_score_treatment= neyman_orthogonal_score_treatment, nu_2 = nu_2)
    RV_alpha, _ = calculate_robustness_value(alpha=significance_level, S2=S2, S=S, theta_s=theta_s, neyman_orthogonal_score_theta = neyman_orthogonal_score_theta, 
                sigma_2=sigma_2, neyman_orthogonal_score_treatment= neyman_orthogonal_score_treatment, nu_2 = nu_2)
    if plot == True:
        effect_strength_outcome = None
        effect_strength_treatment = None
        plot_contour(benchmarking, significance_level, frac_strength_outcome, frac_strength_treatment, effect_strength_outcome,
             effect_strength_treatment, S2, S, theta_s, benchmark_common_causes, results, neyman_orthogonal_score_theta, sigma_2, 
             neyman_orthogonal_score_treatment, nu_2, r2tu_w, r2yu_tw, plot_type=plot_type)


    s = "Sensitivity Analysis to Unobserved Confounding using partial R^2 parameterization\n\n"
    s += "Original Effect Estimate : {0}\n".format(theta_s)
    s += "Robustness Value : {0}\n\n".format(RV)
    s += "Robustness Value (alpha={0}) : {1}\n\n".format(significance_level, RV_alpha)
    s += "Interpretation of results :\n"
    s += "Any confounder explaining less than {0}% percent of the residual variance of both the treatment and the outcome would not be strong enough to explain away the observed effect i.e bring down the estimate to 0 \n\n".format(
         round(RV * 100, 2)
    )
    s += "For a significance level of {0}%, any confounder explaining more than {1}% percent of the residual variance of both the treatment and the outcome would be strong enough to make the estimated effect not 'statistically significant'\n\n".format(
            significance_level * 100, round(RV_alpha * 100, 2)
    )
    print(s)
    return RV, RV_alpha, benchmarking_results

def calculate_robustness_value(alpha, S2, S, theta_s, neyman_orthogonal_score_theta, sigma_2, neyman_orthogonal_score_treatment, nu_2):
        """
        Function to compute the robustness value of estimate against the confounders
        :param alpha: confidence interval for statistical inference

        :returns: robustness value
        """
        for t_val in np.arange(0, 1, 0.01):
           
            lower_confidence_bound, upper_confidence_bound, _ = get_confidence_levels(
                r2yu_tw=t_val, r2tu_w=t_val, significance_level=alpha, S2=S2, S=S, theta_s=theta_s, neyman_orthogonal_score_theta = neyman_orthogonal_score_theta, 
                sigma_2 = sigma_2, neyman_orthogonal_score_treatment= neyman_orthogonal_score_treatment, nu_2 = nu_2
            )
            if theta_s >= 0:
                if lower_confidence_bound <= 0:
                    plot_type = "lower_confidence_bound"
                    return t_val, plot_type
            else:
                if upper_confidence_bound >= 0:
                    plot_type = "upper_confidence_bound"
                    return t_val, plot_type
        return t_val

def get_confidence_levels(r2yu_tw, r2tu_w, significance_level, S2, S, theta_s, neyman_orthogonal_score_theta, sigma_2, neyman_orthogonal_score_treatment, nu_2):
        """
        Returns lower and upper bounds for the effect estimate, given different explanatory powers of unobserved confounders. It uses the following definitions.

        Y_residual  = Y - E[Y | X, T] (residualized outcome)
        T_residual  = T - E[T | X] (residualized treatment)
        theta = E[(Y - E[Y | X, T)(T - E[T | X] )] / E[(T - E[T | X]) ^ 2]
        σ² = E[(Y - E[Y | X, T]) ^ 2] (expected value of residual outcome)
        ν^2 = E[(T - E[T | X])^2] (expected value of residual treatment)
        ψ_θ = m(Ws , g) + (Y - g(Ws))α(Ws) - θ
        ψ_σ² = (Y - g(Ws)) ^ 2 - σ²
        ψ_ν2 = (2m(Ws, α ) - α^2) - ν^2

        :param r2yu_tw: proportion of residual variance in the outcome explained by confounders
        :param r2tu_w: proportion of residual variance in the treatment explained by confounders
        :param significance_level: confidence interval for statistical inference(default = 0.05)
        :param is_partial_linear: whether the data-generating process is assumed to be partially linear

        :returns lower_confidence_bound: lower limit of confidence bound of the estimate
        :returns upper_confidence_bound: upper limit of confidence bound of the estimate
        :returns bias: omitted variable bias for the confounding scenario
        """

        Cg2 = r2yu_tw  # Strength of confounding that omitted variables generate in outcome regression
        
        # Strength of confounding that omitted variables generate in treatment regression
        Calpha2 = r2tu_w / (1 - r2tu_w)
       
        Cg = np.sqrt(Cg2)
        Calpha = np.sqrt(Calpha2)
        
        S = np.sqrt(S2)

        # computing the point estimate for the bounds
        bound = S2 * Cg2 * Calpha2
        bias = np.sqrt(bound)
        theta_lower = theta_s - bias
        theta_upper = theta_s + bias

        if significance_level is not None:
            phi_lower, phi_upper = get_phi_lower_upper(Cg, Calpha, S, neyman_orthogonal_score_theta, sigma_2, neyman_orthogonal_score_treatment, nu_2)

            expected_phi_lower = np.mean(phi_lower * phi_lower)
            expected_phi_upper = np.mean(phi_upper * phi_upper)

            n1 = phi_lower.shape[0]
            n2 = phi_upper.shape[0]

            stddev_lower = np.sqrt(expected_phi_lower / n1)
            stddev_upper = np.sqrt(expected_phi_upper / n2)
            probability = scipy.stats.norm.ppf(1 - significance_level)
            lower_confidence_bound = theta_lower - probability * np.sqrt(
                np.mean(stddev_lower * stddev_lower) + np.var(theta_lower)
            )
            upper_confidence_bound = theta_upper + probability * np.sqrt(
                np.mean(stddev_upper * stddev_upper) + np.var(theta_upper)
            )

        else:
            lower_confidence_bound = theta_lower
            upper_confidence_bound = theta_upper

        return lower_confidence_bound, upper_confidence_bound, bias

def get_phi_lower_upper(Cg, Calpha, S, neyman_orthogonal_score_theta, sigma_2, neyman_orthogonal_score_treatment, nu_2):
        """
        Calculate lower and upper influence function (phi)

        :param Cg: measure of strength of confounding that omitted variables generate in outcome regression
        :param Calpha: measure of strength of confounding that omitted variables generate in treatment regression

        :returns : lower bound of phi, upper bound of phi
        """

        bounds_estimator_upper = neyman_orthogonal_score_theta + ((Cg * Calpha) / (2 * S)) * (
            sigma_2 * neyman_orthogonal_score_treatment + nu_2 * neyman_orthogonal_score_treatment
        )
        bounds_estimator_lower = neyman_orthogonal_score_theta - ((Cg * Calpha) / (2 * S)) * (
            sigma_2 * neyman_orthogonal_score_treatment + nu_2 * neyman_orthogonal_score_treatment
        )

        return bounds_estimator_lower, bounds_estimator_upper



def compute_r2diff_benchmarking_covariates(features, T, Y, benchmark_common_causes, split_indices, algo, params_t, params_y, r2t_w, r2y_tw):
    """
    Change in partial R^2 for treatment and outcome when the benchmark
    covariate(s) are dropped from the confounder set W, using the same
    algo/params already selected for cate_linear's nuisance models — no
    independent grid search, no Reisz representer.
 
    :returns: delta_r2y_wj, delta_r2t_wj
    """
    W_j_df = features.drop(columns=benchmark_common_causes)
    W_j = W_j_df.to_numpy()
 
    r2t_w_j = get_regression_r2(X=W_j, Y=T, split_indices=split_indices, algo=algo, params=params_t)
    delta_r2t_wj = r2t_w - r2t_w_j
 
    # Outcome side: fit Y ~ (T, W_j) jointly (flexible model, same algo as
    # your selected outcome model), mirroring the treatment-side comparison.
    T_W_j = np.hstack([T.reshape(-1, 1), W_j])
    r2y_w_j = get_regression_r2(X=T_W_j, Y=Y, split_indices=split_indices, algo=algo, params=params_y)
    delta_r2y_wj = r2y_tw - r2y_w_j
 
    return delta_r2y_wj, delta_r2t_wj


def perform_benchmarking(r2yu_tw, r2tu_w, significance_level, S2, S, theta_s, neyman_orthogonal_score_theta, sigma_2, neyman_orthogonal_score_treatment, nu_2):
        """
        :param r2yu_tw: proportion of residual variance in the outcome explained by confounders
        :param r2tu_w: proportion of residual variance in the treatment explained by confounders
        :param significance_level: the desired significance level for the bounds
        :param is_partial_linear: whether we assume a partially linear data-generating process


        :returns: python dictionary storing values of r2tu_w, r2yu_tw, short estimate, bias, lower_ate_bound,upper_ate_bound, lower_confidence_bound, upper_confidence_bound
        """
        #max_r2yu_tw = max(r2yu_tw) if np.ndim(r2yu_tw) != 0 else r2yu_tw
        #max_r2tu_w = max(r2tu_w) if np.ndim(r2yu_tw) != 0 else r2tu_w
        #breakpoint()
        lower_confidence_bound, upper_confidence_bound, bias = get_confidence_levels(
            r2yu_tw=r2yu_tw,
            r2tu_w=r2tu_w,
            significance_level=significance_level,
            S2=S2,
            S=S,
            theta_s = theta_s,
            neyman_orthogonal_score_theta = neyman_orthogonal_score_theta, 
            sigma_2 = sigma_2, 
            neyman_orthogonal_score_treatment= neyman_orthogonal_score_treatment, 
            nu_2 = nu_2
        )
        lower_ate_bound, upper_ate_bound, bias = get_confidence_levels(
            r2yu_tw=r2yu_tw, r2tu_w=r2tu_w, significance_level=None, S2=S2, S=S, theta_s=theta_s, neyman_orthogonal_score_theta = neyman_orthogonal_score_theta, 
            sigma_2 = sigma_2, neyman_orthogonal_score_treatment= neyman_orthogonal_score_treatment, nu_2 = nu_2
        )

        benchmarking_results = {
            "r2tu_w": r2yu_tw,
            "r2yu_tw": r2tu_w,
            "short estimate": theta_s,
            "bias": bias,
            "lower_ate_bound": lower_ate_bound,
            "upper_ate_bound": upper_ate_bound,
            "lower_confidence_bound": lower_confidence_bound,
            "upper_confidence_bound": upper_confidence_bound,
        }

        return benchmarking_results
def plot_contour(benchmarking, significance_level, frac_strength_outcome, frac_strength_treatment, effect_strength_outcome, effect_strength_treatment, 
         S2, S, theta_s, benchmark_common_causes, results, neyman_orthogonal_score_theta, sigma_2, neyman_orthogonal_score_treatment, nu_2, r2tu_w, r2yu_tw, plot_type="lower_confidence_bound"):
        """
        Plots and summarizes the sensitivity bounds as a contour plot, as they vary with the partial R^2 of the unobserved confounder(s) with the treatment and the outcome
        Two types of plots can be generated, based on adjusted estimates or adjusted t-values
        X-axis: Partial R^2 of treatment and unobserved confounder(s)
        Y-axis: Partial R^2 of outcome and unobserved confounder(s)
        We also plot bounds on the partial R^2 of the unobserved confounders obtained from observed covariates

        :param plot_type: possible values are 'bias','lower_ate_bound','upper_ate_bound','lower_confidence_bound','upper_confidence_bound'
        """
        r2tu_w_point = r2tu_w
        r2yu_tw_point = r2yu_tw

        critical_value = 0
        num_points_per_contour = 30
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.set_title("Sensitivity contour plot of %s" % plot_type)
        ax.set_xlabel("Partial R^2 of unobserved confounder with treatment")
        ax.set_ylabel("Partial R^2 of unobserved confounder with outcome")
        #breakpoint()
        if effect_strength_treatment is None:
            # adding 1.1 as plotting margin  ensure that the benchmarked part is shown fully in plot
            x_limit = (1.1 * r2tu_w_point) if benchmarking else 0.99
            r2tu_w = np.arange(0.0, x_limit, x_limit / num_points_per_contour)
        else:
            x_limit = max(r2tu_w)
            r2tu_w = r2tu_w
        if effect_strength_outcome is None:
            # adding 1.1 as plotting margin  ensure that the benchmarked part is shown fully in plot
            y_limit = (1.1 * r2yu_tw) if benchmarking else 0.99
            r2yu_tw = np.arange(0.0, y_limit, y_limit / num_points_per_contour)
        else:
            y_limit = r2yu_tw[-1]
            r2yu_tw = r2yu_tw
        ax.set_xlim(-x_limit / 20, x_limit)
        ax.set_ylim(-y_limit / 20, y_limit)

        undjusted_estimates = None
        contour_values = np.zeros((len(r2yu_tw), len(r2tu_w)))

        for i in range(len(r2yu_tw)):
            y = r2yu_tw[i]
            for j in range(len(r2tu_w)):
                x = r2tu_w[j]
                benchmarking_results = perform_benchmarking(
                    r2yu_tw=y,
                    r2tu_w=x,
                    significance_level=significance_level,
                    S2=S2, 
                    S=S,
                    theta_s = theta_s,
                    neyman_orthogonal_score_theta = neyman_orthogonal_score_theta, 
                    sigma_2 = sigma_2, 
                    neyman_orthogonal_score_treatment= neyman_orthogonal_score_treatment, 
                    nu_2 = nu_2
                )
                contour_values[i][j] = benchmarking_results[plot_type]

        contour_plot = ax.contour(
            r2tu_w,
            r2yu_tw,
            contour_values,
            colors="blue",
            linewidths=0.75,
            linestyles="solid",
        )
        ax.clabel(contour_plot, inline=1, fontsize=9, colors="black")

        if critical_value >= contour_values.min() and critical_value <= contour_values.max() and plot_type != "bias":
            contour_plot = ax.contour(
                r2tu_w,
                r2yu_tw,
                contour_values,
                colors="red",
                linewidths=0.75,
                levels=[critical_value],
            )
            ax.clabel(contour_plot, [critical_value], inline=1, fontsize=9, colors="red")

        # Adding unadjusted point estimate
        if (
            plot_type == "lower_confidence_bound"
            or plot_type == "upper_confidence_bound"
            or plot_type == "lower_ate_bound"
            or plot_type == "upper_ate_bound"
        ):
            ax.scatter(
                [0],
                [0],
                marker="D",
                color="black",
                label="Unadjusted({:1.2f})".format(theta_s),
            )

        # Adding bounds to partial R^2 values for given strength of confounders
        if benchmarking:
            if frac_strength_treatment == frac_strength_outcome:
                signs = str(round(frac_strength_treatment, 2))
            else:
                signs = str(round(frac_strength_treatment, 2)) + "/" + str(round(frac_strength_outcome, 2))
            label = signs + " X " + str(benchmark_common_causes) + " ({:1.2f}) ".format(results[plot_type][0])
            ax.scatter(
                r2tu_w_point, r2yu_tw_point, color="red", marker="^", label=label
            )

        plt.margins()
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()


# def estimate_CATE_general(df, feeder, conf_t, conf_y, treatment, outcome, type="nonlinear"):
#     window_size = 10000
#     step_size = 1000
#     n_iterations = 1  # Number of bootstraps in each window

#     # Sort the DataFrame by trend
#     y='outcome'
#     df[y] = df[outcome]
#     df = df.sort_values(by=treatment) #destroys any temporal information here

#     # Parameters for sliding window
#     #Analysis is performed on overlapping windows to observe how effect changes over time
#     # Store CATE and corresponding penetration levels for each window
#     results = []
    
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#     master = pd.read_csv(f"results/causality/{dataset}/master_robustness_results.csv", index_col='algo')
#     best = master['total'].idxmin()
#     params_t = master.loc[best, 'params_t']
#     params_y =master.loc[best, 'params_y']

#     # Total number of windows
#     total_windows = (len(df) - window_size) // step_size + 1

#     # Sliding window analysis
#     for start in range(0, len(df) - window_size + 1, step_size): #overlapping by 5000-500 = 4500 (0-4999, 500-5499)
#         window_data = df.iloc[start:start + window_size]

#         # Calculate the mean trend for the current window
#         mean_t = window_data[treatment].mean()

#         # Use tqdm for bootstrap iterations to estimate causal effect across different sample (build causal distribution)
#         #Capture non-stationary effects
#         with tqdm(total=n_iterations, desc=f'Window {start // step_size + 1}/{total_windows}', leave=False) as pbar:
#             # Bootstrap for the current window
#             for _ in range(n_iterations):
#                 random_subset = window_data.sample(n=min(len(window_data), window_size), replace=True)
#                 df_residualized = residualize_data(random_subset, y, conf_t, conf_y, treatment, 5, best, params_t=params_t, params_y=params_y)
#                 res = fit_residualized_model(df_residualized, conf_y)
#                 # Initialize the analyzer
                
#                 analyser = non_parametric_sensitivity_analysis(
#                     num_splits=5,
#                     shuffle_data=True,
#                     shuffle_random_seed=42,
#                     benchmark_common_causes=["month"],
#                     significance_level=0.05,
#                     frac_strength_treatment=0.5,
#                     frac_strength_outcome=0.5,
#                     g_s_estimator_list = None,
#                     g_s_estimator_param_list = None,
#                     alpha_s_estimator_list = None,
#                     alpha_s_estimator_param_list = None,
#                     theta_s=res['cate_linear'],
#                     observed_common_causes = random_subset[['wind_speed', 'humidity', 'clear_sky_ghi', 'rainfall', 'month', 'Holiday', 'hour']],
#                     outcome=random_subset[outcome],
#                     treatment=random_subset[treatment],
#                     plugin_reisz=False
    
#                 )
                
#                 print(analyser)
#                 """
#                 analyzer = NonParametricSensitivityAnalyzer(
#                    estimator=res['cate_linear'],
#                     num_splits=5,
#                     shuffle_data=True,
#                     shuffle_random_seed=42,
#                     benchmark_common_causes=["month", "clear_sky_ghi"],
#                     significance_level=0.05,
#                     frac_strength_treatment=0.5,
#                     frac_strength_outcome=0.5,
#                     effect_fraction_on_treatment = 0.2,
#                     effect_fraction_on_outcome = 0.2,
#                     theta_s=res['cate_linear'],
#                     observed_common_causes = random_subset[['wind_speed', 'humidity', 'clear_sky_ghi', 'rainfall', 'month', 'Holiday', 'hour']],
#                     outcome=random_subset[outcome],
#                     treatment=random_subset[treatment],
#                     plugin_reisz=False

    
#                 )
#                 # Perform sensitivity analysis
#                 analyzer.check_sensitivity(plot=True)
#                 # Access results
#                 print(analyzer)
#                 print(analyzer.results)
#                 """
#                 # Append the mean trend value and CATE to results
#                 results.append({
#                     'mean_t': mean_t, #observe CATE across time 
#                     'cate_linear': res['cate_linear'],  # Extract the first (and only) coefficient since there is only one feature (temp)
                   
#                 })
#                 pbar.update(1)  # Update the progress bar

    

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(f"results/causality/results_{feeder}.csv", index=False)

    
#     return results_df


# if __name__ == "__main__":
#     # Example usage of NonParametricSensitivityAnalyzer
#     # Load your dataset here
#     import os
#     import sys
#     base_directory = os.getcwd()
#     print(base_directory)
#     dataset = "LCL" #or LCL

#     causal_path = base_directory + f"/data/causal_{dataset}.csv"

#     size=158
#     # Load data
#     df = pd.read_csv(causal_path)
#     df['month'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.month
#     df['day'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.dayofweek 
#     df['hour'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour
#     conf_t, conf_y = ['wind_speed', 'humidity', 'clear_sky_ghi', 'rainfall', 'month', 'Holiday', 'hour'], ['wind_speed', 'humidity', 'clear_sky_ghi', 'rainfall', 'month', 'Holiday', 'hour']
    
#     outcome = ['Aggregate']
#     treatment = ['temperature']

#     estimate_CATE_general(df, "nps", conf_t, conf_y, treatment, outcome, type="nonlinear")

    