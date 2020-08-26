"""
Scripts for 2020 Lockdown Multi-level Modelling

"""

import itertools
import warnings
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.linalg import LinAlgError
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.tools import add_constant

warnings.simplefilter("ignore", ConvergenceWarning)

#### Constants and Labels  ####
FEATS_LISTS = {
    "FS_STATS": ["FS", "FS_5", "FS_10", "FS_50", "FS_90", "FS_95", "FS_Min", "FS_Max"],
    "FS_VARI": ["FS_5_FS_95", "FS_10_FS_90", "FS_Max_FS_Min"],
    "FS_TEMP": ["FS_M0", "FS_nrmse0"],
    "LAeq_STATS": [
        "LAeq",
        "LAeq_5",
        "LAeq_10",
        "LAeq_50",
        "LAeq_90",
        "LAeq_95",
        "LAeq_Max",
        "LAeq_Min",
    ],
    "LAeq_VARI": ["LAeq_10_LAeq_90", "LAeq_5_LAeq_95", "LAeq_Max_LAeq_Min"],
    "LAeq_TEMP": ["LAeq_M0", "LAeq_nrmse0"],
    "LCeq_STATS": [
        "LCeq",
        "LCeq_5",
        "LCeq_10",
        "LCeq_50",
        "LCeq_90",
        "LCeq_95",
        "LCeq_Min",
        "LCeq_Max",
    ],
    "LCeq_VARI": ["LCeq_5_LCeq_95", "LCeq_10_LCeq_90", "LCeq_Max_LCeq_Min"],
    "LCeq_TEMP": ["LCeq_M0", "LCeq_nrmse0"],
    "LZeq_STATS": [
        "LZeq",
        "LZeq_5",
        "LZeq_10",
        "LZeq_50",
        "LZeq_90",
        "LZeq_95",
        "LZeq_Min",
        "LZeq_Max",
    ],
    "LZeq_VARI": ["LZeq_5_LZeq_95", "LZeq_10_LZeq_90", "LZeq_Max_LZeq_Min"],
    "LZeq_TEMP": ["LZeq_M0", "LZeq_nrmse0"],
    "N_STATS": ["N_5", "N_10", "N_50", "N_90", "N_95", "N_Min", "N_Max"],
    "N_VARI": ["N_5_N_95", "N_10_N_90", "N_Max_N_Min"],
    "N_TEMP": ["N_M0", "N_nrmse0", "N_M1", "N_nrmse1", "N_M2", "N_nrmse2"],
    "R_STATS": ["R", "R_5", "R_10", "R_50", "R_90", "R_95", "R_Min", "R_Max"],
    "R_VARI": ["R_5_R_95", "R_10_R_90", "R_Max_R_Min"],
    "R_TEMP": ["R_M0", "R_nrmse0"],
    "S_STATS": ["S", "S_5", "S_10", "S_50", "S_90", "S_95", "S_Min", "S_Max"],
    "S_VARI": ["S_5_S_95", "S_10_S_90", "S_Max_S_Min"],
    "S_TEMP": ["S_M0", "S_nrmse0", "S_M1", "S_nrmse1", "S_M2", "S_nrmse2"],
    "SIL_STATS": [
        "SIL_5",
        "SIL_10",
        "SIL_50",
        "SIL_90",
        "SIL_95",
        "SIL_Min",
        "SIL_Max",
    ],
    "SIL_VARI": ["SIL_5_SIL_95", "SIL_10_SIL_90", "SIL_Max_SIL_Min"],
    "SIL_TEMP": ["SIL_M0", "SIL_nrmse0"],
    "T_STATS": ["T_5", "T_10", "T_50", "T_90", "T_95", "T_Max"],
    "T_VARI": ["T_5_T_95", "T_10_T_90"],
    "T_TEMP": ["T_M0", "T_nrmse0"],
    "I_STATS": ["I", "I_5", "I_10", "I_50", "I_90", "I_95", "I_Min", "I_Max"],
    "I_VARI": ["I_5_I_95", "I_10_I_90", "I_Max_I_Min"],
    "I_TEMP": ["I_M0", "I_nrmse0"],
    "THD_STATS": [
        "THD_5",
        "THD_10",
        "THD_50",
        "THD_90",
        "THD_95",
        "THD_Min",
        "THD_Max",
    ],
    "THD_VARI": ["THD_5_THD_95", "THD_10_THD_90", "THD_Max_THD_Min"],
    "THD_TEMP": ["THD_M0", "THD_nrmse0"],
    "FREQ_FEATS": [
        "SpectralCentroid",
        "MaxFreq",
        "PeakSpectralCentroid",
        "PeakMaxFreq",
    ],
}

ALL_FEATS = sorted({x for v in FEATS_LISTS.values() for x in v})


#### Data Loading ####
def nonlinear_features(data, features, transformations=["log", "square"]):
    new_acoustic_vars = []
    for feature in features:
        new_acoustic_vars.append(feature)
        if feature in [
            "FS_5_FS_95",
            "FS_10_FS_90",
            "FS_Max_FS_Min",
            "LAeq_5_LAeq_90",
            "LAeq_10_LAeq_90",
            "LAeq_Max_LAeq_Min",
            "N_5_N_95",
            "N_10_N_90",
            "N_Max_N_Min",
            "R_5_R_95",
            "R_10_R_90",
            "R_Max_R_Min",
            "S_5_S_95",
            "S_10_S_90",
            "S_Max_S_Min",
            "SIL_5_SIL_95",
            "SIL_10_SIL_90",
            "SIL_Max_SIL_Min",
            "T_5_T_95",
            "T_10_T_90",
            "LZeq_5_LZeq_95",
            "LZeq_10_LZeq_90",
            "LZeq_Max_LZeq_Min",
            "LCeq_5_LCeq_95",
            "LCeq_10_LCeq_90",
            "LCeq_Max_LCeq_Min",
            "THD_5_THD_95",
            "THD_10_THD_90",
            "THD_Max_THD_Min",
        ]:
            continue
        if "THD" in feature:
            continue

        for transform in transformations:
            if transform == "log":
                transform_feature = f"log_{feature}"
                transform_val = np.log(data[feature])
            elif transform == "square":
                transform_feature = f"sq_{feature}"
                transform_val = np.square(data[feature])
            elif transform == "sqrt":
                if any(data[feature]) < 0:
                    break
                else:
                    transform_feature = f"sqrt_{feature}"
                    transform_val = np.sqrt(data[feature])
            else:
                print(f"Transformation not recognised: {transform}")
                continue

            if not transform_val.isnull().values.all():
                data[transform_feature] = transform_val
                new_acoustic_vars.append(transform_feature)

    return data, new_acoustic_vars


#### Stats functions ####
def r_squared(model_fit, data):
    response = model_fit.model.formula.split()[0]
    y = data[response]
    # y_predict = model_fit.predict(data)
    y_predict = model_fit.fittedvalues
    RMSE = np.sqrt(((y - y_predict) ** 2).values.mean())
    return 1.0 - (np.var(y - y_predict) / np.var(y))


def adjusted_r_squared(model_fit, data):
    response = model_fit.model.formula.split()[0]
    y = data[response]
    # y_predict = model_fit.predict(data)
    y_predict = model_fit.fittedvalues
    r_sq = r_squared(model_fit, data)
    return 1 - (1 - r_sq) * (len(y) - 1) / (len(y) - len(model_fit.fe_params) - 1)


def r_sq_marginal(model_fit, data):
    response = model_fit.model.formula.split()[0]


def max_pcor(feature_list, target_feature, covar, data):
    cors_table = pd.DataFrame()
    for feature in feature_list:
        par_cor = pg.partial_corr(data, x=feature, y=target_feature, covar=covar)
        cors_table[feature] = par_cor["r"]

    cors_table = cors_table.T.squeeze()

    max_feature = cors_table.abs().idxmax()
    max_val = cors_table[max_feature]

    return max_feature, max_val


def mlm_vif(model, data):
    features = list(model.fe_params.index[1:])
    X = data[features].dropna()
    X = add_constant(X)
    return pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns,
    )


#### Feature Selection ####
def par_cors(data, response, feature_lists, covar):
    """Determines the max corr feature for all of the given groups of features

    Parameters
    ----------
    data : pd.DataFrame
        [description]
    response : str
        [description]
    feature_lists : list
        list of feature lists
    groups : string
        [description]

    Returns
    -------
    dict
        [description]
    """
    data["group_codes"] = pd.Categorical(data[covar]).codes

    response_features = {}
    for key in feature_lists.keys():
        feature, val = max_pcor(
            feature_list=feature_lists[key],
            target_feature=response,
            covar="group_codes",
            data=data,
        )
        response_features[feature] = val

    return response_features


def mlm_backward_step(data, response, features, groups, sig_level=0.05, verbose=1):
    data = data[
        [response, groups] + features
    ]  # Cut down the dataset to just what is required
    remaining = features.copy()  # Avoid overwriting initial features list

    idx = 1
    while remaining:
        print(f"{idx}|", end="\r")  # process count
        idx += 1

        # Step 2: fit the model with all features
        try:
            init_formula = "{} ~ {}".format(response, " + ".join(remaining))
            init_model = smf.mixedlm(init_formula, data, groups=groups).fit(reml=False)
        except LinAlgError as error:
            # Sometimes a singular matrix error is raised when fitting the model.
            # For now, to handle it we just go ahead and remove the next highest pval as well and move on
            try:
                next_least_sig_feature = pvals.nlargest(count + 2).index[-1]
                next_least_sig_val = pvals.nlargest(count + 2)[-1]
                if verbose >= 1:
                    print(
                        f"Caught a LinAlgError: singular matrix. Removing {next_least_sig_feature}: {next_least_sig_val}.\n"
                    )
                remaining.remove(next_least_sig_feature)
                count += 1
            except UnboundLocalError as error:
                i = np.random.randint(0, high=len(remaining) - 1)
                if verbose >= 1:
                    print(
                        f"Caught a LinAlgError on the first model, before pvals can be defined. Removing {remaining[i]}"
                    )
                remaining.remove(remaining[i])

            if verbose >= 2:
                print(remaining)
                print("\n")
            continue

        # Step 3: identify least significant (highest p-value) feature
        count = 0
        pvals = init_model.pvalues
        pvals = pvals.drop([f"{groups} Var", "Intercept"])
        least_sig_feature = pvals.idxmax()
        least_sig_val = pvals.max()

        if verbose >= 2:
            print(init_formula)
            print(f"{least_sig_feature}: {least_sig_val}")
            print(f"Adjusted R-squared: {adjusted_r_squared(init_model, data)}")
            print("================================================\n")

        # Step 4: ID if least significant feature exceeds significance level
        if least_sig_val > sig_level:
            # Step 5: Remove least significant feature from the set
            remaining.remove(least_sig_feature)
            continue

        if least_sig_val < sig_level:
            # Step 6:
            # if all p-values are better than sig level, select that model
            if len(remaining) == 0:
                return print("No feature meets the selection criterion.")

            model = smf.mixedlm(init_formula, data, groups=groups).fit(reml=False)
            break

    return model


def mlm_step_forward(data, response, features, groups, criterion="aic", verbose=1):
    data = data[
        [response, groups] + features
    ]  # Cut down the dataset to just what is required

    if criterion in ["bic", "aic"]:
        direction = "minimise"
    elif criterion in ["r_squared", "adjusted_r_squared"]:
        direction = "maximise"
    else:
        raise ValueError(f"Criterion type not recognised")

    remaining = features.copy()  # Avoid overwriting initial features list
    selected = []
    if direction == "minimise":
        current_score, best_new_score = 10000.0, 10000.0
    if direction == "maximise":
        current_score, best_new_score = 0.0, 0.0

    idx = 1
    while remaining and current_score == best_new_score:
        print(f"{idx}|", end="\r")  # process count
        idx += 1

        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(response, " + ".join(selected + [candidate]))
            # if verbose is True:
            #     print(formula)

            try:
                model = smf.mixedlm(formula, data, groups=groups).fit(reml=False)

            except LinAlgError as error:
                # Sometimes a singular matrix error is raised when fitting the model.
                if verbose >= 1:
                    print(formula)
                    print(
                        f"Caught a LinAlgError: singular matrix. Skipping {candidate}."
                    )
                continue

            if criterion == "bic":
                score = model.bic
            if criterion == "aic":
                score = model.aic
            if criterion == "r_squared":
                score = r_squared(model, data)
            if criterion == "adjusted_r_squared":
                score = adjusted_r_squared(model, data)

            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort()
        if verbose >= 2:
            print("\nscores_with_candidates: ", scores_with_candidates)
            print("================================================\n")

        if direction == "minimise":
            best_new_score, best_candidate = scores_with_candidates.pop(0)
            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        if direction == "maximise":
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score

        if verbose >= 1:
            print("selected: ", selected)

    formula = "{} ~ {}".format(response, " + ".join(selected))
    model = smf.mixedlm(formula, data, groups=groups).fit(reml=False)

    return model


def vif_reduction(model, data, response, groups, max_vif=10, verbose=1):
    # VIF Reduction
    vif = mlm_vif(model, data)
    if vif.max() > max_vif:
        if verbose >= 1:
            print(
                f"\nWARNING: VIF of some features exceeds stated max VIF ({max_vif})."
            )
            print(vif)
    else:
        if verbose >= 1:
            print("No VIF issues identified.")
        return model, vif

    while vif.max() > max_vif:
        if verbose >= 1:
            print(f"\nRemoving {vif.idxmax()}: {vif.max()}")

        cutdown_features = list(model.fe_params.index[1:])
        cutdown_features.remove(vif.idxmax())
        formula = "{} ~ {}".format(response, " + ".join(cutdown_features))

        model = smf.mixedlm(formula, data, groups=groups).fit(reml=False)
        vif = mlm_vif(model, data)

    return model, vif


#### Full Feature Selection Process ####
def mlm_feature_selection(
    data,
    response,
    features,
    groups,
    backward_selection=True,
    sig_level=0.05,
    criterion="bic",
    check_vif=True,
    max_vif=10,
    verbose=1,
):
    if backward_selection is True:
        print("Running Backward step feature selection.")
        t1_start = process_time()  # start a process timer
        back_model = mlm_backward_step(
            data, response, features, groups, sig_level, verbose
        )
        t1_stop = process_time()

        back_features = list(back_model.fe_params.index[1:])
        if verbose >= 1:
            print("\n")
            print(back_model.summary())
            print("AIC:      ", back_model.aic)
            print("BIC:      ", back_model.bic)

        print(back_model.model.formula)
        print("Adj R-sq: ", adjusted_r_squared(back_model, data))
        print(
            f"\n============== {response} Backwards took {t1_stop-t1_start} secs. ================"
        )
    else:
        init_formula = "{} ~ {}".format(response, " + ".join(features))
        back_model = smf.mixedlm(init_formula, data, groups=groups).fit(reml=False)
        back_features = list(back_model.fe_params.index[1:])

    print("\nRunning Forward step feature selection.")
    t2_start = process_time()
    forward_model = mlm_step_forward(
        data, response, back_features, groups, criterion, verbose
    )
    t2_stop = process_time()
    forward_features = list(forward_model.fe_params.index[1:])

    if verbose >= 1:
        print(forward_model.model.formula)
        print(forward_model.summary())
        print("AIC:      ", forward_model.aic)
        print("BIC:      ", forward_model.bic)
        print(forward_model.random_effects)

    print(forward_model.model.formula)
    print("Adj R-sq: ", adjusted_r_squared(forward_model, data))
    print(
        f"\n============== {response} Forwards took {t2_stop-t2_start} secs. ============\n"
    )

    if check_vif is True:
        # VIF Reduction
        final_model, vif = vif_reduction(
            forward_model, data, response, groups, max_vif, verbose=verbose
        )

        if verbose >= 1:
            print(final_model.model.formula)
            print(final_model.summary())
            print("Adj R-sq: ", adjusted_r_squared(final_model, data))
            print("AIC:      ", final_model.aic)
            print("BIC:      ", final_model.bic)
            print(final_model.random_effects)
    else:
        final_model = forward_model
        vif = mlm_vif(final_model, data)

    return final_model, back_model, forward_model, vif


def summarise_model(model, data, response=None):
    if response is None:
        response = model.model.endog_names

    print(model.model.formula)
    print(model.summary())
    print("Adj R-sq: ", adjusted_r_squared(model, data))
    print("AIC:      ", model.aic)
    print("BIC:      ", model.bic)
    print(model.random_effects)

    performance = pd.DataFrame()
    performance["residuals"] = model.resid.values
    performance["actual"] = data[response]
    # performance["predicted"] = model.predict(data)
    performance["predicted"] = model.fittedvalues

    sns.lmplot(x="predicted", y="residuals", data=performance)
    plt.title(f"Residual vs. Fitted for {response} model")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.show()

    sns.lmplot(x="predicted", y="actual", data=performance, fit_reg=True)
    # plt.scatter(model.fittedvalues, data[response], alpha=0.5)
    plt.title(f"Actual vs. Fitted for {response} model")
    plt.xlabel("Fitted values")
    plt.ylabel("Actual values")
    plt.show()


# Load data
