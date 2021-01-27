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


def rirs_aic(model_fit):
    log_like = model_fit.summary().tables[0][3][3]
    features = model_fit.fe_params.index
    return -2 * float(log_like) + 2 * (len(features) - 1)


# TODO: Finish this and r_sq_conditional
# def r_sq_marginal(model, data):
#     response = model_fit.model.formula.split()[0]


def max_pcor(feature_list, target_feature, covar, data):
    cors_table = pd.DataFrame()
    for feature in feature_list:
        par_cor = pg.partial_corr(data, x=feature, y=target_feature, covar=covar)
        cors_table[feature] = par_cor["r"]

    cors_table = cors_table.T.squeeze()

    max_feature = cors_table.abs().idxmax()
    max_val = cors_table[max_feature]

    return max_feature, max_val


def mlm_vif(model_fit, data):
    features = list(model_fit.fe_params.index[1:])
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


def mlm_backward_step(
    data, response, features, groups, rand_slope=False, sig_level=0.05, verbose=1
):
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
            re_formula = "~ {}".format(" + ".join(remaining)) if rand_slope else None
            formula = "{} ~ {}".format(response, " + ".join(remaining))
            model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
            model_fit = model.fit(reml=False)
        except LinAlgError as error:
            # Sometimes a singular matrix error is raised when fitting the model.
            # For now, to handle it we just go ahead and remove the next highest pval as well and move on
            if "pvals" in locals():
                next_least_sig_feature = pvals.nlargest(count + 2).index[-1]
                next_least_sig_val = pvals.nlargest(count + 2)[-1]
                if verbose >= 1:
                    print(
                        f"\nCaught a LinAlgError: singular matrix. Removing {next_least_sig_feature}: {next_least_sig_val}.\n"
                    )
                remaining.remove(next_least_sig_feature)
                count += 1
            else:
                # If LinAlgError is raised on first model
                i = np.random.randint(
                    0, high=len(remaining) - 1
                )  # select a random feature to remove NOTE: this should be temporary
                if verbose >= 1:
                    print(
                        f"\nCaught a LinAlgError on the first model, before pvals can be defined. Removing {remaining[i]}"
                    )
                remaining.remove(remaining[i])

            if verbose >= 2:
                print(remaining, "\n")
            continue

        if verbose >= 2:
            print(f"Adjusted R-squared: {adjusted_r_squared(model_fit, data)}")
            print("================================================\n")

        # Step 3: identify least significant (highest p-value) feature
        count = 0
        pvals = model_fit.summary().tables[1]["P>|z|"]
        pvals = pd.to_numeric(pvals).drop("Intercept").dropna()
        least_sig_feature = pvals.idxmax()
        least_sig_val = pvals.max()

        if verbose >= 2:
            print(pvals)
            print(f"{least_sig_feature}: {least_sig_val}")
            print(f"Adjusted R-squared: {adjusted_r_squared(model_fit, data)}")
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

            model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
            model_fit = model.fit(reml=False)
            break

    return model_fit


def mlm_step_forward(
    data, response, features, groups, rand_slope=False, criterion="aic", verbose=1
):
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
    if direction == "maximise":
        current_score, best_new_score = 0.0, 0.0
    elif direction == "minimise":
        current_score, best_new_score = 10000.0, 10000.0

    idx = 1
    while remaining and current_score == best_new_score:
        print(f"{idx}|", end="\r")  # process count
        idx += 1

        scores_with_candidates = []
        for candidate in remaining:
            re_formula = (
                "~ {}".format(" + ".join(selected + [candidate]))
                if rand_slope
                else None
            )
            formula = "{} ~ {}".format(response, " + ".join(selected + [candidate]))

            try:
                model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
                model_fit = model.fit(reml=False)

            except LinAlgError as error:
                # Sometimes a singular matrix error is raised when fitting the model.
                if verbose >= 1:
                    print(formula)
                    print(
                        f"Caught a LinAlgError: singular matrix. Skipping {candidate}."
                    )
                continue

            if criterion == "adjusted_r_squared":
                score = adjusted_r_squared(model_fit, data)
            elif criterion == "aic":
                score = rirs_aic(model_fit) if rand_slope else model_fit.aic
            elif criterion == "bic":
                if rand_slope:
                    print(
                        "Cannot calculate BIC for Random Slope Random Intercept model yet."
                    )
                else:
                    score = model_fit.bic
            elif criterion == "r_squared":
                score = r_squared(model_fit, data)
            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort()
        if verbose >= 2:
            print("\nscores_with_candidates: ", scores_with_candidates)
            print("================================================\n")

        if direction == "maximise":
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score

        elif direction == "minimise":
            best_new_score, best_candidate = scores_with_candidates.pop(0)
            if current_score > best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        if verbose >= 1:
            print(f"selected: {selected}  |  {criterion}:  {current_score}")

    re_formula = "~ {}".format(" + ".join(selected)) if rand_slope else None
    formula = "{} ~ {}".format(response, " + ".join(selected))
    model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
    model_fit = model.fit(reml=False)

    return model_fit


def vif_reduction(
    model_fit, data, response, groups, rand_slope=False, max_vif=10, verbose=1
):
    # VIF Reduction
    vif = mlm_vif(model_fit, data)
    if vif.max() > max_vif:
        if verbose >= 1:
            print(f"\nVIF of some features exceeds stated max VIF ({max_vif}).")
            print(vif)
    else:
        if verbose >= 1:
            print("No VIF issues identified.")
        return model_fit, vif

    while vif.max() > max_vif:
        if verbose >= 1:
            print(f"Removing {vif.idxmax()}: {vif.max()}")

        cutdown_features = list(model_fit.fe_params.index[1:])
        cutdown_features.remove(vif.idxmax())
        re_formula = "~ {}".format(" + ".join(cutdown_features)) if rand_slope else None
        formula = "{} ~ {}".format(response, " + ".join(cutdown_features))

        model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
        model_fit = model.fit(reml=False)
        vif = mlm_vif(model_fit, data)

    return model_fit, vif


#### Full Feature Selection Process ####
def mlm_feature_selection(
    data,
    response,
    feature_lists,
    groups,
    rand_slope=False,
    par_cor_selection=False,
    backward_selection=True,
    forward_selection=True,
    sig_level=0.05,
    criterion="aic",
    check_vif=True,
    max_vif=10,
    verbose=1,
):
    results = {}

    if par_cor_selection:
        partial_corrs = par_cors(data, response, feature_lists, groups)
        if verbose >= 1:
            print(partial_corrs)
        features = list(partial_corrs.keys())
        results["partial_corrs"] = partial_corrs
    else:
        features = sorted({x for v in FEATS_LISTS.values() for x in v})

    if backward_selection is True:
        print("Running Backward step feature selection.")
        t1_start = process_time()  # start a process timer
        model_fit = mlm_backward_step(
            data=data,
            response=response,
            features=features,
            groups=groups,
            rand_slope=rand_slope,
            sig_level=sig_level,
            verbose=verbose,
        )
        model = model_fit.model
        t1_stop = process_time()

        features = list(model_fit.fe_params.index[1:])
        results["back_features"] = features

        if verbose >= 1:
            print("\n")
            summarise_model(model_fit, data, plots=False)
            print(
                f"\n============== {response} Backwards took {t1_stop-t1_start} secs. ================"
            )
        results["back_model"] = model
        results["back_model_fit"] = model_fit

    if forward_selection:
        print("\nRunning Forward step feature selection.")
        t2_start = process_time()
        model_fit = mlm_step_forward(
            data=data,
            response=response,
            features=features,
            groups=groups,
            rand_slope=rand_slope,
            criterion=criterion,
            verbose=verbose,
        )
        model = model_fit.model
        t2_stop = process_time()
        results["forward_features"] = list(model_fit.fe_params.index[1:])

        if verbose >= 1:
            print("\n")
            summarise_model(model_fit, data, plots=False)
            print(
                f"\n============== {response} Forwards took {t2_stop-t2_start} secs. ============\n"
            )
        results["forward_model"] = model
        results["forward_model_fit"] = model_fit

    if check_vif is True:
        # VIF Reduction
        model_fit, vif = vif_reduction(
            model_fit=model_fit,
            data=data,
            response=response,
            groups=groups,
            rand_slope=rand_slope,
            max_vif=max_vif,
            verbose=verbose,
        )
        model = model_fit.model

        if verbose >= 2:
            summarise_model(model_fit, data, plots=False)
        results["vif"] = vif

    results["final_model"] = model
    results["final_model_fit"] = model_fit

    return results


def summarise_model(model_fit, data, response=None, plots=True):
    if response is None:
        response = model_fit.model.endog_names

    print(model_fit.model.formula)
    print(model_fit.summary())
    print("Adj R-sq: ", adjusted_r_squared(model_fit, data))
    print("AIC:      ", model_fit.aic)
    print("BIC:      ", model_fit.bic)
    print(model_fit.random_effects)

    performance = pd.DataFrame()
    performance["residuals"] = model_fit.resid.values
    performance["actual"] = data[response]
    performance["predicted"] = model_fit.predict(data)
    performance["fitted"] = model_fit.fittedvalues

    if plots:
        sns.lmplot(x="fitted", y="residuals", data=performance)
        plt.title(f"Residual vs. Fitted for {response} model")
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.show()

        sns.lmplot(x="fitted", y="actual", data=performance, fit_reg=True)
        plt.title(f"Actual vs. Fitted for {response} model")
        plt.xlabel("Fitted values")
        plt.ylabel("Actual values")
        plt.show()

    print("Adj R-sq: ", adjusted_r_squared(model_fit, data))

    return performance


def predict_rirs(model_fit):
    """Returns the fitted values for the model.

    The fitted values reflect the mean structure specified by the 
    fixed effects and the predicted random effects.

    Parameters
    ----------
    model_fit : [type]
        [description]
    """
    fit = np.dot(model_fit.model.exog, model_fit, fe_params)
    re = model_fit.random_effects
    for group_ix, group in enumerate(model_fit.model.group_labels):
        ix = model_fit.model.row_indices[group]

        mat = []
        if model_fit.model.exog_re_li is not None:
            mat.append(model_fit.model.exog_re_li[group_ix])
        for c in model_fit.model.exog_vc.names:
            if group in model_fit.model.exog_vc.names[c]:
                mat.append(model_fit.model.exog_vc.names[c][group])
        mat = np.concatenate(mat, axis=1)

        fit[ix] += np.dot(mat, re[group])

    return fit


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv(
        "C:\\Users\\Andrew\\OneDrive - University College London\\_PhD\\Papers - Drafts\\J3_JASA_1f-Soundscape-Modelling\\results\\2020-10-09\\LondonVeniceGranadaBINResults_2020-10-09.csv"
    )

    response = "Pleasant"
    groups = "LocationID"

    response_features = par_cors(data, response, FEATS_LISTS, groups)
    features = list(response_features.keys())

    model_fit = mlm_backward_step(data, "Pleasant", features, groups, False,)

