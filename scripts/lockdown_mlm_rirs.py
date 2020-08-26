"""
Scripts for extending 2020 Multi-level modeling to 
Random Intercepts & Random Slopes model    
"""

import os
import sys

sys.path.append(
    "C:\\Users\\Andrew\\OneDrive - University College London\\_PhD\\Papers - Drafts\\J5_JASA_Lockdown-SS"
)

from scripts import lockdown_mlm as mlm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.linalg import LinAlgError
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#### Stats Functions ####
def r_squared(model_fit, data):
    response = model_fit.model.endog_names
    y = data[response]
    # y_predict = model_fit.predict(data)
    y_predict = model_fit.fittedvalues
    return 1.0 - (np.var(y - y_predict) / np.var(y))


def adjusted_r_squared(model_fit, data):
    response = model_fit.model.endog_names
    y = data[response]
    # y_predict = model_fit.predict(data)
    y_predict = model_fit.fittedvalues
    r_sq = r_squared(model_fit, data)
    return 1 - (1 - r_sq) * (len(y) - 1) / (
        len(y) - len(model_fit.model.exog_names) - 1
    )


def rirs_aic(model_fit):
    log_like = model_fit.summary().tables[0][3][3]
    features = model_fit.fe_params.index
    return -2 * float(log_like) + 2 * (len(features) - 1)


# def rirs_vif(model, data):
#     features = list(model.)


#### Feature Selection ####
def rirs_backward_step(data, response, features, groups, sig_level=0.05, verbose=1):
    data = data[
        [response, groups] + features
    ]  # Cut down the dataset to just what is required
    remaining = features.copy()  # Avoid overwriting intial features list

    idx = 1  # process count
    while remaining:
        print(f"{idx}|", end="\r")  # To track progress through the process
        idx += 1

        # Step 2: fit the model with all remaining features
        re_formula = " ~ {}".format(
            " + ".join(remaining)
        )  # create random slope portion
        formula = "{} {}".format(response, re_formula)
        if verbose >= 2:
            print(formula, "\n")

        try:
            # Fit the model, catch a common error
            model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
            model_fit = model.fit(reml=False)
        except LinAlgError as error:
            # Sometimes a singular matrix error is raised when fitting the model
            # For now, to handle it we just go ahead and remove the next highest pval as well and move on
            if "pvals" in locals():
                next_least_sig_feature = pvals.nlargest(count + 2).index[-1]
                next_least_sig_val = pvals.nlargest(count + 2)[-1]
                if verbose >= 1:
                    print(
                        f"Caught a LinAlgError: singular matrix. Removing {next_least_sig_feature}: {next_least_sig_val}.\n"
                    )
                remaining.remove(next_least_sig_feature)
                count += 1
            else:
                # If LinAlgError is raised on first model
                i = np.random.randint(
                    0, high=len(remaining) - 1
                )  # select a random feature to remove.
                if verbose >= 1:
                    print(
                        f"Caught a LinAlgError on the first model, before pvals can be defined. Removing {remaining[i]}"
                    )
                remaining.remove(remaining[i])

            if verbose >= 2:
                print(remaining, "\n")
            continue

        if verbose >= 1:
            print(f"Adjusted R-squared: {adjusted_r_squared(model_fit, data)}")
            print("================================================\n")

        # Step 3: Identify least significant (highest p-value) feature
        count = 0
        pvals = model_fit.summary().tables[1]["P>|z|"]
        pvals = pd.to_numeric(pvals).drop("Intercept").dropna()
        least_sig_feature = pvals.idxmax()
        least_sig_val = pvals.max()

        if verbose >= 2:
            print(pvals, f"\n{least_sig_feature}: {least_sig_val}")

        # Step 4: ID if least significant feature exceeds significance level
        if least_sig_val > sig_level:
            # Step 5: Remove least significant feature from the set
            remaining.remove(least_sig_feature)
            continue
        if least_sig_val < sig_level:
            # Step 6: If all p-values are better than sig level, select that model
            if len(remaining) == 0:
                return print("No feature meets the selection criterion.")

            model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
            model_fit = model.fit(reml=False)
            break

    return model_fit


def rirs_forward_step(data, response, features, groups, criterion="aic", verbose=1):
    data = data[
        [response, groups] + features
    ]  # Cut down the dataset to just what is required

    if criterion in ["bic", "aic"]:
        direction = "minimise"
    elif criterion in ["r_squared", "adjusted_r_squared"]:
        direction == "maximise"
    else:
        raise ValueError(f"Criterion type not recognised")

    remaining = features.copy()  # avoid overwriting intial features list
    selected = []
    if direction == "maximise":
        current_score, best_new_score = 0.0, 0.0
    elif direction == "minimise":
        current_score, best_new_score = 10000.0, 10000.0

    idx = 1
    while remaining and current_score == best_new_score:
        print(f"{idx}|", end="\r")
        idx += 1

        scores_with_candidates = []
        for candidate in remaining:
            re_formula = "~ {}".format(" + ".join(selected + [candidate]))
            formula = "{} {}".format(response, re_formula)

            try:
                model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
                model_fit = model.fit(reml=False)

            except LinAlgError as error:
                if verbose >= 1:
                    print(formula)
                    print(
                        f"Caught a LinAlgError: singular matrix. Skipping {candidate}."
                    )
                continue

            if criterion == "adjusted_r_squared":
                score = adjusted_r_squared(model, data)
            elif criterion == "aic":
                score = rirs_aic(model_fit)
            elif criterion == "r_squared":
                score = r_squared(model, data)

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
        if verbose >= 2:
            print("selected: ", selected)
            print("\n")

    re_formula = "~ {}".format(" + ".join(selected))
    formula = "{} {}".format(response, re_formula)
    model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
    model_fit = model.fit(reml=False)

    return model_fit


def rirs_vif_reduction(model, data, response, groups, max_vif=10, verbose=1):
    vif = mlm.mlm_vif(model, data)
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

        re_formula = "~ {}".format(" + ".join(cutdown_features))
        formula = "{} {}".format(response, re_formula)

        model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula).fit(
            reml=False
        )
        vif = mlm_vif(model, data)
    return model, vif


def summarise_rirs(model, data, response=None):
    if response is None:
        response = model.model.endog_names

    print(model.model.formula)
    print(model.summary())
    print("Adj R-sq: ", adjusted_r_squared(model, data))
    print("AIC     : ", rirs_aic(model))
    print(model.random_effects)

    performance = pd.DataFrame()
    performance["residuals"] = model.resid.values
    performance["actual"] = data[response]
    performance["predicted"] = model.predict(data)
    performance["fitted"] = model.fittedvalues

    sns.lmplot(x="residuals", y="actual", data=performance, fit_reg=True)
    plt.title(f"Residuals vs Fitted for {response} model")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.show()

    sns.lmplot(x="fitted", y="actual", data=performance, fit_reg=True)
    plt.title(f"Actual vs Fitted for {response} model")
    plt.xlabel("Fitted values")
    plt.ylabel("Actual values")
    plt.show()

    return print("Adj R-sq: ", adjusted_r_squared(model, data))

