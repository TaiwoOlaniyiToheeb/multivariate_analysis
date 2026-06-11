# Multivariate OLS Regression — Marketing ROI Analysis

**Author:** Olaniyi | **Tool:** Python + statsmodels | **Method:** Multiple Linear Regression (OLS)

## 1. Objective

Model the combined effect of TV spend tier, Radio spend, Social Media spend, and Influencer type on Sales using Ordinary Least Squares (OLS) multiple regression, identify which channels drive incremental ROI, and provide evidence-based budget recommendations.

## 2. Dataset

| Property | Value |
|---|---|
| Rows | 572 |
| Columns | 5 (`TV`, `Radio`, `Social Media`, `Influencer`, `Sales`) |
| Missing values | 0 (no rows dropped) |
| TV tiers | Low, Medium, High |
| Influencer types | Nano, Micro, Macro, Mega |

`Radio` and `Social Media` are continuous numeric spend variables. `TV` and `Influencer` are categorical and required one-hot encoding before modelling.

## 3. Feature Engineering — Encoding

Both categorical variables were one-hot encoded with `pd.get_dummies(..., drop_first=True)` to avoid the dummy variable trap. `drop_first=True` removes the **alphabetically first** category, which produced these effective reference (baseline) categories:

| Variable | Dummy columns retained | Effective reference category |
|---|---|---|
| `TV` | `TV_Low`, `TV_Medium` | **TV = High** |
| `Influencer` | `Influencer_Mega`, `Influencer_Micro`, `Influencer_Nano` | **Influencer = Macro** |

> **Note on documentation vs. implementation:** an earlier section of the notebook states the intended references as TV = Low and Influencer = Nano. Because `drop_first` drops alphabetically (High before Low; Macro before Mega/Micro/Nano), the *actual* dropped categories are High and Macro. All coefficient interpretations below use the actual reference categories (TV = High, Influencer = Macro), since that is what the fitted model reflects.

Final modelling features (7 predictors): `Radio`, `Social Media`, `TV_Low`, `TV_Medium`, `Influencer_Mega`, `Influencer_Micro`, `Influencer_Nano`.

## 4. Exploratory Data Analysis

- Distribution plots for `Radio`, `Social Media`, and `Sales`.
- Boxplots of `Sales` by TV tier and by Influencer type.
- Scatter plots of `Radio` and `Social Media` vs `Sales` with linear trend lines.
- Pearson correlation matrix across all encoded numeric features.

**Correlation with Sales (ranked):**

| Feature | Correlation |
|---|---|
| Radio | +0.8580 |
| Social Media | +0.5420 |
| TV_Medium | +0.0504 |
| Influencer_Mega | +0.0324 |
| Influencer_Nano | +0.0177 |
| Influencer_Micro | −0.0065 |
| TV_Low | −0.8059 |

`Radio` shows a strong positive correlation with Sales, while `TV_Low` shows a strong negative correlation — i.e., being in the Low TV tier is strongly associated with lower Sales (consistent with High-tier TV driving the strongest Sales).

## 5. Model Specification

```
Sales = β₀ + β₁·Radio + β₂·Social_Media + β₃·TV_Low + β₄·TV_Medium
        + β₅·Influencer_Mega + β₆·Influencer_Micro + β₇·Influencer_Nano + ε
```

Fitted with `statsmodels.api.OLS` on 572 observations and 7 predictors plus an intercept.

## 6. OLS Regression Results

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.904
Model:                            OLS   Adj. R-squared:                  0.903
Method:                 Least Squares   F-statistic:                     760.4
Prob (F-statistic):          1.82e-282
No. Observations:                 572   AIC:                             5443.
Df Residuals:                     564   BIC:                             5478.
Df Model:                           7
Covariance Type:            nonrobust
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
const              217.4784      6.584     33.031      0.000     204.546     230.411
Radio                2.9735      0.235     12.644      0.000       2.512       3.435
Social Media        -0.1391      0.676     -0.206      0.837      -1.467       1.189
TV_Low            -154.5736      4.949    -31.231      0.000    -164.295    -144.852
TV_Medium          -75.5947      3.647    -20.726      0.000     -82.759     -68.431
Influencer_Mega      2.4948      3.462      0.721      0.471      -4.305       9.295
Influencer_Micro     2.9391      3.378      0.870      0.385      -3.695       9.574
Influencer_Nano      0.8015      3.346      0.240      0.811      -5.770       7.373
==============================================================================
Omnibus:                       58.711   Durbin-Watson:                   1.874
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               17.802
Skew:                           0.057   Prob(JB):                     0.000136
Kurtosis:                       2.143   Cond. No.                         147.
==============================================================================
```

## 7. Multicollinearity Check — Variance Inflation Factor (VIF)

| Feature | VIF | Status |
|---|---|---|
| TV_Low | 4.08 | ✅ OK |
| Radio | 3.48 | ✅ OK |
| TV_Medium | 2.23 | ✅ OK |
| Social Media | 1.67 | ✅ OK |
| Influencer_Nano | 1.63 | ✅ OK |
| Influencer_Micro | 1.62 | ✅ OK |
| Influencer_Mega | 1.59 | ✅ OK |

All VIF values are well below the conservative threshold of 5, indicating no problematic multicollinearity among predictors.

## 8. Coefficient Interpretation Table

| Predictor | Coefficient | Std Error | p-value | 95% CI | Significant (p<0.05) |
|---|---|---|---|---|---|
| Radio | 2.9735 | 0.2352 | 0.0000 | [2.5116, 3.4355] | ✅ Yes |
| Social Media | −0.1391 | 0.6761 | 0.8371 | [−1.4670, 1.1889] | ❌ No |
| TV_Low | −154.5736 | 4.9494 | 0.0000 | [−164.2952, −144.8520] | ✅ Yes |
| TV_Medium | −75.5947 | 3.6473 | 0.0000 | [−82.7586, −68.4307] | ✅ Yes |
| Influencer_Mega | 2.4948 | 3.4620 | 0.4714 | [−4.3052, 9.2948] | ❌ No |
| Influencer_Micro | 2.9391 | 3.3777 | 0.3846 | [−3.6953, 9.5735] | ❌ No |
| Influencer_Nano | 0.8015 | 3.3457 | 0.8107 | [−5.7699, 7.3730] | ❌ No |

### Interpretation (relative to baselines TV = High, Influencer = Macro)

- **Radio (+2.9735, p < 0.001):** Each additional unit of Radio spend is associated with a ~2.97 unit increase in Sales, holding all else constant. ≈ $2,974 in Sales per $1,000 of Radio spend. Statistically significant.
- **Social Media (−0.1391, p = 0.837):** Effect is essentially zero and not statistically distinguishable from no effect once the other channels are controlled for.
- **TV_Low (−154.5736, p < 0.001):** Moving from High-tier TV to Low-tier TV is associated with a ~154.6 unit drop in Sales, holding other predictors constant. Highly significant — TV tier is the strongest single driver in the model.
- **TV_Medium (−75.5947, p < 0.001):** Moving from High-tier TV to Medium-tier TV is associated with a ~75.6 unit drop in Sales. Also highly significant, but roughly half the penalty of dropping to Low tier — confirming a clear ordering of High > Medium > Low.
- **Influencer_Mega / Micro / Nano:** None of the influencer-tier coefficients are statistically significant (all p > 0.05). Compared to the Macro baseline, no influencer tier shows a reliably different effect on Sales once TV tier and channel spend are accounted for.

## 9. Final Fitted Equation

```
Sales = 217.4784
        + 2.9735 · Radio
        − 0.1391 · Social_Media
        − 154.5736 · TV_Low
        − 75.5947 · TV_Medium
        + 2.4948 · Influencer_Mega
        + 2.9391 · Influencer_Micro
        + 0.8015 · Influencer_Nano
```

(TV_Low and TV_Medium are 1/0 dummies, with TV = High as the omitted baseline; Influencer_Mega/Micro/Nano are 1/0 dummies, with Influencer = Macro as the omitted baseline.)

## 10. Regression Assumption Diagnostics

| # | Assumption | Test | Result |
|---|---|---|---|
| 1 | Linearity | Residuals vs Fitted + LOWESS smoother | Visual check via plot — LOWESS line tracks near zero across the fitted range |
| 2 | Normality of residuals | Q-Q plot + Shapiro-Wilk | Shapiro-Wilk: W = 0.9834, p = 0.0000 → ⚠️ residuals deviate from normality (large sample size makes this test very sensitive to minor departures) |
| 3 | Homoscedasticity | Scale-Location + Breusch-Pagan | Breusch-Pagan: LM = 8.5329, p = 0.2879 → ✅ no evidence of heteroscedasticity |
| 4 | No severe multicollinearity | VIF | All VIF < 5 → ✅ passes |
| 5 | Joint significance | F-test | F = 760.38, p = 1.82e-282 → ✅ model is highly statistically significant |

**Overall model fit:**
- R² = 0.9042 → the model explains **90.4%** of the variance in Sales.
- Adjusted R² = 0.9030 (penalised for 7 predictors), confirming the fit is not driven by overfitting from the number of predictors.

## 11. Business Recommendations

1. **Prioritise High-tier TV campaigns.** TV tier produces by far the largest coefficients in the model (−154.6 for Low vs High, −75.6 for Medium vs High). The drop from High to Low tier corresponds to the single largest swing in predicted Sales of any variable in the dataset.
2. **Continue investing in Radio.** Radio has a statistically significant, positive, and stable per-unit return (~$2,974 in Sales per $1,000 spend), the only continuous channel with a reliable positive effect.
3. **Re-evaluate Social Media spend.** Once TV tier and Radio are controlled for, Social Media spend shows no significant incremental effect on Sales (p = 0.837). This doesn't mean Social Media is worthless — it may be acting earlier in the funnel (awareness) — but its direct, isolated contribution to Sales in this dataset is not detectable.
4. **De-prioritise influencer tier as a standalone lever.** None of the Mega, Micro, or Nano tiers differ significantly from the Macro baseline. Influencer choice does not appear to be a strong independent driver of Sales in this model; if influencer marketing is retained, future analysis should test interaction effects (e.g., influencer tier × TV tier) rather than treating it as an additive term.
5. **Follow-up analysis:** given the significant Shapiro-Wilk result, consider examining residuals for outliers/leverage points and testing transformations (e.g., log-Sales) or interaction terms to improve normality without sacrificing the strong R² already achieved.

## 12. Reproducibility

```bash
pip install -r requirements.txt
```

Key packages used: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scipy`, `scikit-learn`.
