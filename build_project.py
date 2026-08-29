from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MODEL_DIR = ROOT / "model"
HN_MONTHLY = PROCESSED / "hn_seniority_monthly.csv"
for folder in (PROCESSED, FIGURES, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

START, END = "2022-11-01", "2025-05-31"


def format_month_axis(ax, interval=4):
    """Mostra um subconjunto legível dos rótulos YYYY-MM nos eixos de data."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def load_monthly_data():
    genai = pd.read_csv(RAW / "GenAI_posting.csv", parse_dates=["date"])
    sector = pd.read_csv(RAW / "job_postings_by_sector_US.csv", parse_dates=["date"])
    total = pd.read_csv(RAW / "aggregate_job_postings_US.csv", parse_dates=["date"])
    raw_frames = {
        "GenAI_posting.csv": genai, "job_postings_by_sector_US.csv": sector,
        "aggregate_job_postings_US.csv": total}
    raw_info = {
        name: {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "duplicate_rows": int(frame.duplicated().sum()),
            "missing_cells": int(frame.isna().sum().sum()),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        }
        for name, frame in raw_frames.items()
    }

    genai = genai.loc[(genai.jobcountry == "US") & genai.date.between(START, END), ["date", "GenAI_share_postings"]].copy()
    software = sector.loc[(sector.jobcountry == "US") & (sector.display_name == "Software Development") &
                          (sector.variable == "total postings") & sector.date.between(START, END),
                          ["date", "indeed_job_postings_index"]].copy()
    total = total.loc[(total.jobcountry == "US") & (total.variable == "total postings") &
                      total.date.between(START, END), ["date", "indeed_job_postings_index_SA"]].copy()
    for frame in (genai, software, total):
        frame["month"] = frame.date.dt.to_period("M")

    monthly = (genai.groupby("month", as_index=False).GenAI_share_postings.mean()
               .merge(software.groupby("month", as_index=False).indeed_job_postings_index.mean(), on="month")
               .merge(total.groupby("month", as_index=False).indeed_job_postings_index_SA.mean(), on="month")
               .rename(columns={"GenAI_share_postings": "genai_share_pct",
                                "indeed_job_postings_index": "software_postings_index",
                                "indeed_job_postings_index_SA": "total_postings_index_sa"})
               .sort_values("month").reset_index(drop=True))
    monthly["genai_share_pct"] = monthly["genai_share_pct"] * 100
    monthly["month_start"] = monthly.month.dt.to_timestamp()
    monthly["month_label"] = monthly.month.astype(str)
    monthly["time_index"] = np.arange(len(monthly))
    q1, q3 = monthly.software_postings_index.quantile([0.25, 0.75])
    iqr = q3 - q1
    response_outliers = monthly.loc[
        ~monthly.software_postings_index.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr),
        "month",
    ].astype(str).tolist()
    audit = {
        "raw_files": raw_info,
        "filtered_daily_rows": {
            "GenAI": int(len(genai)),
            "Software Development": int(len(software)),
            "Todos os anúncios": int(len(total)),
        },
        "monthly_dataset": {
            "rows": int(len(monthly)),
            "columns": 6,
            "duplicate_months": int(monthly.month.duplicated().sum()),
            "missing_cells": int(monthly.isna().sum().sum()),
            "response_iqr_outlier_months": response_outliers,
            "outliers_removed": 0,
        },
    }
    monthly = monthly[["month_start", "month_label", "time_index", "genai_share_pct",
                       "software_postings_index", "total_postings_index_sa"]]
    monthly.to_csv(PROCESSED / "monthly_regression_dataset.csv", index=False)
    (PROCESSED / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return monthly, audit


def metric_row(name, y_true, pred):
    return {"model": name, "MAE": mean_absolute_error(y_true, pred),
            "RMSE": mean_squared_error(y_true, pred) ** 0.5, "R2": r2_score(y_true, pred)}


def model_summary(model):
    conf = model.conf_int()
    return {"n": int(model.nobs), "r2": float(model.rsquared), "adj_r2": float(model.rsquared_adj),
            "params": {k: float(v) for k, v in model.params.items()},
            "pvalues_hac": {k: float(v) for k, v in model.pvalues.items()},
            "ci95_hac": {k: [float(conf.loc[k, 0]), float(conf.loc[k, 1])] for k in model.params.index}}


def fit_analysis(monthly):
    simple_features = ["genai_share_pct"]
    controlled_features = ["genai_share_pct", "total_postings_index_sa"]
    cut = int(len(monthly) * 0.70)
    train, test = monthly.iloc[:cut], monthly.iloc[cut:]
    simple = LinearRegression().fit(train[simple_features], train.software_postings_index)
    controlled = LinearRegression().fit(train[controlled_features], train.software_postings_index)
    polynomial = Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                           ("linear", LinearRegression())]).fit(train[controlled_features], train.software_postings_index)
    y_test = test.software_postings_index
    test_predictions = test[["month_start", "month_label", "software_postings_index"]].copy()
    test_predictions = test_predictions.rename(columns={"software_postings_index": "actual"})
    test_predictions["Reference mean"] = train.software_postings_index.mean()
    test_predictions["Simple linear"] = simple.predict(test[simple_features])
    test_predictions["Controlled linear"] = controlled.predict(test[controlled_features])
    test_predictions["Polynomial degree 2"] = polynomial.predict(test[controlled_features])
    test_predictions["controlled_residual"] = (
        test_predictions["actual"] - test_predictions["Controlled linear"]
    )
    comparison = pd.DataFrame([
        metric_row("Reference mean", y_test, test_predictions["Reference mean"]),
        metric_row("Simple linear", y_test, test_predictions["Simple linear"]),
        metric_row("Controlled linear", y_test, test_predictions["Controlled linear"]),
        metric_row("Polynomial degree 2", y_test, test_predictions["Polynomial degree 2"]),
    ])
    comparison.to_csv(PROCESSED / "model_comparison.csv", index=False)
    test_predictions.to_csv(PROCESSED / "test_predictions.csv", index=False)

    y = monthly.software_postings_index
    level_simple = sm.OLS(y, sm.add_constant(monthly[simple_features])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    level_controlled = sm.OLS(y, sm.add_constant(monthly[controlled_features])).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    delta = monthly[["software_postings_index", *controlled_features]].diff().dropna()
    difference = sm.OLS(delta.software_postings_index, sm.add_constant(delta[controlled_features])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1})
    inference = {"simple_levels": model_summary(level_simple), "controlled_levels": model_summary(level_controlled),
                 "monthly_changes_sensitivity": model_summary(difference)}
    (PROCESSED / "inference_results.json").write_text(json.dumps(inference, indent=2), encoding="utf-8")

    train_ols = sm.OLS(
        train.software_postings_index,
        sm.add_constant(train[controlled_features]),
    ).fit()
    vif_design = sm.add_constant(train[controlled_features])
    vif = {
        column: float(variance_inflation_factor(vif_design.values, position))
        for position, column in enumerate(vif_design.columns)
        if column != "const"
    }
    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(
        train_ols.resid, train_ols.model.exog
    )
    jb = stats.jarque_bera(train_ols.resid)
    cooks = train_ols.get_influence().cooks_distance[0]
    cooks_threshold = 4 / len(train)
    influential = [
        {"month": train.iloc[position].month_label, "cooks_distance": float(value)}
        for position, value in enumerate(cooks)
        if value > cooks_threshold
    ]
    diagnostics = {
        "scope": "Resíduos de treinamento da especificação linear controlada selecionada",
        "correlations": train[["software_postings_index", *controlled_features]].corr().to_dict(),
        "vif": vif,
        "durbin_watson": float(durbin_watson(train_ols.resid)),
        "breusch_pagan": {
            "lm_statistic": float(bp_lm), "lm_pvalue": float(bp_lm_pvalue),
            "f_statistic": float(bp_f), "f_pvalue": float(bp_f_pvalue),
        },
        "jarque_bera": {"statistic": float(jb.statistic), "pvalue": float(jb.pvalue)},
        "cooks_threshold_4_over_n": float(cooks_threshold),
        "influential_observations": influential,
        "max_cooks_distance": float(cooks.max()),
    }
    (PROCESSED / "diagnostic_summary.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    deployment_model = LinearRegression().fit(monthly[controlled_features], y)
    joblib.dump(deployment_model, MODEL_DIR / "controlled_linear_model.pkl")
    metadata = {
        "features": controlled_features,
        "target": "software_postings_index",
        "coverage": {"start": START, "end": END},
        "selection_train_observations": len(train),
        "selection_test_observations": len(test),
        "deployment_fit_observations": len(monthly),
        "selected_model": "Controlled linear",
        "feature_ranges": {
            feature: [float(monthly[feature].min()), float(monthly[feature].max())]
            for feature in controlled_features
        },
        "target_range": [float(y.min()), float(y.max())],
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {"comparison": comparison, "cut": cut, "train": train, "test": test,
            "simple_model": simple, "controlled_model": controlled, "polynomial_model": polynomial,
            "level_simple": level_simple, "level_controlled": level_controlled, "difference": difference,
            "deployment_model": deployment_model, "train_ols": train_ols,
            "test_predictions": test_predictions, "diagnostics": diagnostics,
            "cooks_distance": cooks, "monthly": monthly, "inference": inference}


def make_figures(result):
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 220, "axes.titleweight": "bold"})
    d, coral, teal, navy = result["monthly"], "#e76f51", "#0f766e", "#16324f"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)
    axes[0].plot(d.month_start, d.genai_share_pct, color=coral, lw=2.5, marker="o", ms=3)
    axes[0].set_ylabel("Proporção de GenAI (%)")
    axes[0].set_title("As referências a GenAI subiram enquanto os anúncios de desenvolvimento de software caíram")
    axes[1].plot(d.month_start, d.software_postings_index, color=teal, lw=2.5, label="Desenvolvimento de software")
    axes[1].plot(d.month_start, d.total_postings_index_sa, color=navy, lw=2, label="Todos os anúncios")
    axes[1].set_ylabel("Índice do Indeed (1º/02/2020 = 100)")
    axes[1].set_xlabel("Mês")
    axes[1].legend(frameon=False, ncol=2)
    format_month_axis(axes[1], interval=4)
    fig.tight_layout(); fig.savefig(FIGURES / "01_trends.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.1))
    sc = ax.scatter(d.genai_share_pct, d.software_postings_index, c=d.time_index, cmap="viridis", s=48, edgecolor="white")
    x = np.linspace(d.genai_share_pct.min(), d.genai_share_pct.max(), 100)
    m = result["level_simple"]
    ax.plot(x, m.params["const"] + m.params["genai_share_pct"] * x, color=coral, lw=2.3, label="Linha OLS")
    ax.set_title("Associação simples em nível: prevalência de GenAI versus anúncios de software")
    ax.set_xlabel("Proporção de anúncios relacionados a GenAI (%)"); ax.set_ylabel("Índice de anúncios de desenvolvimento de software")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02); cbar.set_label("Meses desde nov. de 2022")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIGURES / "02_scatter_levels.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].hist(d.software_postings_index, bins=8, color=teal, edgecolor="white")
    axes[0].set_title("Distribuição da resposta")
    axes[0].set_xlabel("Índice de anúncios de software")
    axes[0].set_ylabel("Meses")
    axes[1].boxplot(d.software_postings_index, vert=True)
    axes[1].set_title("Boxplot da resposta")
    axes[1].set_ylabel("Índice de anúncios de software")
    corr = d[["software_postings_index", "genai_share_pct", "total_postings_index_sa"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=axes[2])
    axes[2].set_title("Correlações quantitativas")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_eda_summary.png", bbox_inches="tight")
    plt.close(fig)

    pred = result["test_predictions"]
    fitted, resid = result["train_ols"].fittedvalues, result["train_ols"].resid
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    axes[0, 0].plot(pred.month_start, pred.actual, marker="o", label="Real", color=navy)
    axes[0, 0].plot(pred.month_start, pred["Controlled linear"], marker="o", label="Previsto", color=coral)
    axes[0, 0].set_title("Real versus previsto fora da amostra")
    format_month_axis(axes[0, 0], interval=2)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].scatter(pred.actual, pred["Controlled linear"], color=teal, s=38)
    limits = [min(pred.actual.min(), pred["Controlled linear"].min()),
              max(pred.actual.max(), pred["Controlled linear"].max())]
    axes[0, 1].plot(limits, limits, "--", color=coral)
    axes[0, 1].set_title("Real versus previsto fora da amostra")
    axes[0, 1].set_xlabel("Índice real")
    axes[0, 1].set_ylabel("Índice previsto")
    axes[0, 2].scatter(fitted, resid, color=navy, s=34)
    axes[0, 2].axhline(0, color=coral, ls="--")
    axes[0, 2].set_title("Resíduos de treinamento versus ajustados")
    axes[0, 2].set_xlabel("Índice ajustado")
    axes[0, 2].set_ylabel("Resíduo")
    axes[1, 0].hist(resid, bins=7, color=teal, alpha=.85, edgecolor="white")
    axes[1, 0].set_title("Distribuição dos resíduos de treinamento")
    axes[1, 0].set_xlabel("Resíduo")
    axes[1, 0].set_ylabel("Contagem")
    stats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Gráfico Q-Q dos resíduos de treinamento")
    axes[1, 1].set_xlabel("Quantis teóricos")
    axes[1, 1].set_ylabel("Valores ordenados")
    axes[1, 2].stem(result["train"].month_start, result["cooks_distance"], basefmt=" ")
    axes[1, 2].axhline(4 / len(result["train"]), color=coral, ls="--", label="4/n")
    axes[1, 2].set_title("Distância de Cook")
    format_month_axis(axes[1, 2], interval=4)
    axes[1, 2].legend(frameon=False)
    fig.suptitle("Modelo linear controlado selecionado: avaliação e diagnósticos", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_diagnostics.png", bbox_inches="tight")
    plt.close(fig)

    if HN_MONTHLY.exists():
        hn = pd.read_csv(HN_MONTHLY, parse_dates=["month_start"])
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(
            hn.month_start, hn.software_posts, color=navy, lw=2.2,
            label="Publicações de empregadores relacionadas a software",
        )
        axes[0].plot(
            hn.month_start, hn.junior_intern_mentions, color=coral, lw=2,
            marker="o", ms=3, label="Menções de início de carreira/estágio",
        )
        axes[0].plot(
            hn.month_start, hn.senior_mentions, color=teal, lw=2,
            marker="o", ms=3, label="Menções sênior",
        )
        axes[0].set_title("Hacker News Who Is Hiring: menções de senioridade nas publicações de software")
        axes[0].set_ylabel("Publicações de empregadores")
        axes[0].legend(frameon=False, ncol=3)
        axes[1].plot(
            hn.month_start, hn.junior_intern_share_3m * 100,
            color=coral, lw=2.3, label="Proporção de início de carreira/estágio (média de 3 meses)",
        )
        axes[1].plot(
            hn.month_start, hn.senior_share_3m * 100,
            color=teal, lw=2.3, label="Proporção sênior (média de 3 meses)",
        )
        axes[1].set_ylabel("Proporção (%)")
        axes[1].set_xlabel("Mês da discussão Who Is Hiring")
        axes[1].legend(frameon=False)
        format_month_axis(axes[1], interval=4)
        fig.tight_layout()
        fig.savefig(FIGURES / "05_hn_seniority.png", bbox_inches="tight")
        plt.close(fig)

        combined = result["monthly"].merge(hn, on="month_start", how="left")
        combined.to_csv(PROCESSED / "combined_indeed_hn_monthly.csv", index=False)


def main():
    monthly, data_audit = load_monthly_data()
    result = fit_analysis(monthly)
    make_figures(result)
    hn_summary = None
    if HN_MONTHLY.exists():
        hn = pd.read_csv(HN_MONTHLY)
        hn_summary = {
            "rows": int(len(hn)),
            "software_posts": int(hn.software_posts.sum()),
            "junior_intern_mentions": int(hn.junior_intern_mentions.sum()),
            "senior_mentions": int(hn.senior_mentions.sum()),
        }
    print(json.dumps({"data_audit": data_audit, "monthly_shape": list(monthly.shape),
                      "comparison": result["comparison"].to_dict("records"),
                      "diagnostics": result["diagnostics"],
                      "hn_seniority": hn_summary,
                      "inference": result["inference"]}, indent=2))


if __name__ == "__main__":
    main()
