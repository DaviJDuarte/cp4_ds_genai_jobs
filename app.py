import json
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
DATA = PROCESSED / "monthly_regression_dataset.csv"
COMPARISON = PROCESSED / "model_comparison.csv"
TEST_PREDICTIONS = PROCESSED / "test_predictions.csv"
DIAGNOSTICS = PROCESSED / "diagnostic_summary.json"
HN_MONTHLY = PROCESSED / "hn_seniority_monthly.csv"
HN_SUMMARY = PROCESSED / "hn_seniority_summary.json"
MODEL = ROOT / "model" / "controlled_linear_model.pkl"
METADATA = ROOT / "model" / "metadata.json"


def format_month_axis(ax, interval=4):
    """Mantém os rótulos dos meses legíveis em gráficos estreitos do Streamlit."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

st.set_page_config(page_title="GenAI e anúncios de software", page_icon="AI", layout="wide")
st.title("Prevalência de GenAI e anúncios de vagas de desenvolvimento de software")
st.caption("Dados mensais do Indeed Hiring Lab | Estados Unidos | associação observacional, não causalidade")
st.markdown(
    "Fontes: [Indeed AI Tracker](https://github.com/hiring-lab/ai-tracker) e "
    "[Indeed Job Postings Tracker](https://github.com/hiring-lab/job_postings_tracker). "
    "A variável resposta é um índice, não uma contagem bruta de vagas únicas."
)


@st.cache_data
def load_artifacts():
    data = pd.read_csv(DATA, parse_dates=["month_start"])
    comparison = pd.read_csv(COMPARISON)
    predictions = pd.read_csv(TEST_PREDICTIONS, parse_dates=["month_start"])
    diagnostics = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    hn_monthly = pd.read_csv(HN_MONTHLY, parse_dates=["month_start"])
    hn_summary = json.loads(HN_SUMMARY.read_text(encoding="utf-8"))
    return data, comparison, predictions, diagnostics, metadata, hn_monthly, hn_summary


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


d, comparison, test_predictions, diagnostics, metadata, hn, hn_summary = load_artifacts()
model = load_model()
final_metrics = comparison.loc[comparison.model == metadata["selected_model"]].iloc[0]
comparison_display = comparison.rename(
    columns={"model": "Modelo", "MAE": "MAE fora da amostra", "RMSE": "RMSE fora da amostra", "R2": "R² fora da amostra"}
).replace(
    {"Modelo": {
        "Reference mean": "Média de referência",
        "Simple linear": "Linear simples",
        "Controlled linear": "Linear controlado",
        "Polynomial degree 2": "Polinomial de grau 2",
    }}
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observações mensais", len(d))
c2.metric("MAE fora da amostra", f"{final_metrics.MAE:.2f}")
c3.metric("RMSE fora da amostra", f"{final_metrics.RMSE:.2f}")
c4.metric("R² fora da amostra", f"{final_metrics.R2:.3f}")
st.info(
    "X₁ = percentual de anúncios que mencionam termos de GenAI; X₂ = índice dessazonalizado de todos os "
    "anúncios; Y = índice de anúncios de desenvolvimento de software (1º de fevereiro de 2020 = 100)."
)

overview_tab, explore_tab, model_tab, hn_tab, predict_tab = st.tabs(
    ["Visão geral", "Exploração", "Avaliação do modelo", "Senioridade no HN", "Previsão ilustrativa"]
)

with overview_tab:
    st.subheader("Pergunta e desenho analítico")
    st.write(
        "A prevalência de linguagem relacionada à GenAI ajuda a explicar a variação no índice de anúncios "
        "de desenvolvimento de software do Indeed, depois de considerar as condições gerais de contratação?"
    )
    st.write(
        "Os 21 primeiros meses selecionam e avaliam os modelos; os 10 meses finais formam um conjunto de "
        "teste cronológico. Após a seleção, a especificação linear controlada é reajustada com os 31 meses "
        "para o aplicativo ilustrativo. Não é um modelo causal nem uma previsão do mercado de trabalho."
    )
    st.subheader("Amostra de dados")
    st.dataframe(
        d[["month_start", "genai_share_pct", "software_postings_index", "total_postings_index_sa"]].rename(
            columns={
                "month_start": "Início do mês",
                "genai_share_pct": "Proporção de GenAI (%)",
                "software_postings_index": "Índice de anúncios de software",
                "total_postings_index_sa": "Índice de todos os anúncios (dessazonalizado)",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Estatísticas descritivas")
    descriptive = d[["genai_share_pct", "software_postings_index", "total_postings_index_sa"]].describe().T.rename(
        index={
            "genai_share_pct": "Proporção de GenAI (%)",
            "software_postings_index": "Índice de anúncios de software",
            "total_postings_index_sa": "Índice de todos os anúncios (dessazonalizado)",
        },
        columns={
            "count": "Contagem",
            "mean": "Média",
            "std": "Desvio-padrão",
            "min": "Mínimo",
            "25%": "25%",
            "50%": "Mediana",
            "75%": "75%",
            "max": "Máximo",
        },
    )
    st.dataframe(descriptive.style.format("{:.2f}"), width="stretch")

with explore_tab:
    st.subheader("Gráfico exploratório 1: tendências mensais")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(d.month_start, d.genai_share_pct, color="#e76f51", marker="o", ms=3)
    axes[0].set_ylabel("Proporção de GenAI (%)")
    axes[0].set_title("Prevalência de GenAI")
    axes[1].plot(d.month_start, d.software_postings_index, color="#0f766e", label="Desenvolvimento de software")
    axes[1].plot(d.month_start, d.total_postings_index_sa, color="#16324f", label="Todos os anúncios")
    axes[1].set_ylabel("Índice de anúncios do Indeed")
    axes[1].set_xlabel("Mês")
    axes[1].legend(frameon=False)
    format_month_axis(axes[1], interval=4)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader("Gráfico exploratório 2: distribuição e associação da resposta")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(d.software_postings_index, bins=8, color="#0f766e", edgecolor="white")
    axes[0].set_title("Distribuição do índice de anúncios de software")
    axes[0].set_xlabel("Índice")
    axes[0].set_ylabel("Meses")
    scatter = axes[1].scatter(
        d.genai_share_pct,
        d.software_postings_index,
        c=d.time_index,
        cmap="viridis",
        edgecolor="white",
    )
    axes[1].set_title("Proporção de GenAI versus anúncios de software")
    axes[1].set_xlabel("Proporção de GenAI (%)")
    axes[1].set_ylabel("Índice de anúncios de software")
    fig.colorbar(scatter, ax=axes[1], label="Meses desde nov. de 2022")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.write(
        "A forte ordenação temporal no gráfico de dispersão e a queda nos dois índices de anúncios indicam "
        "risco de tendência comum. Por isso, os resultados de sensibilidade controlados e em primeira "
        "diferença devem ser considerados junto à regressão bruta."
    )

with model_tab:
    st.subheader("Comparação cronológica do conjunto de teste")
    st.dataframe(
        comparison_display.style.format(
            {"MAE fora da amostra": "{:.2f}", "RMSE fora da amostra": "{:.2f}", "R² fora da amostra": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "Todos os valores de R² no teste são negativos porque os dez últimos meses têm pouca variação na "
        "resposta. O modelo controlado foi selecionado por ter o menor MAE e RMSE fora da amostra, mas seu "
        "desempenho preditivo absoluto continua frágil."
    )

    st.subheader("Valores reais versus previstos fora da amostra")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(test_predictions.month_start, test_predictions.actual, marker="o", label="Real", color="#16324f")
    axes[0].plot(test_predictions.month_start, test_predictions["Controlled linear"], marker="o", label="Previsto", color="#e76f51")
    axes[0].set_ylabel("Índice de anúncios de software")
    axes[0].set_xlabel("Mês de teste")
    format_month_axis(axes[0], interval=2)
    axes[0].legend(frameon=False)
    axes[1].scatter(test_predictions.actual, test_predictions["Controlled linear"], color="#0f766e")
    limits = [
        min(test_predictions.actual.min(), test_predictions["Controlled linear"].min()),
        max(test_predictions.actual.max(), test_predictions["Controlled linear"].max()),
    ]
    axes[1].plot(limits, limits, "--", color="#e76f51")
    axes[1].set_xlabel("Índice real")
    axes[1].set_ylabel("Índice previsto")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader("Resíduos fora da amostra")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.scatter(test_predictions["Controlled linear"], test_predictions.controlled_residual, color="#16324f")
    ax.axhline(0, color="#e76f51", linestyle="--")
    ax.set_xlabel("Índice previsto de anúncios de software")
    ax.set_ylabel("Real − previsto")
    ax.set_title("Resíduos versus valores previstos no período de teste")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader("Equação final de implantação e diagnósticos")
    coefficient_table = pd.DataFrame(
        {
            "Termo": ["Intercepto", "Proporção de GenAI (ponto percentual)", "Ponto do índice de todos os anúncios"],
            "Coeficiente": [model.intercept_, *model.coef_],
        }
    )
    st.dataframe(coefficient_table.style.format({"Coeficiente": "{:.3f}"}), hide_index=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("GenAI VIF", f"{diagnostics['vif']['genai_share_pct']:.2f}")
    d2.metric("VIF do controle", f"{diagnostics['vif']['total_postings_index_sa']:.2f}")
    d3.metric("Durbin–Watson", f"{diagnostics['durbin_watson']:.2f}")
    d4.metric("Distância máxima de Cook", f"{diagnostics['max_cooks_distance']:.2f}")
    st.write(
        "Os diagnósticos de VIF e correlação serial indicam que os dois preditores de tendência estão "
        "fortemente relacionados e que a independência dos resíduos é imperfeita. Consulte o notebook "
        "para os resultados de Breusch–Pagan, Jarque–Bera, distância de Cook, HAC e primeira diferença."
    )

with hn_tab:
    st.subheader("Hacker News Who Is Hiring: menções de início de carreira/estágio versus sênior")
    st.markdown(
        "As discussões mensais vêm da [API de busca do HN](https://hn.algolia.com/api), cuja estrutura de "
        "itens é documentada pela [API oficial do Hacker News](https://github.com/HackerNews/API). Cada "
        "comentário de nível superior é tratado como uma publicação de empregador; um comentário pode anunciar várias funções."
    )
    st.info(
        "Esta é uma visão descritiva de uma comunidade de tecnologia autoselecionada. As palavras-chave "
        "indicam se uma publicação de empregador relacionada a software menciona um grupo de senioridade, "
        "não o número de vagas distintas. Publicações mistas são contadas nas duas séries de menções."
    )

    early = hn.head(6)
    late = hn.tail(6)
    early_junior_count = early.junior_intern_mentions.mean()
    late_junior_count = late.junior_intern_mentions.mean()
    early_senior_count = early.senior_mentions.mean()
    late_senior_count = late.senior_mentions.mean()
    junior_count_change = (late_junior_count / early_junior_count - 1) * 100
    senior_count_change = (late_senior_count / early_senior_count - 1) * 100
    junior_share_change = (late.junior_intern_share.mean() - early.junior_intern_share.mean()) * 100
    senior_share_change = (late.senior_share.mean() - early.senior_share.mean()) * 100

    h1, h2, h3, h4 = st.columns(4)
    h1.metric(
        "Publicações de início de carreira/estágio por mês",
        f"{late_junior_count:.1f}",
        f"{junior_count_change:.1f}% vs. os 6 primeiros meses",
    )
    h2.metric(
        "Publicações sênior por mês",
        f"{late_senior_count:.1f}",
        f"{senior_count_change:.1f}% vs. os 6 primeiros meses",
    )
    h3.metric(
        "Proporção de início de carreira/estágio",
        f"{late.junior_intern_share.mean() * 100:.2f}%",
        f"{junior_share_change:+.2f} pontos percentuais",
    )
    h4.metric(
        "Proporção sênior",
        f"{late.senior_share.mean() * 100:.2f}%",
        f"{senior_share_change:+.2f} pontos percentuais",
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(hn.month_start, hn.software_posts, color="#16324f", lw=2.2, label="Publicações de software")
    axes[0].plot(hn.month_start, hn.senior_mentions, color="#0f766e", lw=2, label="Menções sênior")
    axes[0].plot(hn.month_start, hn.junior_intern_mentions, color="#e76f51", lw=2, label="Menções de início de carreira/estágio")
    axes[0].set_ylabel("Publicações de empregadores")
    axes[0].set_title("Contagens mensais")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(
        hn.month_start, hn.senior_share_3m * 100,
        color="#0f766e", lw=2.3, label="Proporção sênior (média de 3 meses)",
    )
    axes[1].plot(
        hn.month_start, hn.junior_intern_share_3m * 100,
        color="#e76f51", lw=2.3, label="Proporção de início de carreira/estágio (média de 3 meses)",
    )
    axes[1].set_ylabel("Proporção das publicações de software (%)")
    axes[1].set_xlabel("Mês da discussão Who Is Hiring")
    axes[1].set_title("Menções de senioridade normalizadas pelo volume de publicações de software")
    axes[1].legend(frameon=False)
    format_month_axis(axes[1], interval=4)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.write(
        f"Dos seis primeiros aos seis últimos meses, as menções de início de carreira/estágio caíram "
        f"{abs(junior_count_change):.1f}% em contagem mensal e sua proporção ficou praticamente estável "
        f"({junior_share_change:+.2f} pontos percentuais). As menções sênior também caíram em contagem "
        f"({abs(senior_count_change):.1f}%), mas sua proporção aumentou {senior_share_change:+.2f} pontos "
        "percentuais. Nesta amostra do HN, o volume geral de publicações diminuiu, enquanto a composição "
        "continuou fortemente voltada a posições sênior."
    )
    st.dataframe(
        hn[[
            "month_start", "software_posts", "junior_intern_mentions", "senior_mentions",
            "mixed_mentions", "junior_intern_share", "senior_share",
        ]].rename(
            columns={
                "month_start": "Início do mês",
                "software_posts": "Publicações de software",
                "junior_intern_mentions": "Menções de início de carreira/estágio",
                "senior_mentions": "Menções sênior",
                "mixed_mentions": "Menções mistas",
                "junior_intern_share": "Proporção de início de carreira/estágio",
                "senior_share": "Proporção sênior",
            }
        ).style.format(
            {"Proporção de início de carreira/estágio": "{:.2%}", "Proporção sênior": "{:.2%}"}
        ),
        width="stretch",
        hide_index=True,
    )

with predict_tab:
    st.subheader("Previsão condicional ilustrativa")
    st.write(
        "Os controles abaixo usam o mesmo artefato de modelo da amostra completa, verificado no notebook. "
        "Use apenas combinações historicamente plausíveis."
    )
    x_min, x_max = metadata["feature_ranges"]["genai_share_pct"]
    t_min, t_max = metadata["feature_ranges"]["total_postings_index_sa"]
    with st.form("prediction_form"):
        ai = st.number_input(
            "Proporção de GenAI nos anúncios (%)",
            min_value=0.0,
            value=float(d.genai_share_pct.median()),
            step=0.1,
        )
        total = st.number_input(
            "Índice de todos os anúncios (1º de fevereiro de 2020 = 100)",
            min_value=0.0,
            value=float(d.total_postings_index_sa.median()),
            step=1.0,
        )
        submitted = st.form_submit_button("Calcular previsão ilustrativa")

    if submitted:
        outside_marginal_range = ai < x_min or ai > x_max or total < t_min or total > t_max
        if outside_marginal_range:
            st.warning(
                f"Pelo menos uma entrada está fora dos intervalos observados: GenAI {x_min:.2f}%–{x_max:.2f}%; "
                f"índice de todos os anúncios {t_min:.1f}–{t_max:.1f}."
            )
        scale = d[["genai_share_pct", "total_postings_index_sa"]].std().replace(0, 1)
        distances = np.sqrt(
            ((d.genai_share_pct - ai) / scale.genai_share_pct) ** 2
            + ((d.total_postings_index_sa - total) / scale.total_postings_index_sa) ** 2
        )
        if distances.min() > 0.75:
            st.warning(
                "Embora cada valor possa estar dentro de seu intervalo marginal, esta combinação está distante "
                "das combinações mensais observadas no conjunto de dados e pode ser uma extrapolação conjunta."
            )
        inputs = pd.DataFrame(
            {"genai_share_pct": [ai], "total_postings_index_sa": [total]}
        )
        prediction = model.predict(inputs)[0]
        st.metric("Índice previsto de anúncios de desenvolvimento de software", f"{prediction:.1f} pontos de índice")
        y_min, y_max = metadata["target_range"]
        if prediction < y_min or prediction > y_max:
            st.warning(
                f"A resposta prevista está fora do intervalo observado do índice de software "
                f"({y_min:.1f}–{y_max:.1f}); trate-a como uma extrapolação instável."
            )
        st.caption(
            "Este é um resultado condicional do modelo, não um contrafactual causal, uma previsão de contratação ou uma contagem de vagas."
        )
