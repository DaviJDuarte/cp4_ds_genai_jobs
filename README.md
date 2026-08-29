# Anúncios de vagas de GenAI e desenvolvimento de software

## Pergunta de pesquisa

A prevalência de linguagem relacionada à IA generativa ajuda a explicar a variação do índice de anúncios de vagas de desenvolvimento de software do Indeed, depois de considerar as condições gerais de contratação? Este é um projeto de associação observacional, não um estudo causal.

O projeto também responde a uma pergunta descritiva: nas publicações de empregadores do **Who Is Hiring?** do Hacker News, as oportunidades para pessoas em início de carreira/estágio cresceram ou diminuíram em relação às oportunidades sênior?

## Dados e método

O projeto usa três arquivos públicos do Indeed Hiring Lab para os Estados Unidos:

- `GenAI_posting.csv`: proporção diária de anúncios que contêm termos de IA generativa. O repositório documenta termos como Generative AI, Large Language Models e ChatGPT. O arquivo está fixado no commit `073a8db`, pois o repositório removeu posteriormente o CSV histórico.
- `job_postings_by_sector_US.csv`: índices diários de anúncios do Indeed por setor. O setor selecionado é `Software Development`, com a variável `total postings`.
- `aggregate_job_postings_US.csv`: índice diário dessazonalizado de anúncios de todo o mercado, selecionado como `total postings`.

Os dados diários são filtrados para os EUA e para o período completo comum de 01/11/2022 a 31/05/2025; depois, são calculadas médias por mês-calendário e feito um inner join pela competência. Os índices do Indeed usam 1º de fevereiro de 2020 = 100. A variável GenAI é uma proporção percentual, não uma contagem.

Após a filtragem, cada série do Indeed contém 943 registros diários alinhados. São três medições nas mesmas datas, e não 2.829 observações independentes. A regressão usa deliberadamente 31 meses-calendário como unidade de observação: a agregação mensal corresponde à pergunta de pesquisa e ao suplemento do HN, reduz o ruído diário e evita tratar dias adjacentes muito semelhantes como evidência totalmente independente. A preferência da atividade por 100 observações, portanto, não é atendida pela amostra da regressão, mas é uma preferência, não um limite obrigatório; a amostra mensal pequena é tratada explicitamente como uma limitação.

O suplemento do Hacker News cobre os mesmos 31 meses. `fetch_hn_who_is_hiring.py` encontra as discussões mensais `Ask HN: Who is hiring?` pela API pública de busca do HN, fornecida pelo Algolia, e baixa seus comentários de nível superior. Cada comentário de nível superior é tratado como uma publicação de empregador, mesmo quando anuncia várias funções. Expressões regulares transparentes identificam:

- Publicações relacionadas a software, usando termos como software developer, backend, frontend, DevOps, SRE, data engineer, ML engineer, platform engineer e mobile engineer.
- Menções a início de carreira/estágio, incluindo junior, intern, new graduate, entry-level, apprentice e termos de early-career.
- Menções a senioridade, incluindo senior, staff, principal, engineering manager, architect, technical lead e exigências de cinco anos ou mais de experiência.

Uma publicação que corresponde simultaneamente aos termos de início de carreira e sênior é rotulada como `mixed` e contribui para ambas as contagens. As proporções mensais dividem cada contagem de menções pelo número de publicações de empregadores relacionadas a software, evitando confundir mudanças no volume geral das discussões com mudanças na composição de senioridade.

## Modelos

- Referência: média do conjunto de treinamento.
- Linear simples: `software_index ~ genai_share`.
- Linear controlado: `software_index ~ genai_share + total_postings_index_sa`.
- Sensibilidade polinomial de grau 2.
- Erros-padrão HAC/Newey-West com uma defasagem mensal para inferência na amostra completa.
- Modelo de primeira diferença para verificar se os resultados em nível são principalmente uma tendência comum.

A seleção dos modelos usa uma divisão cronológica 70/30 (21 meses de treino e 10 meses de teste). A especificação linear controlada é selecionada pelo MAE e RMSE fora da amostra; depois, é ajustada novamente com os 31 meses para o aplicativo ilustrativo. O suplemento do Hacker News é descritivo e não é incluído como preditor causal. Todos os resultados são apenas associações.

## Resultado do Hacker News

Comparando os seis primeiros meses com os seis últimos da janela comum:

- As menções a início de carreira/estágio caíram de **17,5 para 13,0 publicações por mês** (−25,7%), enquanto sua proporção nas publicações de software ficou praticamente estável: **4,80% para 4,69%** (−0,11 ponto percentual).
- As menções sênior caíram de **183,2 para 148,2 publicações por mês** (−19,1%), mas sua proporção aumentou de **50,40% para 53,61%** (+3,21 pontos percentuais).

Assim, o volume de publicações de empregadores de software no HN diminuiu para os dois grupos, enquanto a distribuição continuou fortemente voltada a posições sênior e se deslocou um pouco mais nessa direção. Isso é evidência sobre a linguagem das publicações no HN, não sobre todo o mercado de trabalho de software.

## Instalação e execução

A partir da raiz do repositório:

```powershell
cd cp4/genai_jobs
python -m pip install -r requirements.txt
```

### Executar o notebook

Abra `notebook.ipynb` no PyCharm ou Jupyter, selecione o ambiente em que `requirements.txt` foi instalado e escolha **Run All** na primeira célula. O notebook usa caminhos relativos ao projeto e contém saídas salvas de uma execução limpa bem-sucedida.

### Recriar artefatos gerados

```powershell
python fetch_hn_who_is_hiring.py
python build_all.py
```

A busca do HN só precisa ser executada novamente quando os comentários públicos em cache forem atualizados. `build_all.py` cria os conjuntos de dados processados, auditorias, previsões de teste, comparação de modelos, diagnósticos, figuras e o modelo serializado usado pelo notebook e pelo aplicativo.

### Executar o aplicativo Streamlit

```powershell
python -m streamlit run app.py
```

Se o sistema usar o inicializador `py`, substitua `python` por `py` nesses comandos.

## Estrutura do projeto

- `notebook.ipynb`: análise completa, executada e interpretada.
- `fetch_hn_who_is_hiring.py`: download reproduzível das discussões do HN e classificação de senioridade.
- `build_project.py`: preparação reproduzível dos dados, modelagem, avaliação, diagnósticos, figuras e exportação do modelo.
- `app.py`: exploração no Streamlit, avaliação dos modelos, distribuição de senioridade do HN e aplicativo de previsão ilustrativa.
- `data/raw/`: arquivos de origem baixados.
- `data/processed/`: dados prontos para análise e resultados auditáveis.
- `model/`: modelo linear controlado ajustado à amostra completa e usado pelo notebook e pelo aplicativo.
- `figures/`: figuras reproduzíveis da análise.

## Limitações

A amostra da regressão tem apenas 31 observações mensais; o período de teste cronológico tem pouca variação na resposta; o Indeed é um indicador específico de uma plataforma; menções a palavras-chave não medem adoção real de IA; séries históricas podem ser revisadas; os preditores são tendências fortemente colineares; há meses influentes e variáveis omitidas; e o desenho não permite estabelecer causalidade. O Hacker News também é uma comunidade de tecnologia autoselecionada, e não um portal de vagas representativo. As marcações por palavras-chave do HN podem deixar passar funções, e um comentário pode conter várias vagas. Mais anos de dados, além de juros, demissões no setor de tecnologia, produção de software e dados estruturados de senioridade por vaga, melhorariam a análise.

## Fontes e licença

- https://github.com/hiring-lab/ai-tracker
- https://github.com/hiring-lab/job_postings_tracker
- https://docs.indeed.com/hiring-lab-api/
- https://hiringlab.indeed.com/2024/11/21/growth-in-ai-job-postings-trends-and-surprises/
- https://hn.algolia.com/api
- https://github.com/HackerNews/API

Os repositórios do Indeed Hiring Lab informam que os dados gerados estão disponíveis sob CC BY 4.0. Data de acesso ao Indeed: 27/08/2026. Os comentários do Hacker News foram acessados pela API pública de busca em 28/08/2026 para esta análise educacional.
