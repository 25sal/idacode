import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================
SARCASM_WEIGHT = 0.5

PATH = "data/CulturalDeepfake"
files = [1]
for fileid in files:
    INPUT_FILE = f"{PATH}/{fileid}_uncertainty_sarcasm_report.csv"
    OUTPUT_HTML = f"{PATH}/{fileid}_fake_news_impact_dashboard.html"

    # =====================================================
    # LOAD DATA
    # =====================================================
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["Content", "stance", "ReactionsCount", "SubCommentsCount", "CommentAt"])
    df["CommentAt"] = pd.to_datetime(df["CommentAt"], errors="coerce")
    df = df.sort_values("CommentAt")

    # =====================================================
    # NORMALIZATION & POLARITY
    # =====================================================
    df["ReactionsNorm"] = df["ReactionsCount"] / (df["ReactionsCount"].max() + 1e-6)
    df["SubcommentsNorm"] = df["SubCommentsCount"] / (df["SubCommentsCount"].max() + 1e-6)

    stance_map = {"believes": 1, "criticizes": -1, "neutral": 0}
    df["Polarity"] = df["stance"].map(stance_map)

    # =====================================================
    # IMPACT SCORES
    # =====================================================
    w_r, w_s = 0.5, 0.5
    df["Impact"] = (w_r * df["ReactionsNorm"] + w_s * df["SubcommentsNorm"])
    df["AdjustedImpact"] = df["Impact"] * df["Polarity"] * (1 - df["sarcasm_score"] * SARCASM_WEIGHT)
    df["Date"] = df["CommentAt"].dt.date

    # =====================================================
    # DAILY AGGREGATION
    # =====================================================
    daily_impact = df.groupby("Date")["AdjustedImpact"].sum().reset_index()
    daily_impact["CumulativeImpact"] = daily_impact["AdjustedImpact"].cumsum()

    # =====================================================
    # METRICHE GLOBALI
    # =====================================================
    mean_by_stance = df.groupby("stance")["AdjustedImpact"].mean()
    total_index = df["AdjustedImpact"].sum()

    direction = (
        "🔥 The fake news is **strengthening** over time."
        if total_index > 0
        else "❄️ The fake news is **weakening** over time."
        if total_index < 0
        else "⚖️ Balanced: no clear trend."
    )

    # =====================================================
    # TOP COMMENTS
    # =====================================================
    top_positive = df.sort_values("AdjustedImpact", ascending=False).head(10)
    top_negative = df.sort_values("AdjustedImpact", ascending=True).head(10)

    # =====================================================
    # PLOT 1 – Time evolution
    # =====================================================
    fig1 = px.line(
        daily_impact,
        x="Date",
        y="CumulativeImpact",
        title="📈 Fake News Reinforcement Over Time (adjusted for sarcasm)",
        markers=True,
        template="plotly_white"
    )
    fig1.update_traces(line_color="royalblue")
    fig1.update_layout(yaxis_title="Cumulative Reinforcement Index", xaxis_title="Date")

    # =====================================================
    # PLOT 2 – Boxplot by stance
    # =====================================================
    fig2 = px.box(
        df,
        x="stance",
        y="AdjustedImpact",
        color="stance",
        title="Impact Distribution per Stance (adjusted for sarcasm)",
        template="plotly_white"
    )
    fig2.update_layout(yaxis_title="Adjusted Impact Score", xaxis_title="Stance")

    # =====================================================
    # PLOT 3 – Scatter: Sarcasm vs Impact
    # =====================================================
    fig3 = px.scatter(
        df,
        x="sarcasm_score",
        y="AdjustedImpact",
        color="stance",
        hover_data=["Content"],
        title="Sarcasm vs Adjusted Impact",
        template="plotly_white"
    )
    fig3.update_layout(xaxis_title="Sarcasm Score", yaxis_title="Adjusted Impact")

    # =====================================================
    # HTML TABLES – Top comments
    # =====================================================
    def make_html_table(subdf, title):
        subdf = subdf[["Content", "stance", "ReactionsCount", "SubCommentsCount", "sarcasm_score", "AdjustedImpact"]].copy()
        subdf["Content"] = subdf["Content"].str.slice(0, 150) + "..."
        html = f"<h3>{title}</h3>" + subdf.to_html(
            index=False, classes="display compact", escape=False, border=0
        )
        return html

    table_positive = make_html_table(top_positive, "🔥 Top 10 Comments Reinforcing the Fake News")
    table_negative = make_html_table(top_negative, "❄️ Top 10 Comments Weakening the Fake News")

    # =====================================================
    # SUMMARY HTML
    # =====================================================
    summary_html = f"""
    <h1>🧠 Fake News Impact Dashboard</h1>
    <p><b>Date generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")}<br>
    <b>Total comments:</b> {len(df)}<br>
    <b>Global Reinforcement Index:</b> {total_index:.3f}<br>
    <b>Interpretation:</b> {direction}</p>

    <h3>Average Adjusted Impact by Stance</h3>
    {mean_by_stance.to_frame().to_html(classes="display compact", border=0)}
    """

    # =====================================================
    # BUILD FINAL HTML
    # =====================================================
    html_output = f"""
    <html>
    <head>
    <meta charset="UTF-8">
    <title>Fake News Impact Dashboard</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css"/>
    <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
    <script src="https://cdn.datatables.net/1.13.1/js/jquery.dataTables.min.js"></script>
    <script>
    $(document).ready(function() {{
    $('table.display').DataTable({{
        "pageLength": 10,
        "lengthMenu": [ [10, 25, 50, -1], [10, 25, 50, "All"] ]
    }});
    }});
    </script>
    </head>
    <body style="font-family: Arial; margin: 30px;">
    {summary_html}
    <h2>📈 Temporal Evolution</h2>
    {fig1.to_html(full_html=False, include_plotlyjs='cdn')}
    <h2>📊 Distribution by Stance</h2>
    {fig2.to_html(full_html=False, include_plotlyjs=False)}
    <h2>🎭 Sarcasm vs Impact</h2>
    {fig3.to_html(full_html=False, include_plotlyjs=False)}
    {table_positive}
    {table_negative}
    </body>
    </html>
    """

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"✅ Dashboard generated: {OUTPUT_HTML}")
