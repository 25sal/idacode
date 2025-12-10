import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from tqdm import tqdm
import plotly.express as px
import re

# ===========================================
# CONFIGURATION
# ===========================================
CONFIDENCE_THRESHOLD = 0.6
SARCASM_THRESHOLD = 0.7

PATH = "data/CulturalDeepfake"
files = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for fileid in files:
    INPUT_FILE = f"{PATH}/{fileid}_classified_comments.csv"         # Input con: text, stance, confidence
    UNCERTAIN_REPORT = f"{PATH}/{fileid}_uncertainty_sarcasm_report.csv"
    SUSPECT_FILE = f"{PATH}/{fileid}_suspect_comments_explained.csv"
    HTML_REPORT = f"{PATH}/{fileid}_uncertainty_sarcasm_dashboard.html"



    # ===========================================
    # LOAD DATA
    # ===========================================
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["Content", "stance", "confidence"])
    uncertain_df = df[df["confidence"] < CONFIDENCE_THRESHOLD].copy()
    print(f"🟡 Found {len(uncertain_df)} uncertain comments out of {len(df)} total.")

    # ===========================================
    # LOAD SARCASM DETECTOR
    # ===========================================
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "mrm8488/t5-base-finetuned-sarcasm-twitter"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        sarcasm_detector = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    except Exception as e:
        print("⚠️ Fallback: using a robust RoBERTa irony classifier instead.")
        sarcasm_detector = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-irony",
            device=-1
        )
    # ===========================================
    # DETECT SARCASM
    # ===========================================
    tqdm.pandas(desc="Detecting sarcasm...")

    def detect_sarcasm(text):
        try:
            result = sarcasm_detector(f"detect sarcasm: {text[:512]}")
            if isinstance(result, list) and "generated_text" in result[0]:
                label = result[0]["generated_text"].strip().lower()
                score = 1.0 if "sarcastic" in label else 0.0
            else:
                label = result[0]["label"].lower()
                score = result[0]["score"]
            return pd.Series({"sarcasm_label": label, "sarcasm_score": score})
        except Exception as e:
            print(f"Error on text: {text[:60]}... ({e})")
            return pd.Series({"sarcasm_label": "error", "sarcasm_score": 0.0})

    uncertain_df[["sarcasm_label", "sarcasm_score"]] = uncertain_df["Content"].progress_apply(detect_sarcasm)

    # ===========================================
    # SIMPLE EXPLANATION FUNCTION
    # ===========================================
    def explain_comment(row):
        text = row["Content"].lower()
        sarcasm = row["sarcasm_score"]
        stance = row["stance"]
        conf = row["confidence"]

        # Pattern detection
        has_emoji = bool(re.search(r"[🤦🙄😂🤣😅🤔😏]", text))
        has_quotes = '"' in text or "'" in text
        has_exclamation = "!" in text
        has_negation = any(word in text for word in ["not", "no", "never", "fake", "hoax"])
        has_question = "?" in text

        if sarcasm > 0.7 and conf < 0.5:
            if stance == "criticizes the fake news":
                return "Likely sarcastic criticism — ironic disbelief or mockery of the fake news."
            elif stance == "believes the fake news":
                return "Possible ironic endorsement — sarcasm misinterpreted as belief."
            elif stance == "neutral":
                return "Ambiguous tone — sarcastic or humorous uncertainty."
        elif has_question and sarcasm > 0.6:
            return "Sarcastic questioning — likely expressing doubt or disbelief."
        elif has_emoji and sarcasm > 0.6:
            return "Sarcastic tone reinforced by emoji."
        elif has_negation and conf < 0.6:
            return "Likely critical or skeptical language."
        elif has_exclamation and sarcasm < 0.4:
            return "Strong emotion, possibly genuine belief."
        else:
            return "Ambiguous or uncertain expression — may need manual review."

    # ===========================================
    # SAVE REPORTS
    # ===========================================
    uncertain_df["explanation"] = uncertain_df.apply(explain_comment, axis=1)

    suspect_df = uncertain_df[
        (uncertain_df["confidence"] < CONFIDENCE_THRESHOLD)
        & (uncertain_df["sarcasm_score"] > SARCASM_THRESHOLD)
    ].copy()

    suspect_df = suspect_df.sort_values(by="sarcasm_score", ascending=False)
    suspect_df.to_csv(SUSPECT_FILE, index=False)
    uncertain_df.to_csv(UNCERTAIN_REPORT, index=False)

    print(f"✅ Detailed CSV saved: {UNCERTAIN_REPORT}")
    print(f"⚠️ {len(suspect_df)} suspect comments saved with explanations: {SUSPECT_FILE}")

    # ===========================================
    # INTERACTIVE DASHBOARD (HTML)
    # ===========================================
    fig = px.scatter(
        uncertain_df,
        x="confidence",
        y="sarcasm_score",
        color="stance",
        hover_data=["Content", "sarcasm_label", "explanation"],
        title="Sarcasm vs Model Confidence (Uncertain Comments)",
        template="plotly_white"
    )
    fig.add_hline(y=SARCASM_THRESHOLD, line_dash="dash", line_color="red", annotation_text="Sarcasm Threshold")
    fig.add_vline(x=CONFIDENCE_THRESHOLD, line_dash="dash", line_color="orange", annotation_text="Confidence Threshold")

    summary_html = f"""
    <h1>🧠 Uncertainty & Sarcasm Analysis Report</h1>
    <p><b>Total comments:</b> {len(df)}<br>
    <b>Uncertain:</b> {len(uncertain_df)}<br>
    <b>Suspect (sarcastic + uncertain):</b> {len(suspect_df)}</p>

    <h3>Distribution of Sarcasm Labels</h3>
    {uncertain_df['sarcasm_label'].value_counts().to_frame().to_html()}

    <h3>Correlation between Stance and Sarcasm</h3>
    {pd.crosstab(uncertain_df['stance'], uncertain_df['sarcasm_label']).to_html()}
    """

    tables_html = """
    <h3>Suspect Comments (with explanations)</h3>
    """ + suspect_df[["Content", "stance", "confidence", "sarcasm_score", "explanation"]].to_html(
        index=False, classes="display compact", escape=False
    )

    html_output = f"""
    <html>
    <head>
    <meta charset="UTF-8">
    <title>Uncertainty & Sarcasm Analysis</title>
    <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
    <script src="https://cdn.datatables.net/1.13.1/js/jquery.dataTables.min.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.1/css/jquery.dataTables.min.css"/>
    <script>
    $(document).ready(function() {{
    $('table.display').DataTable({{
        "pageLength": 10,
        "lengthMenu": [ [10, 25, 50, -1], [10, 25, 50, "All"] ],
        "order": [[ 2, "asc" ]]
    }});
    }});
    </script>
    </head>
    <body style="font-family: Arial; margin: 30px;">
    {summary_html}
    <h2>Interactive Scatter Plot</h2>
    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
    {tables_html}
    </body>
    </html>
    """

    with open(HTML_REPORT, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"🌐 Interactive HTML dashboard saved to {HTML_REPORT}")

    # ===========================================
    # OPTIONAL STATIC PLOT
    # ===========================================
    plt.figure(figsize=(8, 5))
    plt.scatter(uncertain_df["confidence"], uncertain_df["sarcasm_score"], alpha=0.5, label="Uncertain")
    plt.scatter(suspect_df["confidence"], suspect_df["sarcasm_score"], color="red", label="Suspect")
    plt.axvline(CONFIDENCE_THRESHOLD, color="orange", linestyle="--")
    plt.axhline(SARCASM_THRESHOLD, color="red", linestyle="--")
    plt.title("Sarcasm vs Confidence with Explanations")
    plt.xlabel("Confidence")
    plt.ylabel("Sarcasm Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PATH}/{fileid}_uncertainty_sarcasm_plot.png", dpi=150)
    print(f"🖼️ Static plot saved: {PATH}/{fileid}_uncertainty_sarcasm_plot.png")
