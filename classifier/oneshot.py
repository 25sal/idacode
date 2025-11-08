import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

path ="data/CulturalDeepfake"

# ================================
# CONFIGURATION
# ================================
MODEL_NAME = "facebook/bart-large-mnli"
LOCAL_MODEL_DIR = "./local_bart_mnli"
files = [1]
for fileid in files:
    INPUT_FILE = os.path.join(path, f"posts/{fileid}.csv")               # must contain column 'text'
    OUTPUT_FILE = f"{path}/{fileid}_classified_comments.csv"
    DEVICE = 0  # -1 = CPU, 0 = GPU if available

    # ================================
    # MODEL LOADING (offline-ready)
    # ================================
    if not os.path.exists(LOCAL_MODEL_DIR):
        print("🔻 Downloading model (only the first time)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.save_pretrained(LOCAL_MODEL_DIR)
        tokenizer.save_pretrained(LOCAL_MODEL_DIR)
        print("✅ Model downloaded and saved locally.")
    else:
        print("✅ Using local model copy.")

    classifier = pipeline(
        "zero-shot-classification",
        model=LOCAL_MODEL_DIR,
        tokenizer=LOCAL_MODEL_DIR,
        device=DEVICE
    )

    # ================================
    # LOAD COMMENTS
    # ================================
    df = pd.read_csv(INPUT_FILE)
    df = df[['Content','ReactionsCount','SubCommentsCount', 'CommentAt']][1:]
    if "Content" not in df.columns:
        raise ValueError("Input CSV must contain a column named 'Content'")

    print(f"📄 Loaded {len(df)} comments from {INPUT_FILE}")

    # ================================
    # CLASSIFICATION
    # ================================
    labels = ["believes the fake news", "criticizes the fake news", "neutral"]

    stances = []
    confidences = []

    for i, text in enumerate(df["Content"], 1):
        if not isinstance(text, str) or text.strip() == "":
            stances.append("neutral")
            confidences.append(0.0)
            continue
        pred = classifier(
            text,
            candidate_labels=labels,
            hypothesis_template="This comment {}."
        )
        stances.append(pred["labels"][0])
        confidences.append(pred["scores"][0])
        if i % 20 == 0:
            print(f"Processed {i}/{len(df)} comments...")

    df["stance"] = stances
    df["confidence"] = confidences

    # ================================
    # SAVE RESULTS
    # ================================
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")

    # ================================
    # SUMMARY REPORT
    # ================================
    counts = df["stance"].value_counts()
    percentages = (counts / len(df) * 100).round(2)
    summary = pd.DataFrame({"Count": counts, "Percentage": percentages})
    print("\n📊 Summary:")
    print(summary)

    # ================================
    # PLOT RESULTS
    # ================================
    plt.figure(figsize=(6,4))
    plt.bar(summary.index, summary["Count"])
    plt.title("Distribution of Comment Stances")
    plt.xlabel("Stance")
    plt.ylabel("Number of Comments")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{path}/{fileid}_stance_distribution.png", dpi=150)
    plt.show()

    print(f"\n📈 Bar chart saved as: {fileid}_stance_distribution.png")
