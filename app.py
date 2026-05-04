"""
Interactive Web-Based Learning System for Logistic Regression
Backend: Flask
ML: scikit-learn
Visualization: matplotlib, seaborn (rendered as base64 PNG)
"""
import os
import io
import json
import base64
import uuid
from typing import Any, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "logreg-learning-secret-change-me"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# In-memory store keyed by session id (simple, fine for an educational tool).
STORE: Dict[str, Dict[str, Any]] = {}


def sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    s = session["sid"]
    if s not in STORE:
        STORE[s] = {}
    return s


def state() -> Dict[str, Any]:
    return STORE[sid()]


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    sid()
    return render_template("index.html")


# ---------------------------------------------------------------------------
# 1. Dataset upload
# ---------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400
    name = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, f"{sid()}_{name}")
    f.save(path)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400

    st = state()
    st["df"] = df
    st["original_df"] = df.copy()
    st.pop("model", None)
    st.pop("X_train", None)

    return jsonify({
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": df.head().to_dict(orient="records"),
        "missing": df.isna().sum().to_dict(),
    })


@app.route("/api/load_sample", methods=["POST"])
def load_sample():
    path = os.path.join(APP_ROOT, "sample_data", "diabetes.csv")
    df = pd.read_csv(path)
    st = state()
    st["df"] = df
    st["original_df"] = df.copy()
    st.pop("model", None)
    return jsonify({
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": df.head().to_dict(orient="records"),
        "missing": df.isna().sum().to_dict(),
    })


# ---------------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------------
@app.route("/api/preprocess", methods=["POST"])
def preprocess():
    st = state()
    if "df" not in st:
        return jsonify({"error": "Upload a dataset first."}), 400
    cfg = request.json or {}
    target = cfg.get("target")
    missing = cfg.get("missing", "mean")        # mean | median | mode | drop
    encoding = cfg.get("encoding", "label")     # label | onehot
    scale = cfg.get("scale", True)

    df = st["original_df"].copy()
    if target not in df.columns:
        return jsonify({"error": "Target column not found."}), 400

    before_head = df.head().to_dict(orient="records")

    # Missing values
    if missing == "drop":
        df = df.dropna()
    else:
        for c in df.columns:
            if df[c].isna().any():
                if df[c].dtype.kind in "biufc":
                    if missing == "mean":
                        df[c] = df[c].fillna(df[c].mean())
                    elif missing == "median":
                        df[c] = df[c].fillna(df[c].median())
                    else:
                        df[c] = df[c].fillna(df[c].mode().iloc[0])
                else:
                    df[c] = df[c].fillna(df[c].mode().iloc[0])

    y_raw = df[target]
    X = df.drop(columns=[target])

    # Encoding for categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if encoding == "label":
        for c in cat_cols:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    else:
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Encode target if categorical
    target_encoder = None
    if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))
        class_names = target_encoder.classes_.tolist()
    else:
        y = y_raw.values
        class_names = sorted(pd.unique(y).tolist())

    feature_names = X.columns.tolist()
    X_values = X.values.astype(float)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_values = scaler.fit_transform(X_values)

    st.update({
        "df_processed": pd.DataFrame(X_values, columns=feature_names),
        "X": X_values,
        "y": np.array(y),
        "feature_names": feature_names,
        "class_names": [str(c) for c in class_names],
        "target": target,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "preproc_cfg": cfg,
    })

    after_df = pd.DataFrame(X_values, columns=feature_names)
    after_df[target] = y
    return jsonify({
        "before_head": before_head,
        "after_head": after_df.head().to_dict(orient="records"),
        "feature_names": feature_names,
        "class_names": [str(c) for c in class_names],
        "n_classes": int(len(set(y))),
        "shape": [int(X_values.shape[0]), int(X_values.shape[1])],
    })


# ---------------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------------
@app.route("/api/eda", methods=["POST"])
def eda():
    st = state()
    if "df" not in st:
        return jsonify({"error": "Upload a dataset first."}), 400
    df = st["df"].copy()
    target = (request.json or {}).get("target") or st.get("target")
    plots = {}

    num = df.select_dtypes(include=np.number)

    # Histograms
    if not num.empty:
        cols = num.columns[:6]
        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.array(axes).reshape(-1)
        for i, c in enumerate(cols):
            sns.histplot(num[c].dropna(), kde=True, ax=axes[i], color="#6366f1")
            axes[i].set_title(c)
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plots["histograms"] = fig_to_base64(fig)

    # Correlation heatmap
    if num.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(num.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation heatmap")
        plots["heatmap"] = fig_to_base64(fig)

    # Class distribution
    if target and target in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.countplot(x=df[target], ax=ax, palette="viridis")
        ax.set_title(f"Class distribution: {target}")
        plots["class_dist"] = fig_to_base64(fig)

        # Feature vs target (boxplots, first few numerical features)
        feats = [c for c in num.columns if c != target][:4]
        if feats:
            fig, axes = plt.subplots(1, len(feats), figsize=(4 * len(feats), 3))
            if len(feats) == 1:
                axes = [axes]
            for ax, c in zip(axes, feats):
                sns.boxplot(x=df[target], y=df[c], ax=ax, palette="Set2")
                ax.set_title(f"{c} vs {target}")
            plots["feature_vs_target"] = fig_to_base64(fig)

    return jsonify(plots)


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
@app.route("/api/train", methods=["POST"])
def train():
    st = state()
    if "X" not in st:
        return jsonify({"error": "Run preprocessing first."}), 400
    cfg = request.json or {}
    test_size = float(cfg.get("test_size", 0.2))
    k = int(cfg.get("k_folds", 5))
    max_iter = int(cfg.get("max_iter", 1000))

    X, y = st["X"], st["y"]
    n_classes = len(set(y))
    multi = "ovr" if n_classes > 2 else "auto"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if n_classes > 1 else None
    )

    model = LogisticRegression(max_iter=max_iter, multi_class=multi)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(LogisticRegression(max_iter=max_iter, multi_class=multi),
                                X, y, cv=k).tolist()

    st.update({
        "model": model, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    })

    return jsonify({
        "weights": model.coef_.tolist(),
        "bias": model.intercept_.tolist(),
        "feature_names": st["feature_names"],
        "class_names": st["class_names"],
        "train_score": float(model.score(X_train, y_train)),
        "test_score": float(model.score(X_test, y_test)),
        "cv_scores": cv_scores,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "n_classes": n_classes,
    })


# ---------------------------------------------------------------------------
# 5. Step-by-step explanation for a single sample
# ---------------------------------------------------------------------------
@app.route("/api/explain_sample", methods=["POST"])
def explain_sample():
    st = state()
    if "model" not in st:
        return jsonify({"error": "Train the model first."}), 400
    idx = int((request.json or {}).get("index", 0))
    X = st["X_test"]
    if idx < 0 or idx >= len(X):
        idx = 0
    x = X[idx]
    model = st["model"]
    W = model.coef_
    b = model.intercept_
    z = (W @ x) + b
    # sigmoid for binary, softmax for multiclass
    if z.shape[0] == 1:
        prob = 1 / (1 + np.exp(-z))
        probs = [float(1 - prob[0]), float(prob[0])]
    else:
        e = np.exp(z - z.max())
        probs = (e / e.sum()).tolist()
    pred = int(np.argmax(probs))

    terms = []
    for i, (w_row, fname) in enumerate(zip(W, st["class_names"] if W.shape[0] > 1 else [None])):
        terms.append({
            "class": st["class_names"][i] if W.shape[0] > 1 else st["class_names"][1] if len(st["class_names"]) > 1 else "1",
            "contributions": [
                {"feature": fn, "x": float(xv), "w": float(wv), "wx": float(wv * xv)}
                for fn, xv, wv in zip(st["feature_names"], x, w_row)
            ],
            "bias": float(b[i]),
            "z": float(z[i]),
        })

    return jsonify({
        "x": x.tolist(),
        "feature_names": st["feature_names"],
        "terms": terms,
        "probabilities": probs,
        "class_names": st["class_names"],
        "prediction": st["class_names"][pred] if pred < len(st["class_names"]) else str(pred),
        "actual": st["class_names"][int(st["y_test"][idx])] if int(st["y_test"][idx]) < len(st["class_names"]) else str(st["y_test"][idx]),
    })


# ---------------------------------------------------------------------------
# 6. Visualizations: decision boundary + sigmoid curve
# ---------------------------------------------------------------------------
@app.route("/api/visualize", methods=["POST"])
def visualize():
    st = state()
    if "model" not in st:
        return jsonify({"error": "Train the model first."}), 400
    plots = {}

    # Sigmoid curve
    fig, ax = plt.subplots(figsize=(6, 3.5))
    z = np.linspace(-8, 8, 200)
    s = 1 / (1 + np.exp(-z))
    ax.plot(z, s, color="#6366f1", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle=":")
    ax.set_title("Sigmoid: σ(z) = 1 / (1 + e^-z)")
    ax.set_xlabel("z"); ax.set_ylabel("σ(z)")
    plots["sigmoid"] = fig_to_base64(fig)

    # Decision boundary if 2D (or use 2 most-important features via PCA-free path)
    X = st["X"]; y = st["y"]
    if X.shape[1] >= 2:
        # Use first 2 features by importance (|coef| sum)
        importance = np.sum(np.abs(st["model"].coef_), axis=0)
        idx = np.argsort(importance)[::-1][:2]
        X2 = X[:, idx]
        from sklearn.linear_model import LogisticRegression as LR
        m2 = LR(max_iter=1000).fit(X2, y)
        x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
        y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = m2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        scatter = ax.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
        ax.set_xlabel(st["feature_names"][idx[0]])
        ax.set_ylabel(st["feature_names"][idx[1]])
        ax.set_title("Decision boundary (top-2 features)")
        plots["boundary"] = fig_to_base64(fig)

    return jsonify(plots)


# ---------------------------------------------------------------------------
# 7. Predict
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    st = state()
    if "model" not in st:
        return jsonify({"error": "Train the model first."}), 400
    values = (request.json or {}).get("values", {})
    try:
        x = np.array([float(values.get(f, 0)) for f in st["feature_names"]], dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    if st.get("scaler") is not None:
        x = st["scaler"].transform(x)
    model = st["model"]
    z = (model.coef_ @ x[0]) + model.intercept_
    probs = model.predict_proba(x)[0].tolist()
    pred = int(model.predict(x)[0])
    return jsonify({
        "z": z.tolist(),
        "probabilities": probs,
        "class_names": st["class_names"],
        "prediction": st["class_names"][pred] if pred < len(st["class_names"]) else str(pred),
    })


# ---------------------------------------------------------------------------
# 8. Evaluate
# ---------------------------------------------------------------------------
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    st = state()
    if "model" not in st:
        return jsonify({"error": "Train the model first."}), 400
    model = st["model"]
    X_test, y_test = st["X_test"], st["y_test"]
    y_pred = model.predict(X_test)
    n_classes = len(st["class_names"])
    avg = "binary" if n_classes == 2 else "weighted"

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        "report": classification_report(y_test, y_pred, target_names=st["class_names"], zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=st["class_names"], yticklabels=st["class_names"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    metrics["cm_plot"] = fig_to_base64(fig)

    # ROC
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        if n_classes == 2:
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#6366f1")
            metrics["auc"] = float(roc_auc)
        else:
            y_bin = label_binarize(y_test, classes=list(range(n_classes)))
            y_score = model.predict_proba(X_test)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                ax.plot(fpr, tpr, label=f"{st['class_names'][i]} (AUC={auc(fpr, tpr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC curve")
        ax.legend()
        metrics["roc_plot"] = fig_to_base64(fig)
    except Exception as e:
        metrics["roc_error"] = str(e)

    return jsonify(metrics)


# ---------------------------------------------------------------------------
# 9. Ask AI (LLM) — optional, uses OPENAI_API_KEY if set, else rule-based fallback
# ---------------------------------------------------------------------------
KNOWLEDGE = {
    "sigmoid": "The sigmoid function σ(z)=1/(1+e^-z) maps any real number to (0,1), giving us a probability. It's smooth, differentiable, and saturates at 0 and 1 — perfect for binary classification.",
    "threshold": "0.5 is the default threshold because it splits the probability space evenly. If P(y=1|x) ≥ 0.5 we predict class 1. You can tune this threshold based on the cost of false positives vs false negatives.",
    "linear vs logistic": "Linear regression predicts continuous values using y = wx+b. Logistic regression passes that linear combination through the sigmoid to output a probability for classification.",
    "decision boundary": "The decision boundary is where P(y=1|x)=0.5, i.e. where w·x + b = 0. For 2 features it's a line; for n features it's an (n-1)-dimensional hyperplane.",
    "multiclass": "For multiclass problems, One-vs-Rest trains one binary classifier per class. Softmax (multinomial) computes probabilities jointly across all classes.",
}

@app.route("/api/ask", methods=["POST"])
def ask():
    q = ((request.json or {}).get("question") or "").strip().lower()
    if not q:
        return jsonify({"answer": "Please ask a question about logistic regression."})

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            import requests
            ctx = ""
            st = state()
            if "model" in st:
                ctx = (f"Model trained on features {st['feature_names']}, "
                       f"classes {st['class_names']}, "
                       f"test accuracy {st['model'].score(st['X_test'], st['y_test']):.3f}.")
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "gpt-4o-mini", "messages": [
                    {"role": "system", "content": "You are a concise tutor for logistic regression. " + ctx},
                    {"role": "user", "content": q},
                ]},
                timeout=30,
            )
            return jsonify({"answer": r.json()["choices"][0]["message"]["content"]})
        except Exception as e:
            return jsonify({"answer": f"(LLM error, using fallback) {str(e)}"})

    # Fallback knowledge base
    for k, v in KNOWLEDGE.items():
        if k in q:
            return jsonify({"answer": v})
    return jsonify({"answer": (
        "Logistic Regression models P(y=1|x) = σ(w·x + b). It learns weights w and bias b "
        "by minimizing log-loss. Try asking about: sigmoid, threshold, decision boundary, "
        "linear vs logistic, multiclass."
    )})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
