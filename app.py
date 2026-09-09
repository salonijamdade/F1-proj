import json
import warnings
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Review Intelligence · Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --teal:#00E5C4; --blue:#3B82F6; --purple:#A855F7;
    --red:#FF6B6B;  --gold:#FFD93D; --green:#4ADE80;
    --bg:#070B14;   --card:rgba(255,255,255,0.03);
    --border:rgba(255,255,255,0.08); --text:#C8D8F0; --muted:#556688;
}

.stApp { background:var(--bg); color:var(--text); }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.block-container { padding:1.5rem 2rem 4rem; max-width:1300px; }
[data-testid="stSidebar"] { display:none; }

.sec-hdr {
    display:flex; align-items:center; gap:10px;
    border-left:3px solid var(--teal); padding-left:.9rem;
    margin:2.5rem 0 .8rem;
    font-size:.9rem; font-weight:700; color:#E0EEFF;
    letter-spacing:.08em; text-transform:uppercase;
}

[data-testid="metric-container"] {
    background:var(--card); border:1px solid var(--border);
    border-radius:10px; padding:.9rem 1.1rem;
}
[data-testid="metric-container"] label {
    color:var(--muted) !important; font-family:'JetBrains Mono',monospace;
    font-size:.65rem; letter-spacing:.1em; text-transform:uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:var(--teal) !important; font-family:'JetBrains Mono',monospace; font-size:1.4rem;
}

.insight-card {
    background:var(--card); border:1px solid var(--border);
    border-radius:10px; padding:1rem 1.2rem; margin:.5rem 0;
}
.insight-card h4 { margin:0 0 .4rem; font-size:.85rem; font-weight:700; }
.insight-card p  { margin:0; color:#8899BB; font-size:.82rem; line-height:1.5; }

.kw { display:inline-block; background:rgba(0,229,196,.1); border:1px solid rgba(0,229,196,.25);
      color:var(--teal); font-family:'JetBrains Mono',monospace; font-size:.68rem;
      padding:2px 8px; border-radius:4px; margin:2px; }
.kw-red    { background:rgba(255,107,107,.1);  border-color:rgba(255,107,107,.3);  color:var(--red); }
.kw-purple { background:rgba(168,85,247,.1);   border-color:rgba(168,85,247,.3);   color:var(--purple); }
.kw-gold   { background:rgba(255,217,61,.1);   border-color:rgba(255,217,61,.3);   color:var(--gold); }
.kw-green  { background:rgba(74,222,128,.1);   border-color:rgba(74,222,128,.3);   color:var(--green); }

.risk-high { display:inline-block; background:rgba(255,107,107,.15); border:1px solid rgba(255,107,107,.4);
             color:var(--red); border-radius:20px; padding:3px 12px;
             font-family:'JetBrains Mono',monospace; font-size:.7rem; font-weight:700; }
.risk-med  { background:rgba(255,217,61,.12);  border:1px solid rgba(255,217,61,.4);
             color:var(--gold); border-radius:20px; padding:3px 12px; display:inline-block;
             font-family:'JetBrains Mono',monospace; font-size:.7rem; font-weight:700; }
.risk-low  { background:rgba(74,222,128,.1);   border:1px solid rgba(74,222,128,.3);
             color:var(--green); border-radius:20px; padding:3px 12px; display:inline-block;
             font-family:'JetBrains Mono',monospace; font-size:.7rem; font-weight:700; }

.info-box {
    background:rgba(59,130,246,.06); border:1px solid rgba(59,130,246,.2);
    border-radius:8px; padding:.7rem 1rem; color:var(--muted);
    font-size:.8rem; font-family:'JetBrains Mono',monospace; margin:.5rem 0 1rem;
}

.summary-box {
    background:rgba(0,229,196,.04); border:1px solid rgba(0,229,196,.15);
    border-radius:10px; padding:1rem 1.3rem; margin:.5rem 0 1.2rem;
    color:#B0C8E8; font-size:.88rem; line-height:1.7;
}
.summary-box strong { color:var(--teal); }
.summary-box .label {
    font-family:'JetBrains Mono',monospace; font-size:.65rem; font-weight:700;
    color:var(--teal); letter-spacing:.1em; text-transform:uppercase;
    margin-bottom:.5rem; display:block;
}

.stTabs [data-baseweb="tab-list"] { background:transparent; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background:transparent !important; color:var(--muted) !important;
    font-family:'JetBrains Mono',monospace; font-size:.72rem;
    letter-spacing:.05em; text-transform:uppercase;
    border-radius:0 !important; padding:8px 18px;
    border-bottom:2px solid transparent !important;
}
.stTabs [aria-selected="true"] { color:var(--teal) !important; border-bottom:2px solid var(--teal) !important; }

.stChatMessage { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:10px !important; }
.agent-badge {
    display:inline-block; font-family:'JetBrains Mono',monospace;
    font-size:.62rem; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; padding:2px 10px; border-radius:4px; margin-bottom:6px;
}

.settings-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; }
details summary { color:var(--teal) !important; }

.report-wrap {
    background:linear-gradient(135deg,rgba(168,85,247,.04),rgba(59,130,246,.04));
    border:1px solid rgba(168,85,247,.2); border-radius:12px; padding:1.5rem 1.8rem;
}

[data-testid="stDataFrame"] th { background:rgba(0,229,196,.05) !important; color:var(--teal) !important; }

.ag-card {
    background:var(--card); border:1px solid var(--border);
    border-radius:10px; padding:.9rem 1rem; text-align:center;
}
.ag-card .ag-icon { font-size:1.4rem; margin-bottom:.3rem; }
.ag-card .ag-name { font-weight:700; font-size:.78rem; color:#E0EEFF; margin-bottom:.2rem; }
.ag-card .ag-desc { font-size:.72rem; color:var(--muted); line-height:1.4; }
</style>
""", unsafe_allow_html=True)


for k, v in {
    "chat_history":     [],
    "session_memory":   [],
    "analysis_context": {},
    "alert_log":        [],
    "report_cache":     None,
    "pending_query":    None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def sec(title, icon="◆"):
    st.markdown(f'<div class="sec-hdr"><span>{icon}</span> {title}</div>', unsafe_allow_html=True)

def pill(word, color=""):
    return f'<span class="kw {color}">{word}</span>'

def summary_box(label, html):
    st.markdown(
        f'<div class="summary-box"><span class="label">{label}</span>{html}</div>',
        unsafe_allow_html=True
    )

def dark_fig():
    plt.rcParams.update({
        "figure.facecolor":"#070B14","axes.facecolor":"#0D1526",
        "axes.edgecolor":"#1E2A42","axes.labelcolor":"#556688",
        "xtick.color":"#556688","ytick.color":"#556688",
        "text.color":"#C8D8F0","grid.color":"#1E2A42",
    })

AGENT_META = {
    "orchestrator": {"label":"🧠 Orchestrator",  "color":"#3B82F6","badge_cls":""},
    "data_analyst":  {"label":"📊 Data Analyst",  "color":"#00E5C4","badge_cls":""},
    "risk_analyst":  {"label":"⚠️ Risk Analyst",  "color":"#FF6B6B","badge_cls":""},
    "biz_strategist":{"label":"💡 Strategist",    "color":"#FFD93D","badge_cls":""},
    "report_gen":    {"label":"📝 Report Writer", "color":"#A855F7","badge_cls":""},
}


def load_data(file):
    df = pd.read_csv(file)
    before = len(df)
    df.drop_duplicates(inplace=True)
    dupes = before - len(df)
    if "review_text" not in df.columns:
        st.error("CSV must have a 'review_text' column."); st.stop()
    df.dropna(subset=["review_text"], inplace=True)
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df.get("rating", 3), errors="coerce").fillna(3)
    df["verified_purchase"] = (
        df.get("verified_purchase", pd.Series(["No"]*len(df)))
        .fillna("No").astype(str).str.strip().str.title()
    )
    df["product_variant"] = df.get("product_variant", pd.Series(["Unknown"]*len(df))).fillna("Unknown").astype(str)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df, dupes


@st.cache_data(show_spinner=False)
def build_features(texts_t, ratings_t, verified_t, variants_t, n_tfidf=500):
    texts=pd.Series(texts_t); ratings=pd.Series(ratings_t)
    verified=pd.Series(verified_t); variants=pd.Series(variants_t)
    tfidf  = TfidfVectorizer(max_features=n_tfidf, ngram_range=(1,2), stop_words="english")
    X_text = tfidf.fit_transform(texts)
    vp_num = (verified.str.lower()=="yes").astype(float).values.reshape(-1,1)
    var_dum= pd.get_dummies(variants, prefix="pv").astype(float)
    rng    = ratings.max()-ratings.min()
    r_norm = ((ratings-ratings.min())/(rng+1e-9)).values.reshape(-1,1)
    svd    = TruncatedSVD(n_components=min(50,X_text.shape[1]-1), random_state=42)
    X_svd  = svd.fit_transform(X_text)
    X_comb = np.hstack([X_svd, r_norm, vp_num, var_dum.values])
    return normalize(X_comb), X_text, tfidf, svd


@st.cache_data(show_spinner=False)
def find_optimal_k(_X, k_range=(2,8)):
    ks,inertias,sils=[],[],[]
    for k in range(k_range[0], min(k_range[1]+1, len(_X))):
        km=KMeans(n_clusters=k, random_state=42, n_init=10); lbl=km.fit_predict(_X)
        ks.append(k); inertias.append(km.inertia_); sils.append(silhouette_score(_X,lbl))
    return ks[int(np.argmax(sils))], ks, inertias, sils


@st.cache_data(show_spinner=False)
def run_clustering(_X, k):
    km=KMeans(n_clusters=k, random_state=42, n_init=10); kml=km.fit_predict(_X)
    ks=silhouette_score(_X,kml)
    nn=NearestNeighbors(n_neighbors=5).fit(_X); d,_=nn.kneighbors(_X)
    eps=float(np.percentile(d[:,-1],90))
    db=DBSCAN(eps=eps, min_samples=max(3,len(_X)//100)); dbl=db.fit_predict(_X)
    agg=AgglomerativeClustering(n_clusters=k); agl=agg.fit_predict(_X)
    return {
        "kmeans":       {"labels":kml,"silhouette":ks},
        "dbscan":       {"labels":dbl,"n_clusters":len(set(dbl)-{-1}),"noise":int((dbl==-1).sum())},
        "agglomerative":{"labels":agl},
    }


@st.cache_data(show_spinner=False)
def run_topics(texts_t, n_topics=6, method="nmf"):
    texts=list(texts_t)
    cv=CountVectorizer(max_features=1000, stop_words="english", min_df=2)
    Xc=cv.fit_transform(texts); words=cv.get_feature_names_out()
    model=(NMF if method=="nmf" else LatentDirichletAllocation)(n_components=n_topics, random_state=42)
    dt=model.fit_transform(Xc)
    return {i:[words[j] for j in np.argsort(c)[::-1][:10]] for i,c in enumerate(model.components_)}, dt


@st.cache_data(show_spinner=False)
def detect_anomalies(_X, contamination=0.05):
    iso=IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    ip=iso.fit_predict(_X); isc=iso.score_samples(_X)
    lof=LocalOutlierFactor(n_neighbors=20, contamination=contamination); lp=lof.fit_predict(_X)
    return ((ip==-1)|(lp==-1)).astype(int), isc


THEME_MAP = {
    frozenset(["battery","drain","charge","power","life"]):         "Battery Issues",
    frozenset(["camera","photo","picture","lens","video"]):         "Camera Quality",
    frozenset(["fast","smooth","speed","quick","performance"]):     "Performance",
    frozenset(["slow","lag","hang","freeze","crash"]):              "Performance Issues",
    frozenset(["price","value","cost","money","worth"]):            "Value for Money",
    frozenset(["love","great","excellent","amazing","best"]):       "Positive Feedback",
    frozenset(["heat","hot","overheat","warm","temperature"]):      "Heating Problems",
    frozenset(["display","screen","brightness","color","amoled"]):  "Display Quality",
    frozenset(["deliver","pack","ship","box","arrive"]):            "Delivery/Packaging",
    frozenset(["software","update","app","os","android","ui"]):     "Software/UI",
}

def auto_label(words):
    for theme, label in THEME_MAP.items():
        if len(theme & set(words)) >= 2: return label
    return "Mixed Reviews"

def interpret_clusters(df, labels, tfidf, X_text, k):
    results = {}
    for c in range(k):
        mask=labels==c; sub=df[mask]
        if not len(sub): continue
        arr=np.asarray(X_text[mask].mean(axis=0)).flatten()
        top_words=[tfidf.get_feature_names_out()[i] for i in arr.argsort()[::-1][:12]]
        results[c]={
            "size":int(mask.sum()), "pct":round(mask.mean()*100,1),
            "top_words":top_words, "avg_rating":round(sub["rating"].mean(),2),
            "vp_pct":round((sub["verified_purchase"].str.lower()=="yes").mean()*100,1),
            "samples":sub["review_text"].sample(min(5,len(sub)),random_state=42).tolist(),
            "label":auto_label(top_words),
        }
    return results


def build_analysis_context(df, cluster_info, topics, doc_topics,
                           anomaly_flags, iso_scores, km_sil, best_k,
                           anomaly_pct, dbscan_info):
    vp_yes=df[df["verified_purchase"].str.lower()=="yes"]
    vp_no =df[df["verified_purchase"].str.lower()!="yes"]

    cluster_summaries={}
    for c,info in cluster_info.items():
        cluster_summaries[str(c)]={
            "label":info["label"], "size":info["size"], "pct":info["pct"],
            "avg_rating":info["avg_rating"], "vp_pct":info["vp_pct"],
            "top_words":info["top_words"][:6],
            "sample":info["samples"][0][:100] if info["samples"] else "",
        }

    topic_summaries={str(i):{"kw":w[:4]} for i,w in topics.items()}

    anomalous=df[anomaly_flags==1].sort_values("iso_score").head(3)
    anomaly_samples=anomalous["review_text"].str[:80].tolist()

    rating_dist={str(k):int(v) for k,v in df["rating"].value_counts().sort_index().items()}

    return {
        "total":len(df), "avg_rating":round(df["rating"].mean(),2),
        "silhouette":round(km_sil,3), "best_k":best_k,
        "anomaly_pct":anomaly_pct, "anomaly_count":int(anomaly_flags.sum()),
        "dbscan_clusters":dbscan_info["n_clusters"], "dbscan_noise":dbscan_info["noise"],
        "verified_avg":round(vp_yes["rating"].mean(),2) if len(vp_yes) else None,
        "unverified_avg":round(vp_no["rating"].mean(),2) if len(vp_no) else None,
        "verified_count":len(vp_yes), "unverified_count":len(vp_no),
        "clusters":cluster_summaries,
        "topics":topic_summaries,
        "anomaly_samples":anomaly_samples,
        "rating_dist":rating_dist,
    }


GROK_API_KEY = "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"
GROK_URL = "https://api.groq.com/openai/v1/chat/completions"

AGENT_PROMPTS = {
"data_analyst": (
    "You are a data analyst. Analyse the customer review cluster data below.\n"
    "In 150-200 words: explain what each cluster represents, which cluster is most "
    "concerning (lowest rating), and one surprising pattern you notice.\n"
    "Use bullet points. Mention specific cluster numbers and ratings."
),
"risk_analyst": (
    "You are a risk analyst. Using the review data below, assess product risk.\n"
    "In 150-200 words: state overall risk level (HIGH/MEDIUM/LOW), list the top 2 specific "
    "product issues signalled by keywords, and give 2 escalation recommendations.\n"
    "Be direct and specific."
),
"biz_strategist": (
    "You are a business strategist. Using the review cluster data below, give business advice.\n"
    "In 150-200 words: 2 immediate actions, 2 medium-term improvements, and 1 growth opportunity.\n"
    "Ground every point in the actual cluster data."
),
"report_gen": (
    "You are an executive report writer. Write a briefing (under 300 words) using the data below.\n"
    "Format: ## Summary (2 sentences) → ## Top Risks (3 bullets) → "
    "## Opportunities (2 bullets) → ## Recommendations (3 bullets).\n"
    "No jargon. Numbers and cluster names only."
),
"orchestrator": (
    "You are an AI orchestrator. Synthesise these specialist reports into one clear answer "
    "(under 200 words). Remove repetition. Keep only the most important points.\n"
    "Use markdown with one header per theme."
),
}

def call_agent(agent_name: str, user_message: str, context: dict, memory: list) -> str:
    ctx_str = json.dumps(context, separators=(",", ":"), default=str)
    system = AGENT_PROMPTS[agent_name]
    if memory:
        mem = "\n".join(f"{m['role'].upper()}: {m['content'][:120]}" for m in memory[-3:])
        system += f"\n\nPrevious context:\n{mem}"

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"## DATA\n{ctx_str}\n\n## QUESTION\n{user_message}"}
        ],
        "max_tokens": 500,
        "temperature": 0.5,
    }

    wait = 8
    for attempt in range(4):
        try:
            r = requests.post(GROK_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                if attempt < 3:
                    st.toast(f"Rate limit — retrying in {wait}s…", icon="⏳")
                    time.sleep(wait); wait *= 2; continue
                return "⚠️ Rate limit reached. Please wait a minute and try again."
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if attempt < 3: time.sleep(wait); wait *= 2
            else: return f"⚠️ HTTP error: {e}"
        except Exception as e:
            return f"⚠️ Error: {e}"
    return "⚠️ All retries exhausted. Please wait and try again."


def orchestrate_query(user_query: str, context: dict, memory: list) -> list:
    lower = user_query.lower()

    scores = {
        "data_analyst":   sum(1 for w in ["cluster","pattern","topic","trend","group","distribution","what","how many","which"] if w in lower),
        "risk_analyst":   sum(1 for w in ["risk","anomaly","problem","issue","defect","complaint","bad","worst","danger"] if w in lower),
        "biz_strategist": sum(1 for w in ["recommend","action","strategy","improve","marketing","retain","suggest","plan","should"] if w in lower),
        "report_gen":     sum(1 for w in ["report","summary","brief","overview","everything","full"] if w in lower),
    }

    full_pipeline = any(w in lower for w in ["full report","complete report","executive report","all agents"])

    if full_pipeline:
        agents_to_run = ["data_analyst","risk_analyst","biz_strategist","report_gen"]
    else:
        best_score = max(scores.values())
        if best_score == 0:
            agents_to_run = ["data_analyst"]
        else:
            agents_to_run = [max(scores.items(), key=lambda x: x[1])[0]]

    outputs = []
    specialist_outputs = {}

    for i, ag in enumerate(agents_to_run):
        if i > 0: time.sleep(8)
        reply = call_agent(ag, user_query, context, memory)
        specialist_outputs[ag] = reply
        outputs.append({"agent":ag, "content":reply})

    if len(agents_to_run) > 1:
        valid = {a: c for a, c in specialist_outputs.items() if not c.startswith("⚠️")}
        if len(valid) > 1:
            time.sleep(8)
            synth = (
                f"Question: {user_query}\n\n"
                + "\n\n".join(f"[{a.upper()}]\n{c}" for a,c in valid.items())
                + "\n\nSynthesise into one clear answer."
            )
            final = call_agent("orchestrator", synth, context, memory)
            outputs.append({"agent":"orchestrator","content":final})

    return outputs


def generate_report(context: dict, memory: list) -> str:
    return call_agent(
        "report_gen",
        "Generate a full executive briefing report based on the analysis data.",
        context, memory
    )


def check_alerts(context, thresholds):
    alerts=[]
    if context.get("anomaly_pct",0) > thresholds["anomaly_pct"]:
        alerts.append({"level":"HIGH","msg":f"Anomaly rate {context['anomaly_pct']}% exceeds threshold {thresholds['anomaly_pct']}%"})
    if context.get("avg_rating",5) < thresholds["avg_rating"]:
        alerts.append({"level":"HIGH","msg":f"Avg rating {context['avg_rating']} below acceptable threshold of {thresholds['avg_rating']}"})
    for c,info in context.get("clusters",{}).items():
        if info["avg_rating"] < thresholds["cluster_rating"] and info["pct"] > 10:
            alerts.append({"level":"MEDIUM","msg":f"Cluster {c} ({info['label']}) — {info['pct']}% of reviews, avg rating only {info['avg_rating']}"})
    vg=(context.get("verified_avg") or 5)-(context.get("unverified_avg") or 5)
    if abs(vg) > thresholds["vp_gap"]:
        alerts.append({"level":"MEDIUM","msg":f"Verified vs unverified rating gap: {vg:+.2f} — possible fake review pattern"})
    return alerts


def fig_scatter(X2d, labels, cluster_info, title):
    dark_fig(); k=len(set(labels)-{-1}); cols=cm.tab10(np.linspace(0,1,max(k,2)))
    fig,ax=plt.subplots(figsize=(7,5))
    for c in sorted(set(labels)):
        m=labels==c; col="gray" if c==-1 else cols[c%len(cols)]
        lbl="Noise" if c==-1 else f"C{c}: {cluster_info.get(c,{}).get('label','')[:16]}"
        ax.scatter(X2d[m,0], X2d[m,1], c=[col], alpha=.6, s=14, label=lbl)
    ax.set_title(title,color="#C8D8F0"); ax.legend(fontsize=7,framealpha=.15)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); plt.tight_layout(); return fig

def fig_sizes(cluster_info):
    dark_fig()
    labels=[f"C{c}: {i['label'][:14]}" for c,i in cluster_info.items()]
    sizes=[i["size"] for i in cluster_info.values()]
    ratings=[i["avg_rating"] for i in cluster_info.values()]
    colors=[cm.RdYlGn(plt.Normalize(1,5)(r)) for r in ratings]
    fig,ax=plt.subplots(figsize=(8,max(3,len(labels)*.7)))
    bars=ax.barh(labels, sizes, color=colors, alpha=.85); ax.invert_yaxis()
    ax.set_title("Cluster Sizes  (green = high rating · red = low)",color="#C8D8F0")
    ax.grid(axis="x",alpha=.3)
    for bar,sz in zip(bars,sizes):
        ax.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2, str(sz), va="center", color="#8899BB", fontsize=8)
    plt.tight_layout(); return fig

def fig_elbow(ks,inertias,sils,best_k):
    dark_fig(); fig,(a1,a2)=plt.subplots(1,2,figsize=(10,3.5))
    a1.plot(ks,inertias,"o-",color="#00E5C4",lw=2); a1.axvline(best_k,color="#FF6B6B",ls="--",label=f"Best k={best_k}")
    a1.set_title("Elbow — finding the right number of clusters",color="#C8D8F0"); a1.legend(framealpha=.2); a1.grid(True)
    a2.plot(ks,sils,"o-",color="#FFD93D",lw=2); a2.axvline(best_k,color="#FF6B6B",ls="--")
    a2.set_title("Silhouette — how well-separated the clusters are",color="#C8D8F0"); a2.grid(True)
    plt.tight_layout(); return fig

def fig_silhouette(X,labels,k):
    dark_fig(); vals=silhouette_samples(X,labels); cols=cm.tab10(np.linspace(0,1,k))
    fig,ax=plt.subplots(figsize=(7,4)); y=10
    for c in range(k):
        cv=np.sort(vals[labels==c]); yh=y+len(cv)
        ax.fill_betweenx(np.arange(y,yh),0,cv,facecolor=cols[c],alpha=.75)
        ax.text(-.05,(y+yh)/2,str(c),fontsize=8,color=cols[c]); y=yh+5
    ax.axvline(vals.mean(),color="#FF6B6B",ls="--",lw=1.5,label=f"Avg={vals.mean():.2f}")
    ax.set_title("Silhouette Score per Cluster (wider = better separated)",color="#C8D8F0")
    ax.legend(framealpha=.2); plt.tight_layout(); return fig

def fig_topics(topics,doc_topics):
    dark_fig(); avg=doc_topics.mean(axis=0)
    lbls=[f"T{i}: {topics[i][0]}" for i in range(len(topics))]
    fig,ax=plt.subplots(figsize=(9,3))
    ax.bar(lbls, avg, color="#3B82F6", alpha=.85)
    ax.set_title("Topic Prevalence — how commonly each theme appears",color="#C8D8F0")
    plt.xticks(rotation=30,ha="right",fontsize=8); ax.grid(axis="y",alpha=.3)
    plt.tight_layout(); return fig

def fig_anomaly(iso_scores,flags):
    dark_fig(); fig,ax=plt.subplots(figsize=(9,3))
    nm=flags==0; an=flags==1
    ax.scatter(np.where(nm)[0],iso_scores[nm],c="#00E5C4",alpha=.35,s=8,label="Normal review")
    ax.scatter(np.where(an)[0],iso_scores[an],c="#FF6B6B",alpha=.85,s=15,label="Anomalous review")
    ax.set_title("Anomaly Detection — red dots are unusual or suspicious reviews",color="#C8D8F0")
    ax.set_xlabel("Review index"); ax.set_ylabel("Anomaly score (lower = more suspicious)")
    ax.legend(framealpha=.2); ax.grid(alpha=.3); plt.tight_layout(); return fig

def fig_rating(context):
    dark_fig(); rd=context.get("rating_dist",{})
    if not rd: return None
    keys=sorted(rd.keys()); vals=[rd[k] for k in keys]
    fig,ax=plt.subplots(figsize=(5,3))
    ax.bar(keys, vals, color=["#FF6B6B","#FF9500","#FFD93D","#90EE90","#4ADE80"][:len(keys)], alpha=.85)
    ax.set_title("How Customers Rated the Product",color="#C8D8F0"); ax.grid(axis="y",alpha=.3)
    plt.tight_layout(); return fig

def fig_dendro(X):
    dark_fig(); idx=np.random.choice(len(X),min(80,len(X)),replace=False)
    Z=linkage(X[idx],method="ward"); fig,ax=plt.subplots(figsize=(10,4))
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(Z,ax=ax,color_threshold=0,above_threshold_color="#00E5C4",leaf_font_size=0)
    ax.set_title("Hierarchical Clustering Tree — how review groups relate to each other",color="#C8D8F0")
    ax.set_yticks([]); plt.tight_layout(); return fig


def render_cluster_plain_english(cluster_info, anomaly_pct):
    lines = []
    lines.append(f"Your reviews split into <strong>{len(cluster_info)} natural groups</strong>. "
                 f"Here's what each group is saying:")
    lines.append("<br><br>")

    for c, info in cluster_info.items():
        rating = info["avg_rating"]
        mood   = ("😊 Mostly satisfied" if rating >= 4 else
                  "😐 Mixed feelings"   if rating >= 3 else
                  "😠 Mostly unhappy")
        kws    = ", ".join(info["top_words"][:5])
        lines.append(
            f"<b>Cluster {c} — {info['label']}</b> &nbsp;"
            f'<span class="kw">{info["pct"]}% of reviews</span> &nbsp;'
            f'<span class="kw kw-{"green" if rating>=4 else "red" if rating<3 else "gold"}">'
            f'Avg {rating}/5</span><br>'
            f"{mood} · Talking about: <em>{kws}</em><br><br>"
        )

    if anomaly_pct > 5:
        lines.append(
            f'<span class="kw kw-red">⚠ {anomaly_pct}% of reviews look unusual</span> — '
            "these may be spam, fake, or extreme edge cases and deserve manual review."
        )

    summary_box("What your reviews are saying", "".join(lines))


def render_topic_plain_english(topics):
    lines = ["Across all reviews, customers are consistently talking about these themes:<br><br>"]
    colors = ["","kw-red","kw-purple","kw-gold","kw-green",""]
    for i, words in topics.items():
        color = colors[i % len(colors)]
        kws   = " ".join(f'<span class="kw {color}">{w}</span>' for w in words[:6])
        lines.append(f"<b>Theme {i+1}:</b> {kws}<br>")
    summary_box("Common themes in customer feedback", "".join(lines))


def render_anomaly_plain_english(anomaly_count, anomaly_pct, total):
    level = ("🔴 <strong>High</strong>" if anomaly_pct > 10 else
             "🟡 <strong>Moderate</strong>" if anomaly_pct > 5 else
             "🟢 <strong>Low</strong>")
    html = (
        f"Out of <strong>{total:,}</strong> reviews, "
        f"<strong>{anomaly_count}</strong> ({anomaly_pct}%) were flagged as unusual. "
        f"Anomaly level: {level}.<br><br>"
        "Unusual reviews may be: fake/spam reviews, one-off defective unit complaints, "
        "or reviews that don't match the typical pattern for that product. "
        "These are worth reading manually before making product decisions."
    )
    summary_box("What 'anomalies' means in plain terms", html)


def render_chat_history():
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            ag   = msg.get("agent","orchestrator")
            meta = AGENT_META.get(ag, AGENT_META["orchestrator"])
            with st.chat_message("assistant"):
                st.markdown(
                    f'<span class="agent-badge" style="background:rgba(255,255,255,0.06);'
                    f'color:{meta["color"]}">{meta["label"]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(msg["content"])


def chat_interface(context, memory):
    sec("Ask the AI Agents", "💬")

    st.markdown(
        '<div class="info-box">Ask anything about your reviews. '
        'The right specialist agent will answer automatically.</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(4)
    quick = [
        "What are the biggest problems?",
        "Which cluster needs attention?",
        "How do verified buyers differ?",
        "Give me an executive report",
    ]
    for col, q in zip(cols, quick):
        if col.button(q, use_container_width=True, key=f"q_{q[:15]}"):
            st.session_state.pending_query = q

    st.divider()
    render_chat_history()

    user_input = None
    if st.session_state.pending_query:
        user_input = st.session_state.pending_query
        st.session_state.pending_query = None

    typed = st.chat_input("Type your question here…")
    if typed:
        user_input = typed

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role":"user","content":user_input,"agent":""})
        memory.append({"role":"user","content":user_input})

        with st.spinner("Agents analysing…"):
            responses = orchestrate_query(user_input, context, memory)

        for r in responses:
            ag   = r["agent"]
            meta = AGENT_META.get(ag, AGENT_META["orchestrator"])
            with st.chat_message("assistant"):
                st.markdown(
                    f'<span class="agent-badge" style="background:rgba(255,255,255,0.06);'
                    f'color:{meta["color"]}">{meta["label"]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(r["content"])

            st.session_state.chat_history.append({"role":"assistant","content":r["content"],"agent":ag})
            memory.append({"role":"assistant","content":r["content"][:200]})

        st.session_state.session_memory = memory[-12:]


def main():

    st.markdown("# 🤖 Review Intelligence")
    st.markdown(
        '<div class="info-box">'
        'Unsupervised ML + Multi-Agent AI · No labels required · '
        'Upload a CSV to get started'
        '</div>',
        unsafe_allow_html=True
    )

    with st.expander("⚙️ Pipeline Settings — click to customise", expanded=False):
        st.markdown("##### Analysis Settings")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            n_tfidf      = st.slider("TF-IDF vocabulary size", 100, 1000, 500, 50,
                                     help="How many unique words to analyse. Higher = more detail, slower.")
            k_min,k_max  = st.slider("Cluster count range", 2, 12, (2,7),
                                     help="Min and max number of groups to try. The best is auto-selected.")
        with sc2:
            n_topics     = st.slider("Number of topics", 3, 10, 5,
                                     help="How many recurring themes to extract from reviews.")
            topic_method = st.radio("Topic model", ["nmf","lda"], horizontal=True,
                                    help="NMF is faster; LDA is more probabilistic.")
        with sc3:
            dim_method   = st.radio("Cluster map view", ["PCA","t-SNE"], horizontal=True,
                                    help="How to flatten clusters into 2D for the scatter chart.")
            anomaly_pct  = st.slider("Anomaly sensitivity %", 1, 20, 5,
                                     help="% of reviews to flag as anomalous. Higher = more flags.") / 100

        st.markdown("##### Alert Thresholds")
        ta1,ta2,ta3,ta4 = st.columns(4)
        thr_anomaly = ta1.slider("Max anomaly %",    1,  30, 10,
                                  help="Alert if anomaly rate exceeds this.")
        thr_rating  = ta2.slider("Min avg rating",   1.0,4.0,2.5,0.1,
                                  help="Alert if overall avg rating drops below this.")
        thr_cluster = ta3.slider("Min cluster rating",1.0,4.0,2.0,0.1,
                                  help="Alert if any large cluster avg rating drops below this.")
        thr_vp_gap  = ta4.slider("VP rating gap",    0.1,2.0,0.5,0.1,
                                  help="Alert if verified vs unverified gap exceeds this.")

    thresholds = {
        "anomaly_pct":thr_anomaly, "avg_rating":thr_rating,
        "cluster_rating":thr_cluster, "vp_gap":thr_vp_gap
    }

    uploaded = st.file_uploader(
        "Upload CSV — needs `review_text` column · optional: `rating`, `verified_purchase`, `product_variant`, `date`",
        type=["csv"]
    )
    if uploaded is None:
        st.divider()
        ic1,ic2,ic3,ic4 = st.columns(4)
        ic1.markdown('<div class="ag-card"><div class="ag-icon">📊</div><div class="ag-name">Cluster Analysis</div><div class="ag-desc">Groups similar reviews automatically — no labels needed</div></div>',unsafe_allow_html=True)
        ic2.markdown('<div class="ag-card"><div class="ag-icon">🧩</div><div class="ag-name">Topic Modeling</div><div class="ag-desc">Finds recurring themes: battery, camera, pricing…</div></div>',unsafe_allow_html=True)
        ic3.markdown('<div class="ag-card"><div class="ag-icon">⚠️</div><div class="ag-name">Anomaly Detection</div><div class="ag-desc">Flags fake, extreme, or suspicious reviews</div></div>',unsafe_allow_html=True)
        ic4.markdown('<div class="ag-card"><div class="ag-icon">💬</div><div class="ag-name">AI Agent Chat</div><div class="ag-desc">Ask questions in plain English, get data-backed answers</div></div>',unsafe_allow_html=True)
        return

    df, dupes = load_data(uploaded)
    if df.empty: st.error("No data after cleaning."); return
    st.success(f"✅ Loaded **{len(df):,}** reviews · {dupes} duplicates removed")

    prog = st.progress(0, text="Starting analysis…")

    with st.spinner("Building text features…"):
        X_norm,X_text,tfidf,svd = build_features(
            tuple(df["review_text"].values), tuple(df["rating"]),
            tuple(df["verified_purchase"]), tuple(df["product_variant"]), n_tfidf)
    prog.progress(20, "Finding optimal clusters…")

    best_k,ks,inertias,sils = find_optimal_k(X_norm,(k_min,k_max))
    prog.progress(40, "Running clustering algorithms…")

    clustering = run_clustering(X_norm, best_k)
    prog.progress(55, "Extracting topics…")

    topics,doc_topics = run_topics(tuple(df["review_text"]), n_topics, topic_method)
    prog.progress(70, "Detecting anomalies…")

    anomaly_flags,iso_scores = detect_anomalies(X_norm, anomaly_pct)
    prog.progress(85, "Reducing dimensions for visualisation…")

    km_labels=clustering["kmeans"]["labels"]; km_sil=clustering["kmeans"]["silhouette"]
    db_labels=clustering["dbscan"]["labels"]
    df["cluster"]=km_labels; df["anomaly"]=anomaly_flags
    df["iso_score"]=iso_scores; df["dominant_topic"]=doc_topics.argmax(axis=1)

    cluster_info      = interpret_clusters(df,km_labels,tfidf,X_text,best_k)
    anomaly_count     = int(anomaly_flags.sum())
    anomaly_pct_actual= round(anomaly_count/len(df)*100,1)

    if dim_method=="t-SNE":
        Xp=PCA(n_components=min(50,X_norm.shape[1]),random_state=42).fit_transform(X_norm)
        idx=np.random.choice(len(X_norm),min(2000,len(X_norm)),replace=False)
        X2d=TSNE(n_components=2,random_state=42,perplexity=30).fit_transform(Xp[idx])
        lbl_s=km_labels[idx]
    else:
        X2d=PCA(n_components=2,random_state=42).fit_transform(X_norm); lbl_s=km_labels

    prog.progress(100, "Done!"); prog.empty()

    ctx=build_analysis_context(df,cluster_info,topics,doc_topics,anomaly_flags,
                               iso_scores,km_sil,best_k,anomaly_pct_actual,clustering["dbscan"])
    st.session_state.analysis_context=ctx
    alerts=check_alerts(ctx,thresholds)

    if alerts:
        sec("Active Alerts","🚨")
        for a in alerts:
            cls="risk-high" if a["level"]=="HIGH" else "risk-med"
            st.markdown(f'<span class="{cls}">{a["level"]}</span> &nbsp; {a["msg"]}',unsafe_allow_html=True)

    sec("Overview","📊")
    o1,o2,o3,o4,o5,o6 = st.columns(6)
    o1.metric("Total Reviews",   f"{len(df):,}")
    o2.metric("Avg Rating",      f"{ctx['avg_rating']} / 5")
    o3.metric("Customer Groups", best_k,  help="Clusters found automatically")
    o4.metric("Cluster Quality", f"{km_sil:.2f}", help="Silhouette score: closer to 1 = well-separated clusters")
    o5.metric("Topics Found",    n_topics)
    o6.metric("Unusual Reviews", f"{anomaly_count} ({anomaly_pct_actual}%)")

    render_cluster_plain_english(cluster_info, anomaly_pct_actual)

    sec("Customer Groups (Clusters)","🔍")
    st.caption("Each tab is one automatically-discovered group of customers. No manual labelling was used.")

    ctabs = st.tabs([f"Group {c}: {info['label']}" for c,info in cluster_info.items()])
    for tab,(c,info) in zip(ctabs,cluster_info.items()):
        with tab:
            rating=info["avg_rating"]
            mood=("😊 Generally happy" if rating>=4 else "😐 Mixed feelings" if rating>=3 else "😠 Mostly unhappy")
            st.markdown(
                f'<div class="summary-box">'
                f'<b>{mood}</b> · <strong>{info["size"]:,} reviews</strong> ({info["pct"]}% of total) · '
                f'Average rating: <strong>{rating}/5</strong> · '
                f'{info["vp_pct"]}% are verified purchases'
                f'</div>',
                unsafe_allow_html=True
            )
            m1,m2,m3 = st.columns(3)
            m1.metric("Reviews",    f"{info['size']:,}")
            m2.metric("Avg Rating", f"{rating} ⭐")
            m3.metric("Verified %", f"{info['vp_pct']}%")

            st.markdown("**What this group talks about:**")
            kw_color = "kw-red" if rating < 3 else ("kw-green" if rating >= 4 else "kw-gold")
            st.markdown(
                " ".join(pill(w, kw_color) for w in info["top_words"]),
                unsafe_allow_html=True
            )
            st.markdown("**Sample reviews from this group:**")
            for s in info["samples"][:3]:
                st.markdown(f"> *{s[:220]}{'…' if len(s)>220 else ''}*")

    sec("Charts & Visualisations","📈")

    v1,v2 = st.columns(2)
    with v1:
        st.caption("📍 Each dot is a review — reviews close together are similar")
        st.pyplot(fig_scatter(X2d,lbl_s,cluster_info,f"Review Clusters ({dim_method})"))
    with v2:
        st.caption("📊 Bar length = number of reviews · Colour = average rating")
        st.pyplot(fig_sizes(cluster_info))

    v3,v4 = st.columns(2)
    with v3:
        st.caption("📉 Shows how we chose the optimal number of groups")
        st.pyplot(fig_elbow(ks,inertias,sils,best_k))
    with v4:
        st.caption("📏 How well-separated each cluster is — wider = more distinct")
        st.pyplot(fig_silhouette(X_norm,km_labels,best_k))

    v5,v6 = st.columns(2)
    with v5:
        st.caption("🧩 How often each recurring theme appears across all reviews")
        st.pyplot(fig_topics(topics,doc_topics))
    with v6:
        rd=fig_rating(ctx)
        if rd:
            st.caption("⭐ Distribution of star ratings given by customers")
            st.pyplot(rd)

    st.caption("🌳 Tree diagram showing how review groups are hierarchically related")
    st.pyplot(fig_dendro(X_norm))

    sec("Clustering Methods Compared","🏆")
    st.caption("Three different algorithms were run. Here's how they each performed.")
    qt1,qt2,qt3 = st.tabs(["KMeans (primary)","DBSCAN (density)","Agglomerative (tree)"])

    with qt1:
        st.markdown(
            '<div class="summary-box">KMeans divides reviews into a fixed number of groups. '
            'It\'s the primary algorithm used throughout this dashboard.</div>',
            unsafe_allow_html=True
        )
        st.metric("Silhouette Score", f"{km_sil:.3f}", help="0.5+ is good · 0.7+ is great")
        st.metric("Groups found", best_k)
        st.dataframe(pd.DataFrame([
            {"Group":f"C{c}","Theme":i["label"],"Reviews":i["size"],
             "Share":f"{i['pct']}%","Avg Rating":i["avg_rating"],"Verified %":f"{i['vp_pct']}%"}
            for c,i in cluster_info.items()
        ]),use_container_width=True,hide_index=True)

    with qt2:
        st.markdown(
            '<div class="summary-box">DBSCAN finds clusters based on density — '
            'it doesn\'t need a fixed k and can identify "noise" (outlier) points.</div>',
            unsafe_allow_html=True
        )
        d1,d2=st.columns(2)
        d1.metric("Clusters found", clustering["dbscan"]["n_clusters"])
        d2.metric("Noise/outlier reviews", clustering["dbscan"]["noise"])
        db_ph={c:cluster_info.get(c,{"label":f"Group {c}"}) for c in set(db_labels) if c!=-1}
        st.pyplot(fig_scatter(X2d[:len(db_labels)],db_labels[:len(X2d)],db_ph,"DBSCAN Clusters"))

    with qt3:
        st.markdown(
            '<div class="summary-box">Agglomerative clustering builds a hierarchy — '
            'it\'s shown in the tree diagram above. Good for understanding nested relationships.</div>',
            unsafe_allow_html=True
        )
        al=clustering["agglomerative"]["labels"]
        st.dataframe(
            pd.Series(al).value_counts().sort_index()
            .rename_axis("Cluster").reset_index(name="Reviews"),
            use_container_width=True,hide_index=True
        )

    sec("Recurring Themes in Reviews","🧩")
    render_topic_plain_english(topics)
    tcols=st.columns(min(n_topics,3))
    colors=["","kw-red","kw-purple","kw-gold","kw-green",""]
    for i,words in topics.items():
        with tcols[i%len(tcols)]:
            st.markdown(f"**Theme {i+1}**")
            st.markdown(
                " ".join(pill(w,colors[i%len(colors)]) for w in words[:8]),
                unsafe_allow_html=True
            )

    sec("Unusual & Suspicious Reviews","⚠️")
    render_anomaly_plain_english(anomaly_count, anomaly_pct_actual, len(df))

    if anomaly_pct_actual>10: st.error(f"🚨 HIGH — {anomaly_pct_actual}% flagged as unusual")
    elif anomaly_pct_actual>5: st.warning(f"⚠️ MODERATE — {anomaly_pct_actual}% flagged")
    else: st.success(f"✅ LOW — only {anomaly_pct_actual}% flagged as unusual")

    st.pyplot(fig_anomaly(iso_scores,anomaly_flags))
    anom_df=df[df["anomaly"]==1].sort_values("iso_score").head(20)
    scols=[c for c in ["review_text","rating","cluster","iso_score","dominant_topic"] if c in anom_df.columns]
    st.dataframe(anom_df[scols].reset_index(drop=True),use_container_width=True,hide_index=True)

    sec("AI Agents","🧠")
    st.caption("Four specialist agents analyse your data from different angles. "
               "The Orchestrator synthesises their answers when multiple agents respond.")
    ac1,ac2,ac3,ac4 = st.columns(4)
    for col, (key, icon, name, desc) in zip([ac1,ac2,ac3,ac4],[
        ("data_analyst",  "📊","Data Analyst",  "Explains what each cluster means and why it exists"),
        ("risk_analyst",  "⚠️","Risk Analyst",  "Finds product defects and customer pain points"),
        ("biz_strategist","💡","Strategist",    "Turns data into product and marketing actions"),
        ("report_gen",    "📝","Report Writer", "Writes concise executive-level summaries"),
    ]):
        color=AGENT_META[key]["color"]
        col.markdown(
            f'<div class="ag-card">'
            f'<div class="ag-icon">{icon}</div>'
            f'<div class="ag-name" style="color:{color}">{name}</div>'
            f'<div class="ag-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    chat_interface(ctx, st.session_state.session_memory)

    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_history=[]
        st.session_state.session_memory=[]
        st.rerun()

    sec("Auto-Generated Executive Report","📋")
    r1,r2 = st.columns([1,3])
    with r1:
        if st.button("🤖 Generate Report", use_container_width=True):
            with st.spinner("Writing executive briefing…"):
                rt=generate_report(ctx,st.session_state.session_memory)
            st.session_state.report_cache={"text":rt,"ts":datetime.now().strftime("%H:%M:%S")}
        if st.session_state.report_cache:
            st.caption(f"Generated at {st.session_state.report_cache['ts']}")
    with r2:
        if st.session_state.report_cache:
            rt=st.session_state.report_cache["text"]
            st.markdown(f'<div class="report-wrap">', unsafe_allow_html=True)
            st.markdown(rt)
            st.markdown('</div>', unsafe_allow_html=True)
            st.download_button("⬇ Download as Markdown",data=rt.encode(),
                               file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                               mime="text/markdown")

    sec("Explore the Data","🗂️")
    flt=st.selectbox("Filter by group",
                     ["All groups"]+[f"Group {c}: {cluster_info[c]['label']}" for c in cluster_info])
    exp_df=df.copy()
    if flt!="All groups":
        cn=int(flt.split(":")[0].replace("Group","").strip())
        exp_df=exp_df[exp_df["cluster"]==cn]
    dcols=[c for c in ["review_text","rating","cluster","dominant_topic","anomaly",
                        "verified_purchase","product_variant"] if c in exp_df.columns]
    st.dataframe(exp_df[dcols].head(50).reset_index(drop=True),use_container_width=True,hide_index=True)
    st.caption(f"Showing 50 of {len(exp_df):,} reviews in this group")

    sec("Export","⬇")
    e1,e2 = st.columns(2)
    with e1:
        st.download_button("⬇ Download Enriched CSV",
                           data=df.to_csv(index=False).encode(),
                           file_name=f"reviews_enriched_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv",use_container_width=True)
    with e2:
        st.download_button("⬇ Download Analysis JSON",
                           data=json.dumps(ctx,indent=2,default=str).encode(),
                           file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                           mime="application/json",use_container_width=True)


if __name__ == "__main__":
    main()