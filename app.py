import os
import ast
import re
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Global holders
data = None
vectorizer = None
X = None
token_sets = None

APP_TITLE = "🍳 Ingredient-to-Dish Recommender"
APP_VERSION = "v3.1"

st.set_page_config(
    page_title="Dish Recommender",
    page_icon="🍳",
    layout="wide",
)
# -----------------------------
# Google AdSense injection
# -----------------------------
adsense_code = """
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2965987453887789"
     crossorigin="anonymous"></script>
"""
st.markdown(adsense_code, unsafe_allow_html=True)
# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
      :root { --card-bg: rgba(127,127,127,0.06); --chip-bg: rgba(127,127,127,0.16); }
      @media (prefers-color-scheme: dark) {
        :root { --card-bg: rgba(255,255,255,0.05); --chip-bg: rgba(255,255,255,0.12); }
      }
      .recipe-card {
        border: 1px solid rgba(127,127,127,0.25);
        border-radius: 12px;
        padding: 14px 16px;
        background: var(--card-bg);
        margin-bottom: 12px;
      }
      .recipe-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; }
      .chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 2px 0; }
      .chip {
        display: inline-block; padding: 2px 8px; border-radius: 999px;
        background: var(--chip-bg); font-size: 0.85rem;
      }
      .meta { font-size: 0.90rem; opacity: 0.9; }
      .kpi { display:inline-block; padding: 8px 10px; border-radius: 8px; background: var(--card-bg); margin-right: 8px; }
      .separator { height: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(f"Build {APP_VERSION} — Local dataset, fixed scoring, and full recipe guides.")

# -----------------------------
# Text normalisation
# -----------------------------
PHRASE_NORMALIZATIONS = [
    (r"\bscallions?\b", "green onion"),
    (r"\bgarbanzo beans?\b|\bchick ?peas?\b", "chickpea"),
    (r"\baubergines?\b", "eggplant"),
    (r"\bcourgettes?\b", "zucchini"),
    (r"\bbell ?peppers?\b|\bcapsicums?\b", "bell pepper"),
    (r"\bcoriander leaves?\b", "cilantro"),
    (r"\bconfectioners'? sugar\b|\bpowdered sugar\b", "icing sugar"),
]

WORD_RE = re.compile(r"[^a-zA-Z\s]+")


def _safe_literal_eval(x: Any):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def normalize_phrase(text: str) -> str:
    t = text.lower()
    for pat, repl in PHRASE_NORMALIZATIONS:
        t = re.sub(pat, repl, t)
    t = WORD_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def simple_lemmatize(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    t = normalize_phrase(text)
    toks = [simple_lemmatize(tok) for tok in t.split() if tok.strip()]
    return toks


def stringify_ingredient_list(ing_list) -> str:
    if isinstance(ing_list, list):
        parts = []
        for it in ing_list:
            if isinstance(it, str) and it.strip():
                parts.extend(tokenize(it))
        return " ".join(parts)
    elif isinstance(ing_list, str):
        parsed = _safe_literal_eval(ing_list)
        if isinstance(parsed, list):
            return stringify_ingredient_list(parsed)
        return " ".join(tokenize(ing_list))
    return ""


# -----------------------------
# Data loading and preparation
# -----------------------------

@st.cache_data(show_spinner=True)
def load_dataset_from_csv(file: str) -> pd.DataFrame:
    """
    Load a CSV robustly:
    - if it's actually a zip, read the first CSV inside
    - try utf-8, then latin-1
    - auto-detect delimiter
    - skip broken lines instead of crashing
    - if everything fails, fall back to the demo dataset
    """
    # 1) If it's actually a zip (e.g. misnamed .csv), handle that
    try:
        if _is_zip_source(file):
            for enc in ("utf-8", "latin-1"):
                try:
                    return _read_csv_from_zip(file, encoding=enc)
                except Exception:
                    continue
    except Exception:
        # if detection fails, ignore and try normal CSV logic
        pass

    # 2) Normal CSV: try multiple encodings, flexible parser
    last_err = None
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(
                file,
                encoding=enc,
                sep=None,            # let pandas sniff the delimiter
                engine="python",     # more forgiving parser
                on_bad_lines="skip", # skip malformed rows
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            st.warning(f"Parser/encoding error with encoding={enc}: {e}")
            continue
        except Exception as e:
            last_err = e
            st.warning(f"Unexpected CSV error with encoding={enc}: {e}")
            continue

    # 3) If we got here, we couldn't load the file -> fall back to demo
    st.warning(f"⚠️ Could not load {file}. Falling back to demo dataset. Last error: {last_err}")
    return demo_dataset()



def _choose_columns(df: pd.DataFrame) -> List[str]:
    base_cols = ["name", "ingredients"]
    optional = [
        "minutes",
        "n_steps",
        "n_ingredients",
        "tags",
        "description",
        "steps",
        "rating",
        "n_reviews",
    ]
    cols: List[str] = []
    for c in base_cols:
        if c in df.columns:
            cols.append(c)
    for c in optional:
        if c in df.columns:
            cols.append(c)
    return cols


def _parse_listish(col: pd.Series) -> List:
    out = []
    for x in col:
        parsed = _safe_literal_eval(x)
        out.append(parsed)
    return out


def prepare_data(df: pd.DataFrame, max_features: int) -> Tuple[pd.DataFrame, TfidfVectorizer, Any, List[set]]:
    """
    Clean, normalize, and vectorize ingredients.
    No caching here to avoid multiple big models piling up in memory.
    """
    df = _coerce_columns(df)
    cols = _choose_columns(df)
    if "name" not in cols or "ingredients" not in cols:
        raise ValueError(
            "CSV must include at least 'name' and 'ingredients' columns. "
            f"Found columns (after coercion): {list(df.columns)}"
        )

    data_local = df[cols].dropna(subset=["name", "ingredients"]).copy()

    ing_clean_texts, token_sets_local = [], []
    for raw in data_local["ingredients"]:
        s = stringify_ingredient_list(raw)
        ing_clean_texts.append(s)
        token_sets_local.append(set(s.split()))
    data_local["ingredients_norm"] = ing_clean_texts

    if "tags" in data_local.columns:
        data_local["tags_list"] = _parse_listish(data_local["tags"])

    if "steps" in data_local.columns:
        parsed_steps = []
        for val in data_local["steps"]:
            v = _safe_literal_eval(val)
            if isinstance(v, list):
                parsed_steps.append(v)
            elif isinstance(v, str):
                parts = [s.strip() for s in re.split(r"\n+|\.\s+", v) if s.strip()]
                parsed_steps.append(parts)
            else:
                parsed_steps.append([])
        data_local["steps_list"] = parsed_steps

    data_local = data_local[data_local["ingredients_norm"].str.len() > 0].reset_index(drop=True)

    # Use float32 to reduce memory
    vectorizer_local = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        dtype=np.float32,
    )
    X_local = vectorizer_local.fit_transform(data_local["ingredients_norm"])

    return data_local, vectorizer_local, X_local, token_sets_local


# -----------------------------
# "Settings" section (was sidebar)
# -----------------------------
with st.expander("⚙️ Settings", expanded=False):
    st.subheader("Performance")
    fast_mode = st.checkbox(
        "Fast mode (limit rows for quicker loading)",
        value=True,
        help="Process only the first N rows to reduce build time and memory.",
    )
    fast_mode_cap = st.number_input(
        "Rows to process in fast mode",
        min_value=1000,
        max_value=500000,
        value=15000,  # lowered default for memory
        step=1000,
    )
    tfidf_max_features = st.slider(
        "TF-IDF feature cap",
        min_value=5000,
        max_value=60000,
        value=20000,  # lowered default for memory
        step=5000,
    )
    st.caption("Built with Streamlit + scikit‑learn using the local `recipes.csv` file.")


# -----------------------------
# Load local recipes.csv and build model
# -----------------------------
if not os.path.exists("recipes.csv"):
    st.error("Could not find `recipes.csv` in the app directory.")
    st.stop()

df_raw = load_dataset_from_csv("recipes.csv")

if df_raw is None or len(df_raw) == 0:
    st.error("Loaded `recipes.csv` is empty.")
    st.stop()

df_for_processing = df_raw
row_cap_applied = False
if fast_mode and fast_mode_cap > 0 and len(df_raw) > fast_mode_cap:
    df_for_processing = df_raw.head(int(fast_mode_cap)).copy()
    row_cap_applied = True

try:
    with st.spinner("Preparing and vectorizing recipes..."):
        data, vectorizer, X, token_sets = prepare_data(df_for_processing, int(tfidf_max_features))
except Exception as e:
    st.error(f"❌ Error preparing data: {e}")
    st.stop()

if row_cap_applied:
    st.info(
        f"⚡ Fast mode: using the first {fast_mode_cap:,} recipes out of {len(df_raw):,}."
    )

# KPIs
total_recipes = len(data)
with st.container():
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi'><b>Recipes</b><br>{total_recipes:,}</div>", unsafe_allow_html=True)
    has_time = "minutes" in data.columns
    has_steps_col = "n_steps" in data.columns
    has_n_ings = "n_ingredients" in data.columns
    k2.markdown(
        f"<div class='kpi'><b>Fields</b><br>"
        f"{'⏱️ minutes ' if has_time else ''}"
        f"{'📝 steps ' if has_steps_col else ''}"
        f"{'🥘 ingredients ' if has_n_ings else ''}</div>",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"<div class='kpi'><b>Vector Size</b><br>{X.shape[1]:,} features</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

# -----------------------------
# Query controls
# -----------------------------
left, right = st.columns([1.6, 1.0])

with left:
    st.subheader("Enter your ingredients")
    user_input = st.text_input(
        "Comma-separated list (e.g., egg, flour, milk, butter)",
        value="egg, flour, milk",
        help="We will match these against recipe ingredients",
    )
    exclude_input = st.text_input(
        "Exclude ingredients (optional)",
        value="",
        help="Comma-separated items to avoid (e.g., nuts, pork, cilantro).",
    )

with right:
    st.subheader("Result options")
    top_n = st.number_input("Number of results", 1, 50, 10, step=1)

    max_missing = st.slider("Max missing ingredients (per recipe)", 0, 50, 10, 1)

    max_time = None
    if "minutes" in data.columns:
        max_time = st.slider("Max time (minutes)", 0, int(np.nanmax(data["minutes"])), 60, 5)

    max_steps = None
    if "n_steps" in data.columns:
        max_steps = st.slider("Max steps", 0, int(np.nanmax(data["n_steps"])), 12, 1)

    max_ings = None
    if "n_ingredients" in data.columns:
        max_ings = st.slider("Max ingredient count", 0, int(np.nanmax(data["n_ingredients"])), 15, 1)

    tag_filter = []
    if "tags_list" in data.columns:
        st.caption("Filter by tags (optional)")
        all_tags = set()
        for lst in data["tags_list"].dropna():
            if isinstance(lst, list):
                for t in lst:
                    if isinstance(t, str):
                        all_tags.add(t.lower())
                        if len(all_tags) > 500:
                            break
            if len(all_tags) > 500:
                break
        tag_filter = st.multiselect("Tags contain any of:", sorted(all_tags))


# -----------------------------
# Recommendation logic
# -----------------------------
def parse_ingredient_input(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    toks = []
    for p in parts:
        toks.extend(tokenize(p))
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def compute_scores(user_terms, exclude_terms) -> pd.DataFrame:
    if vectorizer is None or X is None or data is None:
        st.error("The model is not initialized properly.")
        return pd.DataFrame()
    if not user_terms:
        st.warning("Please enter at least one ingredient.")
        return pd.DataFrame()

    user_vec = vectorizer.transform([" ".join(user_terms)])
    if X.shape[1] == 0:
        st.error("TF-IDF matrix has 0 features.")
        return pd.DataFrame()

    sims = cosine_similarity(user_vec, X).flatten()

    u_set = set(user_terms)
    excl_set = set(exclude_terms)

    overlaps = np.zeros(len(token_sets), dtype=np.int32)
    missings = np.zeros(len(token_sets), dtype=np.int32)
    has_excluded = np.zeros(len(token_sets), dtype=bool)

    for i, tset in enumerate(token_sets):
        ov = len(u_set & tset)
        overlaps[i] = ov
        missings[i] = max(len(u_set) - ov, 0)
        if excl_set and (len(excl_set & tset) > 0):
            has_excluded[i] = True

    user_len = max(len(u_set), 1)
    overlap_frac = overlaps / user_len
    missing_frac = missings / user_len

    # Fixed weights (no sliders): tuned for sensible behavior
    alpha, beta, gamma = 0.7, 0.25, 0.1
    hybrid = alpha * sims + beta * overlap_frac - gamma * missing_frac

    out = data.copy()
    out["similarity"] = sims
    out["overlap"] = overlaps
    out["missing"] = missings
    out["overlap_frac"] = overlap_frac
    out["missing_frac"] = missing_frac
    out["score"] = hybrid
    out["has_excluded"] = has_excluded

    return out


def apply_filters(
    df: pd.DataFrame,
    max_missing_ing: int,
    max_minutes,
    max_steps_local,
    max_ing_count,
    required_tags: List[str],
) -> pd.DataFrame:
    filt = df[~df["has_excluded"]]
    filt = filt[filt["missing"] <= max_missing_ing]

    if max_minutes is not None and "minutes" in filt.columns:
        filt = filt[pd.to_numeric(filt["minutes"], errors="coerce").fillna(np.inf) <= max_minutes]
    if max_steps_local is not None and "n_steps" in filt.columns:
        filt = filt[pd.to_numeric(filt["n_steps"], errors="coerce").fillna(np.inf) <= max_steps_local]
    if max_ing_count is not None and "n_ingredients" in filt.columns:
        filt = filt[pd.to_numeric(filt["n_ingredients"], errors="coerce").fillna(np.inf) <= max_ing_count]

    if required_tags and "tags_list" in filt.columns:
        req = set([t.lower() for t in required_tags])
        mask = filt["tags_list"].apply(
            lambda lst: isinstance(lst, list) and any((isinstance(x, str) and x.lower() in req) for x in lst)
        )
        filt = filt[mask]

    return filt


# -----------------------------
# Run search
# -----------------------------
run = st.button("🔎 Find Dishes", type="primary", use_container_width=True)

if run:
    user_terms = parse_ingredient_input(user_input)
    exclude_terms = parse_ingredient_input(exclude_input) if exclude_input else []
    if len(user_terms) == 0:
        st.warning("Please enter at least one ingredient.")
        st.stop()

    with st.spinner("Matching your ingredients to recipes…"):
        raw_scores = compute_scores(user_terms, exclude_terms)
        filtered = apply_filters(
            raw_scores,
            max_missing_ing=max_missing,
            max_minutes=max_time if has_time else None,
            max_steps_local=max_steps if has_steps_col else None,
            max_ing_count=max_ings if has_n_ings else None,
            required_tags=tag_filter,
        )

        sort_choice = st.selectbox(
            "Sort results by",
            [
                "Hybrid Score (best overall)",
                "Best TF‑IDF Similarity",
                "Most Overlap",
                "Fewest Missing",
                "Quickest Time",
                "Easiest Steps",
                "Simplest (# ingredients)",
            ],
            index=0,
        )

        if sort_choice == "Hybrid Score (best overall)":
            sorted_df = filtered.sort_values("score", ascending=False)
        elif sort_choice == "Best TF‑IDF Similarity":
            sorted_df = filtered.sort_values("similarity", ascending=False)
        elif sort_choice == "Most Overlap":
            sorted_df = filtered.sort_values(["overlap", "similarity"], ascending=[False, False])
        elif sort_choice == "Fewest Missing":
            sorted_df = filtered.sort_values(["missing", "similarity"], ascending=[True, False])
        elif sort_choice == "Quickest Time" and has_time:
            sorted_df = filtered.sort_values(["minutes", "similarity"], ascending=[True, False])
        elif sort_choice == "Easiest Steps" and has_steps_col:
            sorted_df = filtered.sort_values(["n_steps", "similarity"], ascending=[True, False])
        elif sort_choice == "Simplest (# ingredients)" and has_n_ings:
            sorted_df = sorted_df = filtered.sort_values(["n_ingredients", "similarity"], ascending=[True, False])
        else:
            sorted_df = filtered.sort_values("score", ascending=False)

        results = sorted_df.head(top_n)

        st.subheader("🍽️ Top Recipe Suggestions")
        st.caption(
            f"Showing <b>{len(results)}</b> of <b>{len(filtered)}</b> matched recipes "
            f"(from {total_recipes:,} total).",
            unsafe_allow_html=True,
        )

        if results.empty:
            st.info("No matches found with current filters. Try allowing more missing ingredients.")
            st.stop()

        def render_card(row):
            parts = []
            parts.append("<div class='recipe-card'>")
            title = str(row.get("name", "Untitled Recipe"))
            parts.append(f"<div class='recipe-title'>{title}</div>")

            meta_bits = []
            if "minutes" in row and pd.notna(row["minutes"]):
                meta_bits.append(f"⏱️ {int(row['minutes'])} min")
            if "n_steps" in row and pd.notna(row.get("n_steps", np.nan)):
                meta_bits.append(f"📝 {int(row['n_steps'])} steps")
            if "n_ingredients" in row and pd.notna(row.get("n_ingredients", np.nan)):
                meta_bits.append(f"🥘 {int(row['n_ingredients'])} ingredients")
            meta_bits.append(f"🔎 sim {row['similarity']:.2f}")
            meta_bits.append(f"✨ score {row['score']:.2f}")
            parts.append(f"<div class='meta'>{' | '.join(meta_bits)}</div>")

            if "ingredients_norm" in row and isinstance(row["ingredients_norm"], str):
                recipe_terms = set(row["ingredients_norm"].split())
                ov_terms = [t for t in user_terms if t in recipe_terms]
                if ov_terms:
                    chips = "".join([f"<span class='chip'>{t}</span>" for t in ov_terms[:12]])
                    parts.append(f"<div class='chips'>{chips}</div>")

            parts.append("</div>")
            st.markdown("".join(parts), unsafe_allow_html=True)

            with st.expander("Full recipe guide", expanded=False):
                has_any_detail = False

                if "description" in row and isinstance(row["description"], str) and row["description"].strip():
                    st.markdown(f"**Description:** {row['description']}")
                    has_any_detail = True

                if "ingredients" in row:
                    raw_ings = row["ingredients"]
                    ings = _safe_literal_eval(raw_ings) if isinstance(raw_ings, str) else raw_ings
                    st.markdown("**Ingredients:**")
                    if isinstance(ings, list):
                        pretty = sorted(set([str(i).strip() for i in ings if str(i).strip()]))
                        st.write(", ".join(pretty))
                    elif isinstance(ings, str):
                        st.write(ings)
                    has_any_detail = True

                steps_shown = False
                if "steps_list" in row and isinstance(row["steps_list"], list) and len(row["steps_list"]) > 0:
                    st.markdown("**Steps:**")
                    for idx, s in enumerate(row["steps_list"], 1):
                        st.write(f"{idx}. {s}")
                    steps_shown = True
                    has_any_detail = True
                elif "steps" in row and isinstance(row["steps"], str) and row["steps"].strip():
                    parts_steps = [s.strip() for s in re.split(r"\n+|\.\s+", row["steps"]) if s.strip()]
                    if parts_steps:
                        st.markdown("**Steps:**")
                        for idx, s in enumerate(parts_steps, 1):
                            st.write(f"{idx}. {s}")
                        steps_shown = True
                        has_any_detail = True

                if not has_any_detail:
                    st.caption("No additional details available in this dataset row.")

        grid_cols = st.columns(2)
        for i, (_, r) in enumerate(results.iterrows()):
            with grid_cols[i % 2]:
                render_card(r)

        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

else:
    st.info("Enter ingredients, adjust filters, and click **Find Dishes** to see suggestions.")
