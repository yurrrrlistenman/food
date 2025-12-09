
# Features:
# - Local file, upload, or demo dataset
# - TF-IDF + cosine similarity + ingredient overlap hybrid score
# - Filters: time, steps, ingredient count, similarity, exclusions
# - Optional tag filtering (if 'tags' column exists)
# - Beautiful card UI with badges + explainer
# - Download results as CSV
# -----------------------------------------------------------
import os
import ast
import re
import pandas as pd
import zipfile
import importlib
import subprocess
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any

import numpy as np
import streamlit as st

data = None
vectorizer = None
X = None
token_sets = None

PANDAS_REPAIR_NOTE: str | None = None


def _import_pandas_with_repair():
    """
    Import pandas and attempt a local reinstall if the package is partially missing.
    This specifically catches cases where pandas import fails because submodules
    like pandas._config are absent (corrupted install).
    """
    global PANDAS_REPAIR_NOTE
    try:
        import pandas as pd
      
    except ModuleNotFoundError as exc:
        missing_target = getattr(exc, "name", "") or ""
        if not missing_target.startswith("pandas"):
            raise

        warn_msg = (
            f"Pandas is not installed correctly (missing '{missing_target}'). "
            "Attempting a quick reinstall so the app can continue."
        )
        PANDAS_REPAIR_NOTE = warn_msg

        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pandas"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            PANDAS_REPAIR_NOTE = (
                f"{warn_msg}\n\nAutomatic repair failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {result.stdout.strip()}\n"
                f"stderr: {result.stderr.strip()}\n"
                "Please reinstall pandas manually (e.g. `pip install --upgrade pandas`)."
            )
            raise RuntimeError(PANDAS_REPAIR_NOTE) from exc

        # Try import again after reinstall
        try:
            module = importlib.import_module("pandas")
        except ModuleNotFoundError as new_exc:
            PANDAS_REPAIR_NOTE = (
                f"{warn_msg}\n\n"
                "Automatic reinstall completed but pandas still fails to import. "
                "Please reinstall pandas manually (e.g. `pip install --upgrade --force-reinstall pandas`)."
            )
            raise RuntimeError(PANDAS_REPAIR_NOTE) from new_exc

        PANDAS_REPAIR_NOTE = (
            "Detected a broken pandas install and reinstalled it automatically. "
            "If you continue to see issues, please reinstall pandas manually."
        )
        return module


APP_TITLE = "🍳 Ingredient-to-Dish Recommender"
APP_VERSION = "v2.1"

st.set_page_config(
    page_title="Dish Recommender",
    page_icon="🍳",
    layout="wide",
)



import csv


def _reset_stream(handle: Any):
    """Seek to start for file-like sources (UploadedFile, BytesIO, etc.)."""
    try:
        handle.seek(0)
    except Exception:
        pass
        
def demo_df():
    return pd.DataFrame([
        {"name": "Fluffy Pancakes",
         "ingredients_norm": "flour milk egg sugar butter"},
        {"name": "Tomato Basil Pasta",
         "ingredients_norm": "pasta tomato garlic olive oil basil"},
    ])


def _is_zip_source(src: Any) -> bool:
    """Best-effort detection for misnamed zip files like 'recipes.csv'."""
    if isinstance(src, (str, os.PathLike)):
        return zipfile.is_zipfile(src)
    try:
        pos = src.tell()
    except Exception:
        return False
    try:
        src.seek(0)
        return zipfile.is_zipfile(src)
    finally:
        src.seek(pos)


def _read_csv_from_zip(src: Any, encoding: str) -> pd.DataFrame:
    """Open the first CSV member inside a zip archive into a DataFrame."""
    if hasattr(src, "seek"):
        src.seek(0)
    with zipfile.ZipFile(src) as zf:
        csv_members = [name for name in zf.namelist() if name.lower().endswith(".csv") and not name.endswith("/")]
        if not csv_members:
            raise ValueError("Zip archive does not contain a CSV file.")
        target = csv_members[0]
        with zf.open(target) as zipped_file:
            return pd.read_csv(zipped_file, encoding=encoding)

def clean_csv(input_file: str, output_file: str, expected_columns: int = None):
    """
    Cleans a CSV by fixing rows with too many or too few columns.
    
    Parameters:
    - input_file: path to original CSV
    - output_file: path to save cleaned CSV
    - expected_columns: number of columns expected (optional, will detect from header if None)
    """
    cleaned_rows = 0
    skipped_rows = 0

    with open(input_file, encoding="latin-1", newline='') as infile, \
         open(output_file, "w", encoding="latin-1", newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # detect number of columns from header if not provided
        if expected_columns is None:
            header = next(reader)
            expected_columns = len(header)
            writer.writerow(header)
        else:
            header = next(reader)
            if len(header) == expected_columns:
                writer.writerow(header)

        for row in reader:
            if len(row) == expected_columns:
                writer.writerow(row)
                cleaned_rows += 1
            elif len(row) > expected_columns:
                new_row = row[:expected_columns-1] + [",".join(row[expected_columns-1:])]
                writer.writerow(new_row)
                cleaned_rows += 1
            else:
                skipped_rows += 1

    print(f"CSV cleaned: {cleaned_rows} rows kept/fixed, {skipped_rows} rows skipped")


# Minimal CSS for clean cards & badges that adapts to light/dark
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
      .metric-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 8px 0; }
      .muted { opacity: 0.75; }
      .small { font-size: 0.90rem; }
      .kpi { display:inline-block; padding: 8px 10px; border-radius: 8px; background: var(--card-bg); margin-right: 8px; }
      .separator { height: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(f"Power user build {APP_VERSION} — TF‑IDF similarity + overlap, filters, and a polished display.")

if PANDAS_REPAIR_NOTE:
    st.warning(PANDAS_REPAIR_NOTE)


# -----------------------------
# Utilities: parsing & cleaning
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
    """Attempt to parse Python-literal-like strings to lists; otherwise return as-is."""
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
    # Remove punctuation/digits -> words only
    t = WORD_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def simple_lemmatize(token: str) -> str:
    """Very light rules: handles plural forms; avoids heavyweight NLP deps."""
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
    """Join a list of ingredient phrases into one normalized string."""
    if isinstance(ing_list, list):
        parts = []
        for it in ing_list:
            if isinstance(it, str) and it.strip():
                parts.extend(tokenize(it))
        return " ".join(parts)
    elif isinstance(ing_list, str):
        # maybe "[...]" or raw string
        parsed = _safe_literal_eval(ing_list)
        if isinstance(parsed, list):
            return stringify_ingredient_list(parsed)
        return " ".join(tokenize(ing_list))
    return ""


# -----------------------------
# Data loading
# -----------------------------

@st.cache_data(show_spinner=True)
def load_dataset_from_csv(file: str) -> pd.DataFrame:
    """
    Load a CSV robustly:
    - if it's a zip, read the first CSV inside
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
    optional = ["minutes", "n_steps", "n_ingredients", "tags", "description", "steps", "rating", "n_reviews"]
    cols = [c for c in base_cols if c in df.columns] + [c for c in optional if c in df.columns]
    # Try RAW_recipes.csv column names fallback
    if "name" not in cols and "title" in df.columns:
        cols.append("title")
    if "ingredients" not in cols and "ingredient" in df.columns:
        cols.append("ingredient")
    return cols

def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Unify title/name + ingredient column names if needed
    if "name" not in df.columns and "title" in df.columns:
        df = df.rename(columns={"title": "name"})
    if "ingredients" not in df.columns and "ingredient" in df.columns:
        df = df.rename(columns={"ingredient": "ingredients"})
    return df

def _parse_listish(col: pd.Series) -> List:
    out = []
    for x in col:
        parsed = _safe_literal_eval(x)
        out.append(parsed)
    return out

@st.cache_data(show_spinner=True)
def load_dataset_from_csv(file: str) -> pd.DataFrame:
    """
    Load a CSV robustly:
    - if it's a zip, read the first CSV inside
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



# -----------------------------
# Demo dataset (tiny but useful)
# -----------------------------
@st.cache_data
def demo_dataset() -> pd.DataFrame:
    recs = [
        {
            "name": "Fluffy Pancakes",
            "minutes": 15,
            "n_steps": 6,
            "n_ingredients": 7,
            "ingredients": ["all-purpose flour", "milk", "egg", "baking powder", "sugar", "salt", "butter"],
            "tags": ["breakfast", "vegetarian", "quick"],
            "description": "Simple, soft pancakes for breakfast.",
            "steps": ["Whisk dry", "Add wet", "Mix", "Rest", "Cook", "Serve"],
        },
        {
            "name": "Classic Omelette",
            "minutes": 10,
            "n_steps": 5,
            "n_ingredients": 5,
            "ingredients": ["egg", "milk", "butter", "salt", "pepper"],
            "tags": ["breakfast", "gluten-free", "quick"],
            "description": "A tender omelette with basic pantry items.",
            "steps": ["Beat eggs", "Heat pan", "Add butter", "Cook eggs", "Fold & serve"],
        },
        {
            "name": "Tomato Basil Pasta",
            "minutes": 20,
            "n_steps": 7,
            "n_ingredients": 8,
            "ingredients": ["spaghetti", "tomato", "garlic", "olive oil", "basil", "salt", "pepper", "parmesan"],
            "tags": ["dinner", "vegetarian"],
            "description": "Light, fresh pasta with tomato and basil.",
            "steps": ["Boil pasta", "Saute garlic", "Add tomato", "Simmer", "Combine", "Add basil", "Serve"],
        },
        {
            "name": "Chicken Stir-Fry",
            "minutes": 18,
            "n_steps": 8,
            "n_ingredients": 11,
            "ingredients": ["chicken breast", "soy sauce", "ginger", "garlic", "broccoli", "bell pepper", "onion", "cornstarch", "oil", "salt", "pepper"],
            "tags": ["dinner", "quick"],
            "description": "Fast stir-fry with a glossy sauce.",
            "steps": ["Slice chicken", "Prep veg", "Stir-fry", "Add sauce", "Thicken", "Finish", "Taste", "Serve"],
        },
        {
            "name": "Chocolate Chip Cookies",
            "minutes": 22,
            "n_steps": 9,
            "n_ingredients": 10,
            "ingredients": ["flour", "butter", "sugar", "brown sugar", "egg", "vanilla", "baking soda", "salt", "chocolate chips", "milk"],
            "tags": ["dessert", "snack"],
            "description": "Chewy, melty cookies for everyone.",
            "steps": ["Cream butter/sugars", "Add egg/vanilla", "Add dry", "Mix chips", "Scoop", "Bake", "Cool", "Enjoy", "Store"],
        },
    ]
    return pd.DataFrame(recs)


# -----------------------------
# Sidebar: Data source & Controls
# -----------------------------
with st.sidebar:
    st.subheader("Data Source")
    data_source = st.radio(
        "Choose dataset input:",
        ["Use local recipes.csv", "Upload CSV", "Use demo (built-in)"],
        index=0,
    )

    uploaded = None
    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV (must include at least 'name' and 'ingredients')", type=["csv"])

    st.markdown("---")
    st.subheader("Performance")
    fast_mode = st.checkbox(
        "Fast mode (limit rows for quicker loading)",
        value=True,
        help="Process only the first N rows to drastically reduce TF-IDF build time.",
    )
    fast_mode_cap = st.number_input(
        "Rows to process in fast mode",
        min_value=1000,
        max_value=500000,
        value=25000,
        step=1000,
        help="Only used when fast mode is enabled and your dataset has more rows.",
    )
    tfidf_max_features = st.slider(
        "TF-IDF feature cap",
        min_value=5000,
        max_value=60000,
        value=30000,
        step=5000,
        help="Lower values trade a bit of accuracy for faster vectorizer fitting.",
    )

    st.markdown("---")
    st.subheader("Scoring Weights (Advanced)")
    w_similarity = st.slider("Weight: TF‑IDF Similarity (α)", 0.0, 1.0, 0.70, 0.05)
    w_overlap = st.slider("Weight: Ingredient Overlap (β)", 0.0, 1.0, 0.25, 0.05)
    w_missing_penalty = st.slider("Penalty: Missing Fraction (γ)", 0.0, 1.0, 0.10, 0.05)
    st.caption("Hybrid score = α·similarity + β·overlap − γ·missing_fraction")

    st.markdown("---")
    st.subheader("About")
    st.write("• Built with Streamlit + scikit‑learn")
    st.write("• Kaggle Food.com dataset compatible (RAW_recipes.csv or your recipes.csv)")
    st.write("• This app never sends your data anywhere — local only.")

# -----------------------------
# Load dataset according to choice & build model
# -----------------------------
df_raw = None

if data_source == "Use demo (built-in)":
    df_raw = demo_dataset()

elif data_source == "Upload CSV" and uploaded is not None:
    df_raw = load_dataset_from_csv(uploaded)

elif data_source == "Use local recipes.csv":
   #  if os.path.exists("recipes.csv"):
     #   df_raw = load_dataset_from_csv("recipes.csv")
    elif os.path.exists("RAW_recipes.csv"):
        df_raw = load_dataset_from_csv("RAW_recipes.csv")
    else:
        st.warning("No local recipes.csv or RAW_recipes.csv found. Falling back to demo dataset.")
        df_raw = demo_dataset()

else:
    if data_source == "Upload CSV":
        st.info("← Please upload a CSV to continue, or switch to demo.")
        st.stop()
    else:
        st.warning("No dataset selected. Falling back to demo dataset.")
        df_raw = demo_dataset()

# If for any reason df_raw is still None or empty, use demo
if df_raw is None or len(df_raw) == 0:
    st.warning("Loaded dataset is empty. Falling back to demo dataset.")
    df_raw = demo_dataset()

# Apply fast-mode row cap
df_for_processing = df_raw
row_cap_applied = False
if fast_mode and fast_mode_cap > 0 and len(df_raw) > fast_mode_cap:
    df_for_processing = df_raw.head(int(fast_mode_cap)).copy()
    row_cap_applied = True

# Prepare data (normalize + vectorize)
try:
    with st.spinner("Preparing and vectorizing recipes..."):
        data, vectorizer, X, token_sets = prepare_data(df_for_processing, int(tfidf_max_features))
except Exception as e:
    st.error(f"❌ Error preparing data: {e}")
    st.stop()

if row_cap_applied:
    st.info(
        f"⚡ Fast mode active: using the first {fast_mode_cap:,} recipes out of {len(df_raw):,}. "
        "Turn off fast mode in the sidebar to process the full dataset."
    )


# KPIs
total_recipes = len(data)
with st.container():
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi'><b>Recipes</b><br>{total_recipes:,}</div>", unsafe_allow_html=True)
    has_time = "minutes" in data.columns
    has_steps = "n_steps" in data.columns
    has_n_ings = "n_ingredients" in data.columns
    k2.markdown(
        f"<div class='kpi'><b>Fields</b><br>"
        f"{'⏱️ minutes ' if has_time else ''}"
        f"{'📝 steps ' if has_steps else ''}"
        f"{'🥘 ingredients ' if has_n_ings else ''}</div>",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"<div class='kpi'><b>Vector Size</b><br>{X.shape[1]:,} features</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

# -----------------------------
# Query controls (Main)
# -----------------------------
left, right = st.columns([1.6, 1.0])

with left:
    st.subheader("Enter your ingredients")
    user_input = st.text_input(
        "Comma-separated list (e.g., egg, flour, milk, butter)",
        value="egg, flour, milk",
        help="We will match these against recipe ingredients"
    )
    exclude_input = st.text_input(
        "Exclude ingredients (optional)",
        value="",
        help="Comma-separated items to avoid (e.g., nuts, pork, cilantro)."
    )

with right:
    st.subheader("Filters")
    top_n = st.number_input("Number of results", 1, 50, 10, step=1)
    min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.05, 0.01)
    max_missing = st.slider("Max missing ingredients (per recipe)", 0, 50, 10, 1)

    max_time = None
    if "minutes" in data.columns:
        max_time = st.slider("Max time (minutes)", 0, int(np.nanmax(data["minutes"])) if has_time else 180, 60, 5)

    max_steps = None
    if "n_steps" in data.columns:
        max_steps = st.slider("Max steps", 0, int(np.nanmax(data["n_steps"])) if has_steps else 30, 12, 1)

    max_ings = None
    if "n_ingredients" in data.columns:
        max_ings = st.slider("Max ingredient count", 0, int(np.nanmax(data["n_ingredients"])) if has_n_ings else 40, 15, 1)

    tag_filter = []
    if "tags_list" in data.columns:
        st.caption("Filter by tags (optional)")
        # Collect unique tags (lowercased) safely — cap to first 500 distinct for UI sanity
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
    # dedupe but preserve order
    seen = set()
    uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def compute_scores(user_terms, exclude_terms, weights):
    if vectorizer is None or X is None or data is None:
        st.error("The model is not initialized properly (data/vectorizer/X is None).")
        return pd.DataFrame()
    if not user_terms:
        st.warning("Please enter at least one ingredient.")
        return pd.DataFrame()

    user_vec = vectorizer.transform([" ".join(user_terms)])
    if X.shape[1] == 0:
        st.error("TF-IDF matrix has 0 features. Check your dataset cleaning.")
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

    alpha, beta, gamma = weights
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
    min_similarity: float,
    max_missing_ing: int,
    max_minutes,
    max_steps,
    max_ing_count,
    required_tags: List[str],
) -> pd.DataFrame:
    filt = df[df["similarity"] >= min_similarity]
    filt = filt[~filt["has_excluded"]]
    filt = filt[filt["missing"] <= max_missing_ing]

    if max_minutes is not None and "minutes" in filt.columns:
        filt = filt[pd.to_numeric(filt["minutes"], errors="coerce").fillna(np.inf) <= max_minutes]
    if max_steps is not None and "n_steps" in filt.columns:
        filt = filt[pd.to_numeric(filt["n_steps"], errors="coerce").fillna(np.inf) <= max_steps]
    if max_ing_count is not None and "n_ingredients" in filt.columns:
        filt = filt[pd.to_numeric(filt["n_ingredients"], errors="coerce").fillna(np.inf) <= max_ing_count]

    if required_tags and "tags_list" in filt.columns:
        # Keep rows where any tag in tags_list matches any in required_tags
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
        raw_scores = compute_scores(user_terms, exclude_terms, (w_similarity, w_overlap, w_missing_penalty))
        filtered = apply_filters(
            raw_scores,
            min_similarity=min_sim,
            max_missing_ing=max_missing,
            max_minutes=max_time if has_time else None,
            max_steps=max_steps if has_steps else None,
            max_ing_count=max_ings if has_n_ings else None,
            required_tags=tag_filter,
        )

        # Ranking options
        sort_choice = st.selectbox(
            "Sort results by",
            ["Hybrid Score (best overall)", "Best TF‑IDF Similarity", "Most Overlap", "Fewest Missing", "Quickest Time", "Easiest Steps", "Simplest (# ingredients)"],
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
        elif sort_choice == "Easiest Steps" and has_steps:
            sorted_df = filtered.sort_values(["n_steps", "similarity"], ascending=[True, False])
        elif sort_choice == "Simplest (# ingredients)" and has_n_ings:
            sorted_df = filtered.sort_values(["n_ingredients", "similarity"], ascending=[True, False])
        else:
            sorted_df = filtered.sort_values("score", ascending=False)

        results = sorted_df.head(top_n)

        # Summary
        st.subheader("🍽️ Top Recipe Suggestions")
        st.caption(
            f"Showing <b>{len(results)}</b> of <b>{len(filtered)}</b> matched recipes "
            f"(from {total_recipes:,} total).",
            unsafe_allow_html=True,
        )

        if results.empty:
            st.info("No strong matches found with current filters. Try lowering the similarity threshold or allowing more missing ingredients.")
            st.stop()

        # Build a CSV for download
        def build_export_df(df: pd.DataFrame) -> pd.DataFrame:
            cols = ["name", "similarity", "overlap", "missing", "score"]
            for c in ["minutes", "n_steps", "n_ingredients"]:
                if c in df.columns:
                    cols.append(c)
            # Include a preview of ingredients and tags
            cols += [c for c in ["ingredients", "tags"] if c in df.columns]
            export_df = df[cols].copy()
            export_df["similarity"] = export_df["similarity"].round(3)
            export_df["score"] = export_df["score"].round(3)
            return export_df

        export_csv = build_export_df(results).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results as CSV",
            data=export_csv,
            file_name="dish_recommendations.csv",
            mime="text/csv",
        )

        # Render result cards
        def render_card(row):
            parts = []
            parts.append("<div class='recipe-card'>")
            # Title
            title = str(row.get("name", "Untitled Recipe"))
            parts.append(f"<div class='recipe-title'>{title}</div>")

            # Meta line
            meta_bits = []
            if "minutes" in row and pd.notna(row["minutes"]):
                meta_bits.append(f"⏱️ {int(row['minutes'])} min")
            if "n_steps" in row and pd.notna(row["n_steps"]):
                meta_bits.append(f"📝 {int(row['n_steps'])} steps")
            if "n_ingredients" in row and pd.notna(row["n_ingredients"]):
                meta_bits.append(f"🥘 {int(row['n_ingredients'])} ingredients")
            meta_bits.append(f"🔎 sim {row['similarity']:.2f}")
            meta_bits.append(f"✨ score {row['score']:.2f}")
            parts.append(f"<div class='meta'>{' | '.join(meta_bits)}</div>")

            # Chips: overlap terms (intersection with user)
            if "ingredients_norm" in row and isinstance(row["ingredients_norm"], str):
                recipe_terms = set(row["ingredients_norm"].split())
                ov_terms = [t for t in user_terms if t in recipe_terms]
                if ov_terms:
                    chips = "".join([f"<span class='chip'>{t}</span>" for t in ov_terms[:12]])
                    parts.append(f"<div class='chips'>{chips}</div>")

            parts.append("</div>")
            st.markdown("".join(parts), unsafe_allow_html=True)

            # Expanders
            with st.expander("Missing ingredients & score details", expanded=False):
                recipe_terms = set(row["ingredients_norm"].split()) if isinstance(row["ingredients_norm"], str) else set()
                u_set = set(user_terms)
                missing_terms = sorted(list(u_set - (u_set & recipe_terms)))
                if missing_terms:
                    st.write("**Missing from your list:** " + ", ".join(missing_terms))
                else:
                    st.write("**Missing from your list:** (none)")

                st.write(
                    f"**Score breakdown**  • similarity = {row['similarity']:.3f}  • overlap = {row['overlap']}/{len(user_terms)}"
                    f"  • missing = {row['missing']} ({row['missing_frac']:.2f} frac)"
                )
                st.caption("Hybrid score = α·similarity + β·overlap_frac − γ·missing_frac")

            # Details: description / steps / raw ingredients
            has_any_detail = False
            with st.expander("Recipe details"):
                if "description" in row and isinstance(row["description"], str) and row["description"].strip():
           
                    st.write(row["description"])
                    has_any_detail = True

                if "ingredients" in row and isinstance(row["ingredients"], (list, str)):
                    # Show ingredients list in raw form (list if possible)
                    ings = _safe_literal_eval(row["ingredients"]) if isinstance(row["ingredients"], str) else row["ingredients"]
                    if isinstance(ings, list):
                        st.markdown("**Ingredients:**")
                        st.write(", ".join(sorted(set([str(i) for i in ings if str(i).strip()]))))
                        has_any_detail = True

                if "steps_list" in row and isinstance(row["steps_list"], list) and len(row["steps_list"]) > 0:
                    st.markdown("**Steps:**")
                    for idx, s in enumerate(row["steps_list"], 1):
                        st.write(f"{idx}. {s}")
                    has_any_detail = True

                if not has_any_detail:
                    st.caption("No additional details available in this dataset row.")

        # Grid display: two columns for dense layout
        grid_cols = st.columns(2)
        for i, (_, r) in enumerate(results.iterrows()):
            with grid_cols[i % 2]:
                render_card(r)

        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)
         #Adjust weights and filters in the sidebar to reflect your preferences (speed vs. match quality, etc.).")
else:
    st.info("Enter ingredients, adjust filters, and click **Find Dishes** to see suggestions.")
