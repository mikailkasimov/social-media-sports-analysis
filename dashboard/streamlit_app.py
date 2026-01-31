import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from utils import *

DATA_FILES = {
    "reddit_posts": "../data/reddit_posts_labelled_filtered_2025-10-31_to_2025-11-14.parquet",
    "reddit_comments": "../data/reddit_comments_labelled_filtered_2025-10-31_to_2025-11-14.parquet",
    "chan_posts": "../data/chan_posts_labelled_filtered_2025-10-31_to_2025-11-14.parquet",
}

PERSPECTIVE_FILES = {
    "reddit_posts": "../data/reddit_posts_perspective.parquet",
    "reddit_comments": "../data/reddit_comments_perspective.parquet",
    "chan_posts": "../data/chan_posts_perspective.parquet",
}

FREQ_MAP = {
    "15min": "15min", "30min": "30min", "1h": "1H", "3h": "3H", "6h": "6H",
    "12h": "12H", "1d": "1D", "2d": "2D", "3d": "3D",
}
 
SPORTS = [ 
    "american football","basketball","baseball","soccer","ice hockey", "mma","boxing","track and field",
    "volleyball","beach volleyball", "swimming","golf","table tennis","car racing","snowboarding", "lacrosse",
    "skiing","fishing","tennis","softball", "wrestling","motocross","kickball","gymnastics","rugby", "x games",
    "chess","bowling","hunting","climbing", "skateboarding","pool","surfing","horse racing","rodeo", "ultimate frisbee",
    "ice skating","badminton","cycling","darts", "weightlifting","karate","cheerleading","motorcycle racing","target shooting", 
    "handball","fencing","pickleball","kayaking","disc golf", "dance","water polo","figure skating","taekwondo","skeet shooting", 
    "roller skating","running","racquetball","bodybuilding","sumo", 
]

PERSPECTIVE_COLS = [
    "TOXICITY","SEVERE_TOXICITY","INSULT","PROFANITY","THREAT",
    "IDENTITY_ATTACK","SEXUALLY_EXPLICIT"
]

st.set_page_config(page_title="4chan and Reddit Sports Subcommunity Analysis",layout="wide")
st.title("Reddit and 4chan Sports Community Analysis")
@st.cache_data(show_spinner=False)
def load_data():
    out = {}
    for ds in DATA_FILES.keys():
        df1 = pd.read_parquet(DATA_FILES[ds])
        df2 = pd.read_parquet(PERSPECTIVE_FILES[ds])
        out[ds] = pd.concat([df1.reset_index(drop=True), df2[PERSPECTIVE_COLS].reset_index(drop=True)],axis=1)
    return out
ALL_DATA = load_data()
dashboard = st.sidebar.selectbox("Dashboard", ["Activity", "Toxicity", "Sports", "All"], index=0)
# -------------------- ACTIVITY TAB
if dashboard == "Activity":
    ds = st.sidebar.selectbox("Dataset", list(DATA_FILES.keys()))
    freq = st.sidebar.selectbox("Bin Size", list(FREQ_MAP.keys()), index=2)
    bin_freq = FREQ_MAP[freq]
    with st.container():
        st.markdown("## Activity Overview")
        df = ALL_DATA[ds].copy()
        group_col = "flair" if ds.startswith("reddit") else "label"
        df["created_utc"] = (pd.to_datetime(df["created_utc"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None))
        df = df.rename(columns={"created_utc": "created_est"})        
        df["bin"] = df["created_est"].dt.floor(bin_freq)
        #create df groupbed by flair / label
        agg = (df.groupby([group_col, "bin"]).size().rename("count").reset_index())
        full = pd.date_range(agg["bin"].min(), agg["bin"].max(), freq=bin_freq)
        pivot = (agg.pivot(index=group_col, columns="bin", values="count").reindex(columns=full).fillna(0))
        #plot lineplot
        time_series = (agg.groupby("bin")["count"].sum().reindex(full, fill_value=0).rename_axis("bin").reset_index())
        fig = px.line(
            time_series,
            x="bin",
            y="count",
            labels={"bin": "Time", "count": "Count"},
            title="Total Activity Over Time",
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
        fig = px.imshow(
            pivot,
            x=pivot.columns,
            y=pivot.index,
            labels={
                "x": "Time", 
                "y": group_col.capitalize(), 
                "color": "Count"
            },
            aspect="auto",
            color_continuous_scale="viridis",
            title=f"Activity Heatmap by {group_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
        c1, c2, c3, c4, c5, c6 = st.columns([2, 7, 0.5, 7, 0.1, 2], vertical_alignment="center")
        c1.markdown("**What happened on**")
        with c2:
            date_choice = st.date_input("date", value=date.today(), label_visibility="collapsed")
        c3.markdown("**in**")
        with c4:
            sport_choice = st.selectbox("sport", SPORTS, label_visibility="collapsed")
        c5.markdown("**?**")
        asked = c6.button("Search", type="primary")
        prompt = f"What happened on {date_choice.isoformat()} in {sport_choice}?"
        if asked:
            st.session_state["answer"] = generate_response(prompt)
        if st.session_state.get("answer"):
            st.markdown("### Result")
            st.write(st.session_state["answer"])

# -------------------- TOXICITY TAB
elif dashboard == "Toxicity":
    with st.container():
        st.markdown("## Toxicity Overview")
        ds = st.sidebar.selectbox("Dataset", list(ALL_DATA.keys()))
        df = ALL_DATA[ds].copy()
        df[PERSPECTIVE_COLS] = df[PERSPECTIVE_COLS].apply(pd.to_numeric,errors='coerce')
        vals = df[PERSPECTIVE_COLS].mean()
        chart_df = vals.reset_index()
        chart_df.columns = ["attribute", "mean_score"]
        fig = px.bar(
            chart_df, 
            x="attribute", 
            y="mean_score", 
            title="Mean Google Perspective scores",
            color_discrete_sequence=["salmon"],
        )
        fig.update_yaxes(range=[0, 0.4])
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
        #correlation heatmap
        corr = df[PERSPECTIVE_COLS].corr(method="pearson")
        fig = px.imshow(
            corr,
            x=corr.columns,
            y=corr.index,
            zmin=-1, zmax=1,
            text_auto=True,
            aspect="auto",
            title="Google Perspective Attribute Correlations",
            color_continuous_scale="rdbu_r",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
        #toxicit vs sport correlation
        group_col = "flair" if ds.startswith("reddit") else "label"
        df[group_col] = df[group_col]
        df_ = df.dropna(subset=[group_col]).copy()
        df__ = df_.groupby(group_col)[PERSPECTIVE_COLS].mean()
        fig = px.imshow(
            df__,
            aspect="auto",
            title=f"Mean Perspective attributes by {group_col}",
            color_continuous_scale="viridis",  # change if you want
        )
        fig.update_coloraxes(cmin=0, cmax=0.4)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- SPORTS TAB
elif dashboard == "Sports":
    with st.container():
        st.markdown("## Sports Overview")
        ds = st.sidebar.selectbox("Dataset", list(DATA_FILES.keys()))
        df = ALL_DATA[ds].copy()
        group_col = "flair" if ds.startswith("reddit") else "label"
        labels = df[group_col].value_counts().head(250).reset_index()
        fig = px.bar(labels, 
            x=group_col, 
            y="count", 
            title=f"Top {group_col} by Count"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------- ALL TAB
else:
    with st.container():
        CHAN_TO_REDDIT = {
            "ice hockey": "hockey",
            "american football": "football",
            "track and field": "track & field",
            "mma": "fighting",
            "boxing": "fighting",
            "car racing": "motorsports",
            "motorcycle racing": "motorsports",
            "kayaking": "canoe/kayaking",
            "target shooting": "shooting",
            "ice skating": "skating",
            "rugby": "rugby union"
        }
        # we do this in order to get matching values for "label" and for "flair"
        for name in ("reddit_posts", "reddit_comments"):
            if name in ALL_DATA and "flair" in ALL_DATA[name].columns:
                ALL_DATA[name]["flair"] = (ALL_DATA[name]["flair"].replace({"track &amp; field": "track & field"}))
        ALL_DATA["chan_posts"]["label"] = (ALL_DATA["chan_posts"]["label"].replace(CHAN_TO_REDDIT))
        all_toxicity_dist = pd.concat([d[PERSPECTIVE_COLS].apply(pd.to_numeric, errors='coerce')
            .assign(dataset=name).melt("dataset", var_name="attribute", value_name="score") 
            for name, d in ALL_DATA.items()],ignore_index=True).dropna(subset=["score"])
        #plot toxicity cdf here
        cdf_chart = px.ecdf(
            all_toxicity_dist,
            x="score",
            color="dataset",
            facet_col="attribute",
            facet_col_wrap=3,
            title="Toxicity CDF across datasets",
        )
        st.plotly_chart(cdf_chart, use_container_width=True)
        st.divider()
        #plot sports cdf here
        cols = {}
        for name, d in ALL_DATA.items():
            group_col = "flair" if name.startswith("reddit") else "label"
            cols[name] = d[group_col].value_counts(normalize=True).mul(100)
        w = pd.DataFrame(cols).fillna(0)
        t = w.sum(1).nlargest(25).index
        dist = (w.loc[t].reset_index().melt("index", var_name="dataset", value_name="pct").rename(columns={"index": "sport"}))
        fig = px.bar(
            dist,
            x="sport",
            y="pct",
            color="dataset",
            barmode="group",
            title="Sports distribution across datasets",
        )
        st.plotly_chart(fig, use_container_width=True) 