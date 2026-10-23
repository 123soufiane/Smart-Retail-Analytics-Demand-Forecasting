"""Streamlit dashboard backed by the forecast API."""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
COLORS = {
    "navy": "#10233F",
    "blue": "#356AE6",
    "teal": "#16A6A1",
    "muted": "#66758A",
    "grid": "#E8EDF4",
}

st.set_page_config(
    page_title="Retail Forecasts | Smart Retail",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
      [data-testid="stMetric"] {
        background: #fff; border: 1px solid #E8EDF4; padding: 1rem 1.15rem;
        border-radius: 12px; box-shadow: 0 2px 8px rgba(16,35,63,.04);
      }
      [data-testid="stMetricLabel"] p { color: #66758A; font-size: .85rem; }
      [data-testid="stMetricValue"] { color: #10233F; }
      [data-testid="stSidebar"] { border-right: 1px solid #E8EDF4; }
      div[data-testid="stDataFrame"] { border: 1px solid #E8EDF4; border-radius: 10px; }
      .eyebrow { color: #356AE6; text-transform: uppercase; letter-spacing: .12em;
        font-size: .75rem; font-weight: 700; margin-bottom: .35rem; }
      .subtle { color: #66758A; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def get_api_data(path: str, params: tuple[tuple[str, str], ...] = ()) -> dict:
    response = requests.get(
        f"{API_BASE_URL}{path}", params=dict(params), timeout=15
    )
    response.raise_for_status()
    return response.json()


def request_data(path: str, params: dict | None = None) -> dict:
    try:
        return get_api_data(path, tuple(sorted((params or {}).items())))
    except requests.RequestException as error:
        st.error(
            f"Could not load data from the forecast API at {API_BASE_URL}. "
            f"Start the API and try again. Details: {error}"
        )
        st.stop()
    return {}


def format_quantity(value: float) -> str:
    return f"{value:,.0f}"


def configure_chart(fig, height: int = 390):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=16, t=36, b=12),
        font=dict(family="sans-serif", color=COLORS["navy"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


with st.sidebar:
    st.markdown("## 📈 Smart Retail")
    st.caption("Demand intelligence")
    st.divider()
    level = st.radio(
        "Forecast view",
        options=("store", "product"),
        format_func=lambda value: (
            "Store × product" if value == "store" else "Product"
        ),
    )
    st.markdown("---")
    st.caption("Data refreshes automatically every minute.")
    if st.button("↻ Refresh data", width="stretch"):
        get_api_data.clear()
        st.rerun()

summary = request_data("/api/summary", {"level": level})
result = request_data(
    "/api/forecasts", {"level": level, "limit": 10000}
)
all_forecasts = pd.DataFrame(result["forecasts"])

if all_forecasts.empty:
    st.title("Demand forecast")
    st.info("No forecasts are available yet. Run the prediction pipeline first.")
    st.stop()

st.markdown('<div class="eyebrow">Forecast intelligence</div>', unsafe_allow_html=True)
st.title("Demand forecast")
st.markdown(
    f'<p class="subtle">Forecast for <b>{summary["predict_week"]}</b> '
    f'· Last observed week: {summary["last_week"]}</p>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("#### Filter forecasts")
    filter_columns = st.columns([1, 1, 1.5])
    filtered = all_forecasts.copy()
    if level == "store":
        store_options = sorted(filtered["store_name"].dropna().unique().tolist())
        selected_stores = filter_columns[0].multiselect(
            "Stores", store_options, placeholder="All stores"
        )
        if selected_stores:
            filtered = filtered[filtered["store_name"].isin(selected_stores)]

    product_options = sorted(filtered["product_name"].dropna().unique().tolist())
    selected_products = filter_columns[1 if level == "store" else 0].multiselect(
        "Products", product_options, placeholder="All products"
    )
    if selected_products:
        filtered = filtered[filtered["product_name"].isin(selected_products)]

    search = filter_columns[2].text_input(
        "Search", placeholder="Search product, SKU, or store"
    ).strip()
    if search:
        search_fields = ["product_name", "sku"]
        if level == "store":
            search_fields.append("store_name")
        matches = pd.Series(False, index=filtered.index)
        for field in search_fields:
            matches |= filtered[field].astype(str).str.contains(
                search, case=False, regex=False, na=False
            )
        filtered = filtered[matches]

if filtered.empty:
    st.warning("No forecasts match the selected filters.")
    st.stop()

last_quantity = float(filtered["last_quantity"].sum())
predicted_quantity = float(filtered["predicted_quantity"].sum())
change = (
    (predicted_quantity - last_quantity) / last_quantity * 100
    if last_quantity
    else None
)
store_count = (
    int(filtered["store_id"].nunique()) if level == "store" else None
)
product_count = int(filtered["product_id"].nunique())
forecast_count = len(filtered)

st.markdown("")
kpis = st.columns(4)
kpis[0].metric(
    "Forecast demand",
    format_quantity(predicted_quantity),
    delta=None if change is None else f"{change:+.1f}% vs last week",
)
kpis[1].metric("Last week demand", format_quantity(last_quantity))
kpis[2].metric("Products", f"{product_count:,}")
kpis[3].metric(
    "Stores" if level == "store" else "Forecast rows",
    f"{store_count:,}" if store_count is not None else f"{forecast_count:,}",
)

top_n = st.slider(
    "Items shown in charts",
    min_value=min(5, forecast_count),
    max_value=min(30, forecast_count),
    value=min(12, forecast_count),
)
ranked = filtered.nlargest(top_n, "predicted_quantity").copy()
ranked["label"] = (
    ranked["store_name"].astype(str) + " · "
    if level == "store"
    else ""
) + ranked["product_name"].astype(str)

left_chart, right_chart = st.columns([1.15, 1])
with left_chart:
    st.markdown("#### Highest forecast demand")
    bar = px.bar(
        ranked.sort_values("predicted_quantity"),
        x="predicted_quantity",
        y="label",
        orientation="h",
        labels={"predicted_quantity": "Forecast quantity", "label": ""},
        color_discrete_sequence=[COLORS["blue"]],
        hover_data={
            "predicted_quantity": ":,.2f",
            "last_quantity": ":,.2f",
            "label": True,
        },
    )
    bar.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Forecast: %{x:,.2f}<extra></extra>",
    )
    st.plotly_chart(
        configure_chart(bar, max(360, 28 * len(ranked))),
        width="stretch",
    )

with right_chart:
    st.markdown("#### Forecast vs. last week")
    comparison = px.scatter(
        filtered,
        x="last_quantity",
        y="predicted_quantity",
        color="store_name" if level == "store" else "product_name",
        hover_name="product_name",
        hover_data={
            "sku": True,
            "store_name": level == "store",
            "last_quantity": ":,.2f",
            "predicted_quantity": ":,.2f",
        },
        labels={
            "last_quantity": "Last week",
            "predicted_quantity": "Forecast week",
            "store_name": "Store",
            "product_name": "Product",
        },
        color_discrete_sequence=px.colors.qualitative.Safe,
        opacity=0.78,
    )
    max_quantity = max(
        float(filtered["last_quantity"].max()),
        float(filtered["predicted_quantity"].max()),
    )
    comparison.add_trace(
        go.Scatter(
            x=[0, max_quantity],
            y=[0, max_quantity],
            mode="lines",
            name="No change",
            line=dict(color=COLORS["muted"], dash="dash"),
            hoverinfo="skip",
        )
    )
    st.plotly_chart(configure_chart(comparison), width="stretch")

st.markdown("#### Forecast details")
table_columns = [
    column
    for column in (
        "store_name",
        "product_name",
        "sku",
        "last_week",
        "predict_week",
        "last_quantity",
        "predicted_quantity",
    )
    if column in filtered.columns
]
table = filtered[table_columns].sort_values(
    "predicted_quantity", ascending=False
)
for date_column in ("last_week", "predict_week"):
    if date_column in table.columns:
        table[date_column] = pd.to_datetime(table[date_column], errors="coerce")
st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    height=420,
    column_config={
        "store_name": "Store",
        "product_name": "Product",
        "sku": "SKU",
        "last_week": st.column_config.DateColumn("Last week"),
        "predict_week": st.column_config.DateColumn("Forecast week"),
        "last_quantity": st.column_config.NumberColumn(
            "Last week demand", format="%,.2f"
        ),
        "predicted_quantity": st.column_config.NumberColumn(
            "Forecast demand", format="%,.2f"
        ),
    },
)
csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered forecasts",
    data=csv,
    file_name=f"{level}_demand_forecasts.csv",
    mime="text/csv",
    width="stretch",
)
