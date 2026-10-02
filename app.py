"""
Streamlit app (Phase 4): where is headphone price out of line with perceived value?

Run from the project folder, environment active:
    streamlit run app.py

Tabs: Analytics (band statistics, themes, overpriced-for-band products, suspect listings)
and Ask the reviews (cited Q&A, next step). Numbers come from analytics.py; see CLAUDE.md
for the rules behind them (per-band statistics, excluded listings).
"""

from __future__ import annotations

import html
import re

import altair as alt
import pandas as pd
import streamlit as st

import analytics as an

st.set_page_config(page_title="Headphone Pricing Intelligence", layout="wide")

# ---------------------------------------------------------------- palette
# Reference palette from the dataviz skill (validated light and dark steps).
# Overpriced = red pole, good value = blue pole, neutral = gray midpoint.
# surface = Streamlit's page background, used for the 2px gaps between adjacent fills.
PALETTE = {
    "light": {"surface": "#ffffff", "neg": "#e34948", "pos": "#2a78d6", "mid": "#f0efec", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#0e1117", "neg": "#e66767", "pos": "#3987e5", "mid": "#383835", "ink": "#ffffff", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
theme_type = getattr(getattr(st.context, "theme", None), "type", None)
C = PALETTE["dark" if theme_type == "dark" else "light"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def styled(chart: alt.Chart) -> alt.Chart:
    return (chart.configure(background="transparent", font=FONT)
            .configure_view(stroke=None)
            .configure_axis(labelColor=C["ink2"], titleColor=C["ink2"], gridColor=C["grid"],
                            domainColor=C["axis"], tickColor=C["axis"], labelFontSize=12, titleFontSize=12,
                            titleFontWeight="normal")
            .configure_legend(labelColor=C["ink2"], titleColor=C["ink2"], labelFontSize=12, titleFontSize=12,
                              orient="top", title=None))


# ---------------------------------------------------------------- data

@st.cache_data
def load():
    d = an.load()
    pv = an.product_value(d["reviews"], d["products"])
    return {**d, "summary": an.band_summary(d["reviews"]), "pv": pv,
            "themes_by_band": an.theme_by_band(d["reviews"], d["themes"]),
            "suspect": an.suspect_listings(d["products"])}


d = load()
summary = d["summary"]
LABELS = list(summary["label"])  # band labels in price order


def pct(x: float) -> str:
    return f"{x:.0%}"


def review_md(text: str) -> str:
    """Review text safe for st.markdown: no HTML, and $ doesn't trigger math rendering."""
    text = re.sub(r"<br\s*/?>", "\n", str(text))
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"([\\`*_{}\[\]<>#+\-.!|$~])", r"\\\1", text).replace("\n", "  \n")


# ---------------------------------------------------------------- header

st.title("Headphone Pricing Intelligence")
st.caption("Where is price out of line with perceived value? Amazon headphone reviews "
           "(Amazon Reviews 2023, McAuley Lab, UCSD), tagged for value for money by Claude.")

tab_analytics, tab_qa = st.tabs(["Analytics", "Ask the reviews"])

with tab_analytics:
    flag, budget = summary.iloc[-1], summary.iloc[0]
    st.markdown(
        f"#### Flagship buyers who comment on value call it overpriced **{pct(flag.negative)}** of the time, "
        f"against **{pct(budget.negative)}** for budget buyers.")
    st.caption(f"Perceived value holds steady from budget through mid-range, then drops at \\$100 and again at "
               f"\\$200. Shares are among reviews that give a value verdict; "
               f"{summary['verdicts'].sum():,} of {summary['reviews'].sum():,} reviews do.")

    with st.expander("How to read these numbers"):
        n_ex = len(d["suspect"])
        st.markdown(
            f"- **Value verdict** is value for money only, not overall satisfaction: *positive* = worth the price, "
            f"*overpriced* = not worth it, *neutral* = mentions price without a clear verdict. Reviews that never "
            f"mention price have no verdict.\n"
            f"- **Statistics are per band.** Premium and flagship were deliberately oversampled so each band has "
            f"enough reviews; pooling bands together would misrepresent the category.\n"
            f"- **{n_ex} listings are excluded** (reseller markups, non-headphones, multi-packs); see "
            f"*Suspect listings* below.\n"
            f"- Ranges in brackets are 95% confidence intervals.")

    # ------------------------------------------------ verdict by band
    col1, col2 = st.columns(2, gap="large")
    s = summary.assign(
        ci=[f"{lo:.0%} to {hi:.0%}" for lo, hi in zip(summary.negative_lo, summary.negative_hi)],
        neg_pct=summary.negative.map(pct))
    tooltip = [alt.Tooltip("label:N", title="Band"),
               alt.Tooltip("negative:Q", title="Overpriced", format=".1%"),
               alt.Tooltip("ci:N", title="95% interval"),
               alt.Tooltip("verdicts:Q", title="Value verdicts", format=","),
               alt.Tooltip("reviews:Q", title="Reviews", format=","),
               alt.Tooltip("products:Q", title="Products")]
    with col1:
        st.markdown("**Share calling it overpriced, by price band**")
        x = alt.X("label:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140))
        base = alt.Chart(s).encode(x=x, tooltip=tooltip)
        bars = base.mark_bar(color=C["neg"], cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=44).encode(
            y=alt.Y("negative:Q", title="Overpriced share", axis=alt.Axis(format="%", tickCount=5),
                    scale=alt.Scale(domain=[0, 0.5])))
        ci = base.mark_rule(color=C["ink2"], strokeWidth=1.5).encode(y="negative_lo:Q", y2="negative_hi:Q")
        labels = base.mark_text(dy=-8, color=C["ink"], fontSize=12, fontWeight="bold").encode(
            y="negative_hi:Q", text="neg_pct:N")
        st.altair_chart(styled((bars + ci + labels).properties(height=300)), width="stretch")
        st.caption("Whiskers: 95% interval. Hover a bar for sample sizes.")

    with col2:
        st.markdown("**Value verdict mix, by price band**")
        rows = []
        for r in summary.itertuples():
            half = r.neutral / 2
            for name, x0, x1, share in [("Overpriced", -half - r.negative, -half, r.negative),
                                        ("Neutral", -half, half, r.neutral),
                                        ("Good value", half, half + r.positive, r.positive)]:
                rows.append({"label": r.label, "verdict": name, "x0": x0, "x1": x1, "share": share,
                             "verdicts": r.verdicts})
        mix = pd.DataFrame(rows)
        chart = alt.Chart(mix).mark_bar(size=26, stroke=C["surface"],
                                        strokeWidth=2).encode(
            y=alt.Y("label:N", sort=LABELS, title=None, axis=alt.Axis(labelLimit=160)),
            x=alt.X("x0:Q", title="← overpriced · good value →",
                    axis=alt.Axis(format="%", labelExpr="format(abs(datum.value), '.0%')"),
                    scale=alt.Scale(domain=[-0.5, 0.9])),
            x2="x1:Q",
            color=alt.Color("verdict:N", sort=["Overpriced", "Neutral", "Good value"],
                            scale=alt.Scale(domain=["Overpriced", "Neutral", "Good value"],
                                            range=[C["neg"], C["axis"], C["pos"]])),
            tooltip=[alt.Tooltip("label:N", title="Band"), alt.Tooltip("verdict:N", title="Verdict"),
                     alt.Tooltip("share:Q", title="Share", format=".1%"),
                     alt.Tooltip("verdicts:Q", title="Value verdicts", format=",")])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=C["axis"]).encode(x="x:Q")
        st.altair_chart(styled((chart + zero).properties(height=300)), width="stretch")
        st.caption("Centered on neutral. Positive outweighs negative in every band, but the margin shrinks above \\$100.")

    table = pd.DataFrame({
        "Band": summary["label"],
        "Products": summary["products"],
        "Reviews": summary["reviews"],
        "Mention price": summary["mention_price"].map(pct),
        "Value verdicts": summary["verdicts"],
        "Good value": summary["positive"].map(pct),
        "Neutral": summary["neutral"].map(pct),
        "Overpriced": summary["negative"].map(pct),
        "Overpriced, 95% interval": [f"{lo:.0%} to {hi:.0%}" for lo, hi in zip(summary.negative_lo, summary.negative_hi)],
    })
    st.dataframe(table, hide_index=True, width="stretch")

    # ------------------------------------------------ themes
    st.divider()
    st.subheader("What buyers praise and complain about as price rises")
    st.caption("Color: net sentiment on each topic (positive minus negative mentions, as a share of all mentions). "
               "Number: share of the band's reviews that mention it. Hover for counts.")
    tb = d["themes_by_band"].copy()
    tb["label"] = tb["band"].map(dict(zip(summary["band"], summary["label"])))
    tb["theme_name"] = tb["theme"].str.replace("_", " ").str.replace(" app", " / app").str.capitalize()
    order = (tb.groupby("theme_name")["mentions"].sum().sort_values(ascending=False).index.tolist())
    tb["prev_txt"] = tb["prevalence"].map(lambda v: f"{v:.0%}" if v >= 0.005 else "<1%")
    heat = alt.Chart(tb).encode(
        x=alt.X("label:N", sort=LABELS, title=None, axis=alt.Axis(labelAngle=0, orient="top", labelLimit=140)),
        y=alt.Y("theme_name:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        tooltip=[alt.Tooltip("theme_name:N", title="Topic"), alt.Tooltip("label:N", title="Band"),
                 alt.Tooltip("prevalence:Q", title="Share of reviews", format=".1%"),
                 alt.Tooltip("net:Q", title="Net sentiment", format="+.2f"),
                 alt.Tooltip("mentions:Q", title="Mentions", format=","),
                 alt.Tooltip("positive:Q", title="Positive"), alt.Tooltip("negative:Q", title="Negative"),
                 alt.Tooltip("mixed:Q", title="Mixed")])
    cells = heat.mark_rect(stroke=C["surface"], strokeWidth=2,
                           cornerRadius=3).encode(
        color=alt.Color("net:Q", title="Net sentiment",
                        scale=alt.Scale(domain=[-1, 0, 1], range=[C["neg"], C["mid"], C["pos"]], interpolate="rgb"),
                        legend=alt.Legend(format="+.1f", orient="right", gradientLength=180)))
    text = heat.mark_text(fontSize=12, color=C["ink"]).encode(text="prev_txt:N")
    st.altair_chart(styled((cells + text).properties(height=460)), width="stretch")
    st.caption("Sound quality and value-for-money sentiment move in opposite directions as price rises; "
               "connectivity and customer service turn negative at flagship prices. "
               "Small cells (few mentions) are noisy: check the counts before reading much into them.")

    # ------------------------------------------------ overpriced for band
    st.divider()
    st.subheader("Overpriced for their band")
    st.caption("Products priced above their band's median whose overpriced share beats the band average. "
               "**Clearly above** means the 95% interval's low end still beats the band average, so it isn't "
               "just a few unhappy reviews. Ranked by that low end. Select a row to read its reviews.")
    min_v = st.slider("Minimum value verdicts per product", 5, 20, 8,
                      help="Fewer verdicts means more products but noisier shares.")
    op = an.overpriced_for_band(d["pv"], min_verdicts=min_v)
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**{len(op)} products**, {int(op['clearly_above'].sum())} clearly above their band average")
    only_clear = c2.toggle("Clearly above only", value=False)
    if only_clear:
        op = op[op["clearly_above"]]
    band_lbl = dict(zip(summary["band"], summary["label"]))
    view = pd.DataFrame({
        "Product": op["title"].str.slice(0, 90),
        "Band": op["band"].map(band_lbl),
        "Price": op["price"],
        "Band median": op["band_median_price"],
        "Overpriced": op["share"] * 100,
        "95% interval": [f"{lo:.0%} to {hi:.0%}" for lo, hi in zip(op.share_lo, op.share_hi)],
        "Band average": op["band_rate"] * 100,
        "Value verdicts": op["verdicts"],
        "Reviews": op["reviews"],
        "Clearly above": op["clearly_above"],
    })
    event = st.dataframe(
        view, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key="overpriced_table",
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Band median": st.column_config.NumberColumn(format="$%.2f"),
            "Product": st.column_config.TextColumn(width="large"),
            "Overpriced": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "Band average": st.column_config.NumberColumn(format="%.0f%%"),
            "Clearly above": st.column_config.CheckboxColumn(),
        })
    picked = event.selection.rows if event and event.selection else []
    if picked:
        p = op.iloc[picked[0]]
        st.markdown(f"##### {review_md(p.title)}")
        st.caption(f"\\${p.price:,.2f} · {band_lbl[p.band]} · {p.store} · {p.reviews} reviews sampled, "
                   f"{p.verdicts} with a value verdict ({p.negative} overpriced)")
        verdict_name = {"negative": "Overpriced", "neutral": "Neutral", "positive": "Good value"}
        for r in an.product_reviews(d["reviews"], p.parent_asin).itertuples():
            with st.container(border=True):
                st.markdown(f"**{r.review_id}** · {'★' * int(r.rating)}{'☆' * (5 - int(r.rating))} · "
                            f"{verdict_name[r.value_sentiment]}" + (f" · {review_md(r.title)}" if r.title else ""))
                st.markdown(review_md(r.text))

    # ------------------------------------------------ suspect listings
    st.divider()
    st.subheader("Suspect listings")
    sus = d["suspect"]
    counts = sus["why"].value_counts()
    st.caption(f"{len(sus)} listings left out of every statistic above: "
               + ", ".join(f"{n} {w.lower()}" for w, n in counts.items())
               + ". The dataset's price is a single snapshot, and some listings were third-party resellers "
                 "charging several times the normal price. Claude estimated each product's normal price; "
                 "listings more than twice that were flagged, and borderline cases were checked by hand.")
    st.dataframe(pd.DataFrame({
        "Why excluded": sus["why"],
        "Product": sus["title"].str.slice(0, 90),
        "Listed price": sus["price"],
        "Estimated normal price": [f"${lo:,.0f}-{hi:,.0f}" if pd.notna(lo) and pd.notna(hi) else "unknown"
                                   for lo, hi in zip(sus.typical_low, sus.typical_high)],
        "Reason": sus["override_note"].where(sus["override_note"].notna(), sus["reason"]),
    }), hide_index=True, width="stretch",
        column_config={"Listed price": st.column_config.NumberColumn(format="$%.2f"),
                       "Product": st.column_config.TextColumn(width="large")})

with tab_qa:
    st.info("Coming next: ask questions about the reviews and get answers with clickable citations.")
