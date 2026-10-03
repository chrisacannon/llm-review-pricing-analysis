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
import os
import re
import threading
import time

import altair as alt
import pandas as pd
import streamlit as st

import datasource

st.set_page_config(page_title="Headphone Pricing Intelligence", layout="wide")


def secret(name: str) -> str | None:
    """Streamlit secrets when deployed, the environment locally."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file
        pass
    return os.environ.get(name)


@st.cache_resource(show_spinner=False)
def fetch_data(repo: str):
    """Once per server: download the private data repo (see datasource.py)."""
    return datasource.download(repo, secret("HF_TOKEN"))


if not datasource.has_data():
    if not secret("HF_DATA_REPO"):
        st.error("No review data found. Run the pipeline locally, or set HF_DATA_REPO and HF_TOKEN "
                 "in Streamlit secrets to download it.")
        st.stop()
    with st.spinner("Downloading the review data (first start only)..."):
        fetch_data(secret("HF_DATA_REPO"))

import analytics as an  # noqa: E402  (after the data is in place: it reads from datasource.data_dir())

# Guardrails for the public demo: limits apply to questions on the app's own API key;
# a visitor's own key has no limit. Example questions are served from a cache and are free.
SESSION_LIMIT = 5          # per browser session (Chris, 2026-10-02)
DAILY_LIMIT = 100          # across all visitors, ~$2/day at ~$0.02 an answer; backstop for page reloads
MAX_QUESTION_CHARS = 500

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
    import qa

    VALUE_NAMES = {"positive": "Good value", "negative": "Overpriced", "neutral": "Neutral",
                   "not_mentioned": "No price talk"}
    THEME_LIST = sorted(t for t in d["themes"]["theme"].unique() if t != "other")

    def app_key() -> str | None:
        return secret("ANTHROPIC_API_KEY")

    @st.cache_resource
    def daily_usage() -> dict:
        """Questions answered on the app's key, shared by every visitor (resets when the app restarts)."""
        return {"date": None, "count": 0, "lock": threading.Lock()}

    def daily_count() -> int:
        u = daily_usage()
        return u["count"] if u["date"] == time.strftime("%Y-%m-%d") else 0

    def record_daily_use() -> None:
        u = daily_usage()
        with u["lock"]:
            today = time.strftime("%Y-%m-%d")
            if u["date"] != today:
                u["date"], u["count"] = today, 0
            u["count"] += 1

    @st.cache_data
    def example_cache() -> dict:
        return qa.load_example_cache()

    def use_example(question: str, filters: dict) -> None:
        st.session_state["qa_question"] = question
        st.session_state["qa_bands"] = filters.get("bands", [])
        st.session_state["qa_themes"] = filters.get("themes", [])
        st.session_state["qa_value"] = filters.get("value", [])
        st.session_state["qa_stars"] = (1, 5)
        cached = example_cache().get(qa.example_key(question, filters))
        if cached:  # instant, free, and doesn't count against the visitor's limit
            st.session_state.setdefault("qa_history", []).insert(
                0, {"question": question, "filters": filters, "res": cached, "cached": True})

    st.markdown("Ask a question about the reviews. Claude answers from the most relevant reviews only and "
                "cites each one; click a citation to read the review.")
    st.caption("Try an example (saved answers, free):")
    ex_cols = st.columns(2)
    for i, (q, f) in enumerate(qa.EXAMPLES):
        ex_cols[i % 2].button(q, on_click=use_example, args=(q, f), width="stretch", key=f"example_{i}")

    question = st.text_area("Question", key="qa_question", height=80, max_chars=MAX_QUESTION_CHARS,
                            placeholder="e.g. What do buyers of $100-199 headphones say about battery life?")
    f1, f2, f3, f4 = st.columns(4)
    bands = f1.multiselect("Price band", an.BAND_ORDER, key="qa_bands", format_func=an.band_label)
    themes = f2.multiselect("Topic", THEME_LIST, key="qa_themes",
                            format_func=lambda t: t.replace("_", " ").capitalize())
    value = f3.multiselect("Value verdict", list(VALUE_NAMES), key="qa_value", format_func=VALUE_NAMES.get)
    stars = f4.slider("Stars", 1, 5, (1, 5), key="qa_stars")

    with st.expander("Use your own Claude API key (no question limit)"):
        st.text_input("Anthropic API key", type="password", key="visitor_key",
                      help="Used only for your questions in this browser session. It isn't stored or logged.")
        st.caption("Get a key at console.anthropic.com. Each answer costs about \\$0.02 on your account.")
    visitor_key = st.session_state.get("visitor_key", "").strip()

    used = st.session_state.get("qa_asked", 0)
    note = None
    if visitor_key:
        key, blocked, note = visitor_key, None, "Using your API key."
    elif not app_key():
        key, blocked = None, "Questions are unavailable: no API key is configured. Add your own key above."
    elif used >= SESSION_LIMIT:
        key, blocked = None, (f"You've used the {SESSION_LIMIT} free questions for this session. "
                              "Add your own API key above to keep asking.")
    elif daily_count() >= DAILY_LIMIT:
        key, blocked = None, ("The demo has reached its daily question limit. Try again tomorrow, "
                              "or add your own API key above.")
    else:
        key, blocked = app_key(), None
        note = f"{SESSION_LIMIT - used} of {SESSION_LIMIT} free questions left this session."

    ask = st.button("Ask", type="primary", disabled=not question.strip() or not key)
    if blocked:
        st.info(blocked)
    elif note:
        st.caption(note)

    if ask and key:
        import anthropic
        filters = {"bands": bands or None, "themes": themes or None, "value": value or None,
                   "min_rating": float(stars[0]) if stars != (1, 5) else None,
                   "max_rating": float(stars[1]) if stars != (1, 5) else None}
        res = None
        with st.spinner("Finding relevant reviews and asking Claude..."):
            try:
                res = qa.answer(question.strip()[:MAX_QUESTION_CHARS], k=15,
                                client=anthropic.Anthropic(api_key=key, max_retries=3), **filters)
            except anthropic.AuthenticationError:
                st.error("That API key was rejected. Check it and try again.")
            except anthropic.APIError as e:
                st.error(f"Claude API error: {e}")
        if res is not None:
            if not visitor_key:
                st.session_state["qa_asked"] = used + 1
                record_daily_use()
            st.session_state.setdefault("qa_history", []).insert(
                0, {"question": question.strip(), "filters": filters, "res": res, "cached": False})
            st.rerun()  # refresh the remaining-questions note

    def filter_summary(f: dict) -> str:
        parts = []
        if f.get("bands"):
            parts.append(", ".join(an.band_label(b) for b in f["bands"]))
        if f.get("themes"):
            parts.append(", ".join(t.replace("_", " ") for t in f["themes"]))
        if f.get("value"):
            parts.append(", ".join(VALUE_NAMES[v] for v in f["value"]))
        if f.get("min_rating") is not None:
            parts.append(f"{f['min_rating']:.0f}-{f['max_rating']:.0f} stars")
        return " · ".join(parts) if parts else "All reviews"

    def answer_html(text: str, by_id: dict, n: int) -> str:
        """Answer markdown with each citation turned into a link to its review below; hover shows a snippet."""
        def link(m):
            rid = m.group(0)
            r = by_id.get(rid)
            if r is None:
                return rid
            tip = html.escape(f"{r['product_title'][:60]} (${r['price']:.2f}, {r['rating']:.0f} stars): "
                              f"{str(r['text'])[:160]}...", quote=True)
            return f'<a href="#cite-{n}-{rid}" title="{tip}">{rid}</a>'
        text = html.escape(text, quote=False).replace("$", "\\$")
        return qa.CITE_RE.sub(link, text)

    def esc(s) -> str:
        """HTML-escape, with $ as an entity so Streamlit doesn't read it as math."""
        return html.escape(str(s)).replace("$", "&#36;")

    def cited_cards_html(cited: list[str], by_id: dict, n: int) -> str:
        """One collapsible card per cited review, CSS only: a citation link targets the card (#id),
        which opens it via :target; clicking the header toggles a hidden checkbox."""
        verdict_name = {**VALUE_NAMES, "untagged": "Untagged"}
        css = (f"<style>.cite-toggle{{display:none}}"
               f".cite-card{{border:1px solid {C['grid']};border-radius:8px;margin:0 0 6px;padding:8px 12px;"
               f"scroll-margin-top:80px}}"
               f".cite-head{{cursor:pointer;display:block;font-size:0.9rem}}"
               f".cite-head .meta{{color:{C['ink2']}}}"
               f".cite-body{{display:none;margin-top:8px;font-size:0.9rem;line-height:1.5}}"
               f".cite-body .sub{{color:{C['ink2']};font-size:0.8rem;margin-bottom:6px}}"
               f".cite-card:target{{border-color:{C['pos']}}}"
               f".cite-card:target .cite-body,.cite-toggle:checked+.cite-card .cite-body{{display:block}}</style>")
        cards = []
        for rid in cited:
            r = by_id[rid]
            stars_txt = "★" * int(r["rating"]) + "☆" * (5 - int(r["rating"]))
            band_name = an.band_label(qa.band_of(r["price"]))
            body = esc(re.sub(r"<br\s*/?>", "\n", str(r["text"]))).replace("\n", "<br>")
            sub = (f"{esc(r['product_title'])} · {esc(r['store'])} · value verdict: "
                   f"{verdict_name.get(r['value_sentiment'], r['value_sentiment'])}"
                   + (f" · {esc(r['date'])}" if r.get("date") else ""))
            cards.append(
                f'<input type="checkbox" class="cite-toggle" id="cb-{n}-{rid}">'
                f'<div class="cite-card" id="cite-{n}-{rid}">'
                f'<label class="cite-head" for="cb-{n}-{rid}"><b>{rid}</b> '
                f'<span class="meta">· {stars_txt} · &#36;{r["price"]:,.2f} · {esc(band_name)} · '
                f'{esc(r["product_title"][:70])}</span></label>'
                f'<div class="cite-body"><div class="sub">{sub}</div>{body}</div></div>')
        return css + "".join(cards)

    history = st.session_state.get("qa_history", [])
    if history:
        total = sum(h["res"]["cost"] for h in history if not h.get("cached"))
        st.caption(f"{len(history)} question{'s' if len(history) != 1 else ''} this session · "
                   f"total cost \\${total:.3f}")
    for n, h in enumerate(history):
        res = h["res"]
        by_id = {r["review_id"]: r for r in res["reviews"]}
        with st.container(border=True):
            st.markdown(f"**{review_md(h['question'])}**")
            st.caption(f"Filters: {filter_summary(h['filters'])}")
            st.markdown(answer_html(res["answer"], by_id, n), unsafe_allow_html=True)
            cost_txt = (f"saved example answer ({res.get('cached_on', '')}), no cost" if h.get("cached")
                        else f"cost \\${res['cost']:.3f}")
            st.caption(f"{len(res['reviews'])} reviews retrieved, {len(res['cited'])} cited · "
                       f"{cost_txt} · These are the closest-matching reviews, not a random "
                       f"sample, so counts in the answer aren't frequencies.")
            if res["invalid_citations"]:
                st.warning("Cited IDs not among the retrieved reviews (ignored): "
                           + ", ".join(res["invalid_citations"]))
            if res["cited"]:
                st.markdown("**Cited reviews**")
                st.markdown(cited_cards_html(res["cited"], by_id, n), unsafe_allow_html=True)

# ---------------------------------------------------------------- footer (both tabs)

st.divider()
st.caption(
    "**Data:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), McAuley Lab, UC San Diego: "
    "Y. Hou, J. Li, Z. He, A. Yan, X. Chen, J. McAuley, *Bridging Language and Items for Retrieval and "
    "Recommendation*, [arXiv:2403.03952](https://arxiv.org/abs/2403.03952) (2024). A sample of headphone "
    "reviews through September 2023; listed prices are a single snapshot. "
    "Review excerpts are shown as evidence for the analysis and attributed by review ID; the full sample "
    "isn't redistributed. Value tags, price checks and answers are generated by Claude (Anthropic) and can "
    "be wrong. A non-commercial portfolio project, not affiliated with Amazon or the McAuley Lab.")
