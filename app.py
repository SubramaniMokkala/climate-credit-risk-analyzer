import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from google import genai
import os

# Page config
st.set_page_config(
    page_title="CarbonLens",
    page_icon="🌍",
    layout="wide"
)

# Configure Gemma 4 via Google AI Studio
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Risk tier colors
TIER_COLORS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e67e22",
    "Critical": "#e74c3c"
}

@st.cache_data
def load_data():
    df = pd.read_csv("climate_credit_risk_data.csv")
    return df

@st.cache_resource
def load_model():
    with open("climate_credit_risk_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def generate_report(company_symbol, df, language="English"):
    company = df[df["Symbol"] == company_symbol].iloc[0]
    lang_note = lang_instruction.get(language, "Write the report in English.")
    prompt = f"""You are a climate credit risk analyst at a financial institution.
Based on the following data, write a professional 3-paragraph credit risk memo.
{lang_note}
Company: {company["Longname"]}
Sector: {company["Sector"]}
Industry: {company["Industry"]}
Country: {company["Country"]}
Market Cap: ${company["Marketcap"]/1e9:.1f}B
Revenue Growth: {company["Revenuegrowth"]*100:.1f}%
Climate Risk Assessment:
- Climate Credit Risk Tier: {company["climate_risk_tier"]}
- Climate Credit Risk Score: {company["climate_credit_risk_score"]:.2f} / 1.0
- Country Climate Vulnerability: {company["vulnerability_score"]:.3f}
- Country Climate Readiness: {company["readiness_score"]:.3f}
- ND-GAIN Country Score: {company["gain_score"]:.1f} / 100
- High Carbon Sector: {"Yes" if company["high_carbon_sector"] == 1 else "No"}
- ESG Risk Level: {company.get("ESG Risk Level", "N/A")}
Write exactly 3 paragraphs:
Paragraph 1: Company overview and sector climate exposure
Paragraph 2: Key climate risk drivers and financial implications
Paragraph 3: Risk tier conclusion and recommendation for lenders"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < 2:
                import time
                time.sleep(3)
            else:
                return f"Error generating report: {str(e)}"

def climate_chat(question, company_symbol, df, report):
    company = df[df["Symbol"] == company_symbol].iloc[0]
    report_summary = report[:1500] if len(report) > 1500 else report
    prompt = f"""You are a climate credit risk analyst.
You have prepared a risk report for {company["Longname"]} with these key facts:
- Climate Risk Tier: {company["climate_risk_tier"]}
- Risk Score: {company["climate_credit_risk_score"]:.2f} / 1.0
- Sector: {company["Sector"]}
- High Carbon Sector: {"Yes" if company["high_carbon_sector"] == 1 else "No"}
- ESG Risk Level: {company.get("ESG Risk Level", "N/A")}
- Country Readiness: {company["readiness_score"]:.3f}
Report excerpt:
{report_summary}
Answer this question concisely and professionally in 2-3 sentences:
{question}"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < 2:
                import time
                time.sleep(3)
            else:
                return f"Error generating response: {str(e)}"

# ── MAIN APP ──
st.title("🌍 CarbonLens")
st.caption("Powered by XGBoost + Gemma 4 | Data: S&P 500, ND-GAIN, ESG Ratings")

df = load_data()
model = load_model()

# Sidebar
st.sidebar.header("Select Company")

# Impact Counter
st.sidebar.markdown("---")
st.sidebar.markdown("### 🌍 CarbonLens Impact")
st.sidebar.metric("Companies Analyzed", "421")
st.sidebar.metric("Fossil Fuel Risk Flagged", "$869B")
st.sidebar.metric("Cost to Access", "$0")
st.sidebar.markdown("---")

# Language selector
language = st.sidebar.selectbox(
    "🌐 Report Language",
    ["English", "Hindi", "Spanish", "French", "German", "Arabic"]
)

lang_instruction = {
    "English": "Write the report in English.",
    "Hindi": "Write the report in Hindi (हिंदी).",
    "Spanish": "Write the report in Spanish (Español).",
    "French": "Write the report in French (Français).",
    "German": "Write the report in German (Deutsch).",
    "Arabic": "Write the report in Arabic (العربية)."
}

sectors = ["All"] + sorted(df["Sector"].unique().tolist())
selected_sector = st.sidebar.selectbox("Filter by Sector", sectors)

filtered_df = df if selected_sector == "All" else df[df["Sector"] == selected_sector]

company_options = filtered_df[["Symbol", "Shortname"]].apply(
    lambda x: f"{x['Symbol']} — {x['Shortname']}", axis=1
).tolist()

selected = st.sidebar.selectbox("Company", company_options)
selected_symbol = selected.split(" — ")[0]
company = df[df["Symbol"] == selected_symbol].iloc[0]

# Navigation
page = st.sidebar.radio("Navigation", ["📊 Risk Analyzer", "📄 AI Report", "💬 Chat", "🔍 Explainability"])

# ── PAGE 1: RISK ANALYZER ──
if page == "📊 Risk Analyzer":
    st.header(f"📊 {company['Longname']}")

    col1, col2, col3, col4 = st.columns(4)
    tier = str(company["climate_risk_tier"])
    color = TIER_COLORS.get(tier, "#95a5a6")

    col1.metric("Risk Tier", tier)
    col2.metric("Risk Score", f"{company['climate_credit_risk_score']:.2f} / 1.0")
    col3.metric("Sector", company["Sector"])
    col4.metric("Market Cap", f"${company['Marketcap']/1e9:.1f}B")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Score Breakdown")
        categories = ["ESG Risk", "Vulnerability", "Carbon Sector", "Low Readiness", "Low Profitability"]
        values = [
            company["norm_esg"] * 0.30,
            company["norm_vulnerability"] * 0.25,
            company["norm_carbon_sector"] * 0.20,
            company["norm_readiness_inv"] * 0.15,
            company["norm_profit_inv"] * 0.10
        ]
        fig = go.Figure(go.Bar(
            x=values, y=categories,
            orientation="h",
            marker_color=color
        ))
        fig.update_layout(xaxis_title="Contribution to Risk Score", height=300,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Country Climate Profile")
        fig2 = go.Figure(go.Scatterpolar(
            r=[company["gain_score"]/100,
               1 - company["vulnerability_score"],
               company["readiness_score"],
               0.5],
            theta=["GAIN Score", "Low Vulnerability", "Readiness", "Climate Gap"],
            fill="toself",
            marker_color=color
        ))
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Sector Risk Comparison")
    sector_avg = df.groupby("Sector")["climate_credit_risk_score"].mean().sort_values()
    fig3 = go.Figure(go.Bar(
        x=sector_avg.values,
        y=sector_avg.index,
        orientation="h",
        marker_color=["#e74c3c" if s == company["Sector"] else "#3498db" for s in sector_avg.index]
    ))
    fig3.update_layout(xaxis_title="Avg Climate Credit Risk Score", height=350,
                       margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig3, use_container_width=True)

# ── PAGE 2: AI REPORT ──
elif page == "📄 AI Report":
    st.header(f"📄 AI Risk Report — {company['Longname']}")
    st.caption("Generated by Gemma 4 via Google AI Studio")

    if st.button("🔄 Generate Report", type="primary"):
        with st.spinner("Gemma 4 is analyzing climate risk data..."):
            report = generate_report(selected_symbol, df, language)
            st.session_state["report"] = report
            st.session_state["report_symbol"] = selected_symbol

    if "report" in st.session_state and st.session_state.get("report_symbol") == selected_symbol:
        st.markdown(st.session_state["report"])
        st.download_button(
            "⬇️ Download Report",
            st.session_state["report"],
            file_name=f"{selected_symbol}_climate_risk_report.txt",
            mime="text/plain"
        )

# ── PAGE 3: CHAT ──
elif page == "💬 Chat":
    st.header(f"💬 Ask About {company['Longname']}")
    st.caption("Ask Gemma 4 questions about the climate risk assessment")

    if "report" not in st.session_state or st.session_state.get("report_symbol") != selected_symbol:
        st.warning("⚠️ Please generate a report first on the AI Report page.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Ask about this company's climate risk..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer = climate_chat(prompt, selected_symbol, df, st.session_state["report"])
                    st.write(answer)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})

# ── PAGE 4: EXPLAINABILITY ──
elif page == "🔍 Explainability":
    st.header(f"🔍 Why is {company['Longname']} rated {company['climate_risk_tier']}?")
    st.caption("Transparent risk factor breakdown — powered by explainable AI")

    # Risk score breakdown
    st.subheader("Risk Score Components")
    components = {
        "ESG Risk (30%)": company["norm_esg"] * 0.30,
        "Climate Vulnerability (25%)": company["norm_vulnerability"] * 0.25,
        "High Carbon Sector (20%)": company["norm_carbon_sector"] * 0.20,
        "Low Climate Readiness (15%)": company["norm_readiness_inv"] * 0.15,
        "Low Profitability (10%)": company["norm_profit_inv"] * 0.10,
    }

    total = sum(components.values())
    for label, value in components.items():
        pct = (value / total * 100) if total > 0 else 0
        col1, col2, col3 = st.columns([3, 1, 1])
        col1.write(label)
        col2.write(f"{value:.3f}")
        col3.write(f"{pct:.1f}%")

    st.divider()

    # Visual bar chart
    import plotly.graph_objects as go
    tier = str(company["climate_risk_tier"])
    color_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e67e22", "Critical": "#e74c3c"}
    color = color_map.get(tier, "#95a5a6")

    fig = go.Figure(go.Bar(
        x=list(components.values()),
        y=list(components.keys()),
        orientation="h",
        marker_color=color,
        text=[f"{v:.3f}" for v in components.values()],
        textposition="outside"
    ))
    fig.update_layout(
        title=f"Risk Factor Contributions — {company['Shortname']}",
        xaxis_title="Contribution to Climate Credit Risk Score",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Key facts table
    st.subheader("Key Risk Indicators")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Risk Score", f"{company['climate_credit_risk_score']:.2f} / 1.0")
        st.metric("ESG Risk Level", str(company.get("ESG Risk Level", "N/A")))
        st.metric("High Carbon Sector", "Yes ⚠️" if company["high_carbon_sector"] == 1 else "No ✅")
    with col2:
        st.metric("Country Vulnerability", f"{company['vulnerability_score']:.3f}")
        st.metric("Country Readiness", f"{company['readiness_score']:.3f}")
        st.metric("ND-GAIN Score", f"{company['gain_score']:.1f} / 100")

    st.divider()

    # Gemma 4 plain-language explanation
    st.subheader("🤖 Gemma 4 Explains")
    if st.button("Generate Plain-Language Explanation", type="primary"):
        with st.spinner("Gemma 4 is analyzing..."):
            explain_prompt = f"""You are a climate risk explainer for non-technical audiences.
Explain in simple, plain language (3-4 sentences) why {company["Longname"]} 
received a {company["climate_risk_tier"]} climate credit risk rating with a score 
of {company["climate_credit_risk_score"]:.2f} out of 1.0.
Key facts:
- Sector: {company["Sector"]}
- High Carbon Sector: {"Yes" if company["high_carbon_sector"] == 1 else "No"}
- ESG Risk Level: {company.get("ESG Risk Level", "N/A")}
- Country Climate Readiness: {company["readiness_score"]:.3f}
Use simple language a non-expert can understand. Avoid jargon."""

            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemma-4-31b-it",
                        contents=explain_prompt
                    )
                    st.info(response.text)
                    break
                except Exception as e:
                    if attempt < 2:
                        import time
                        time.sleep(3)
                    else:
                        st.error(f"Error: {str(e)}")
