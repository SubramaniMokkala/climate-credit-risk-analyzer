
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import google.generativeai as genai
import os

# Page config
st.set_page_config(
    page_title="Climate Credit Risk Analyzer",
    page_icon="🌍",
    layout="wide"
)

# Configure Gemma 4 via Google AI Studio
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
gemma = genai.GenerativeModel("gemma-2-0-flash-exp")

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

def generate_report(company_symbol, df):
    company = df[df["Symbol"] == company_symbol].iloc[0]
    prompt = f"""You are a climate credit risk analyst at a financial institution.
Based on the following data, write a professional 3-paragraph credit risk memo.

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
    response = gemma.generate_content(prompt)
    return response.text

def climate_chat(question, company_symbol, df, report):
    company = df[df["Symbol"] == company_symbol].iloc[0]
    prompt = f"""You are a climate credit risk analyst.
You have already prepared the following risk report for {company["Longname"]}:

{report}

Answer concisely and professionally.
Question: {question}"""
    response = gemma.generate_content(prompt)
    return response.text

# ── MAIN APP ──
st.title("🌍 Climate Credit Risk Analyzer")
st.caption("Powered by XGBoost + Gemma 4 | Data: S&P 500, ND-GAIN, ESG Ratings")

df = load_data()
model = load_model()

# Sidebar
st.sidebar.header("Select Company")
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
page = st.sidebar.radio("Navigation", ["📊 Risk Analyzer", "📄 AI Report", "💬 Chat"])

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
            report = generate_report(selected_symbol, df)
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
