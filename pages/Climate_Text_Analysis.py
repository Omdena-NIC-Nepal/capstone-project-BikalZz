import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from data_utils import analyze_climate_text, analyze_climate_sentiment


def show():
    st.title("📝 Climate Text Analysis")
    
    st.markdown("""
    Analyze climate reports using multiple methods:
    - NLP entity recognition
    - Temperature trends
    - TextBlob sentiment
    - Lexicon-based sentiment (custom CSV)
    """)
    
    user_text = st.text_area("Paste climate-related text:", height=200,
                           placeholder="E.g., 'Global warming is causing catastrophic damage to ecosystems...'")
    
    if not user_text:
        return
        
    # Get all analyses
    nlp_analysis = analyze_climate_text(user_text)
    textblob_sentiment = TextBlob(user_text).sentiment
    lexicon_sentiment = analyze_climate_sentiment(user_text)
    
    # Create tabs
    tab_names = [
        "🌍 Entities", 
        "🌡️ Temperature", 
        "📈 Trends",
        "😊 TextBlob Sentiment",
        "📖 Lexicon Sentiment",
        "🔍 Full Analysis"
    ]
    tabs = st.tabs(tab_names)
    
    # Tab 1: Entities Identification
    with tabs[0]:
        st.subheader("Identified Entities")
        st.dataframe(
            pd.DataFrame(nlp_analysis["climate_entities"], columns=["Entity", "Type"]),
            height=300,
            hide_index=True
        )
        st.subheader("Visualization")
        st.components.v1.html(nlp_analysis["visualization"], height=300, scrolling=True)
    
    # Tab 2: Temperature word Identification
    with tabs[1]:
        if nlp_analysis["temperature_mentions"]:
            st.subheader("🌡️ Temperature Context")
            for i, mention in enumerate(nlp_analysis["temperature_mentions"], 1):
                st.markdown(f"{i}. {mention}")
            
            temp_terms = {k:v for k,v in nlp_analysis["climate_term_frequency"].items() 
                         if k in ["temperature", "heat", "cold", "warming"]}
            if temp_terms:
                st.subheader("Term Frequency")
                st.bar_chart(pd.DataFrame.from_dict(temp_terms, orient="index"))
        else:
            st.warning("No temperature mentions detected")
    
    # Tab 3: Trend word Identification
    with tabs[2]:
        if nlp_analysis["trend_indicators"]:
            trend_df = pd.DataFrame(nlp_analysis["trend_indicators"], columns=["Direction", "Term"])
            st.dataframe(trend_df, hide_index=True)
            
            fig = px.pie(
                names=trend_df["Direction"].value_counts().index,
                values=trend_df["Direction"].value_counts().values,
                title="Trend Direction Distribution",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend indicators found")
    
    # Tab 4:TextBlob Sentiment
    with tabs[3]:
        st.subheader("TextBlob Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        col1.metric("Polarity", f"{textblob_sentiment.polarity:.2f}",
                   delta="Positive" if textblob_sentiment.polarity > 0 else "Negative" if textblob_sentiment.polarity < 0 else "Neutral")
        col2.metric("Subjectivity", f"{textblob_sentiment.subjectivity:.2f}")
        
        st.markdown("""
        **Interpretation**:
        - Polarity: -1 (Negative) → +1 (Positive)
        - Subjectivity: 0 (Objective) → 1 (Subjective)
        """)
    
    # NEW Tab 5: Lexicon Sentiment
    with tabs[4]: 
        # Header 
        st.markdown("""
        <div style="background:#f0f2f6;padding:15px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:#2c3e50;margin:0;">🌡️ Climate Sentiment Analysis</h2>
            <p style="color:#7f8c8d;margin:0;">Using enhanced climate lexicons</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Scorecards with icons
        cols = st.columns(4)
        with cols[0]:
            score = lexicon_sentiment.get('score', 0)
            delta_label = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
            color = "normal" if delta_label=="Positive" else "inverse" if delta_label == "Negative" else "off"
            st.metric(
                label="**Sentiment Score**",
                value=f"{score:.2f}",
                delta=delta_label,
                delta_color=color
            )
        with cols[1]:
            st.metric(
                label="**Negative Intensity**",
                value=f"{lexicon_sentiment.get('neg_score', 0):.2f}",
                help="Higher values indicate stronger negative sentiment"
            )
        with cols[2]:
            st.metric(
                label="**Positive Intensity**",
                value=f"{lexicon_sentiment.get('pos_score', 0):.2f}",
                help="Higher values indicate stronger positive sentiment"
            )
        with cols[3]:
            st.metric(
                label="**Key Phrases**",
                value=len(lexicon_sentiment.get('key_phrases', [])),
                help="Significant phrases affecting sentiment"
            )
        
        # Visualization section
        st.markdown("---")
        st.subheader("📊 Sentiment Distribution")
        
        if lexicon_sentiment.get('sentiment_counts'):
            fig = px.pie(
                names=list(lexicon_sentiment['sentiment_counts'].keys()),
                values=list(lexicon_sentiment['sentiment_counts'].values()),
                hole=0.4,
                color=list(lexicon_sentiment['sentiment_counts'].keys()),
                color_discrete_map={
                    'positive': '#27ae60',
                    'negative': '#e74c3c'
                }
            )
            fig.update_layout(
                margin=dict(t=30, b=10),
                showlegend=False,
                hoverlabel=dict(bgcolor="white")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Key phrases with modern cards
        st.markdown("---")
        st.subheader("🔍 Impactful Phrases")
        
        if lexicon_sentiment.get('key_phrases'):
            for phrase in lexicon_sentiment['key_phrases']:
                bg_color = "#eafaf1" if phrase['polarity'] > 0 else "#fdedec"
                border_color = "#2ecc71" if phrase['polarity'] > 0 else "#e74c3c"
                
                st.markdown(
                    f"""
                    <div style="
                        background:{bg_color};
                        border-left:4px solid {border_color};
                        padding:12px;
                        border-radius:4px;
                        margin-bottom:10px;
                    ">
                        <div style="font-weight:500;margin-bottom:5px;">{phrase['text']}</div>
                        <div style="display:flex;gap:15px;">
                            <span style="color:{border_color}">
                                {phrase['category'].upper()}: {phrase['polarity']:.2f}
                            </span>
                            <span style="color:#7f8c8d">
                                Terms: {', '.join(phrase['terms'])}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No significant phrases detected")
        
        # Matched terms table (collapsible)
        with st.expander("📝 View All Matched Terms", expanded=False):
            if lexicon_sentiment.get('matched_words'):
                df = pd.DataFrame(lexicon_sentiment['matched_words'])
                st.dataframe(
                    df.sort_values('polarity', ascending=False),
                    column_config={
                        "word": "Term",
                        "polarity": st.column_config.ProgressColumn(
                            "Polarity",
                            format="%.2f",
                            min_value=-1,
                            max_value=1
                        ),
                        "category": st.column_config.SelectboxColumn(
                            "Category",
                            options=["positive", "negative"]
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("No lexicon matches found")
        
        # Analysis summary
        st.markdown("---")
        if lexicon_sentiment.get('analysis'):
            st.caption(f"🔬 Analysis Summary: {lexicon_sentiment['analysis']}")

    # Tab 6: Full Analysis (Original)
    with tabs[5]:
        with st.expander("Basic NLP Analysis"):
            st.json({
                "tokens": nlp_analysis["tokens"][:20],
                "lemmas": nlp_analysis["lemmas"][:20],
                "pos_tags": nlp_analysis["pos_tags"][:10],
                "entities": nlp_analysis["entities"]
            }, expanded=False)
        
        with st.expander("Climate-Specific Analysis"):
            st.json({
                "climate_entities": nlp_analysis["climate_entities"],
                "temperature_mentions": nlp_analysis["temperature_mentions"],
                "trend_indicators": nlp_analysis["trend_indicators"],
                "climate_term_frequency": nlp_analysis["climate_term_frequency"]
            }, expanded=False)
        
        with st.expander("Sentiment Comparisons"):
            st.write("TextBlob:", {
                "polarity": textblob_sentiment.polarity,
                "subjectivity": textblob_sentiment.subjectivity
            })
            st.write("Lexicon:", {
                "score": lexicon_sentiment["score"],
                "label": lexicon_sentiment["label"],
                "analysis": lexicon_sentiment["analysis"]
            })
        
        st.subheader("Document Visualization")
        st.components.v1.html(nlp_analysis["visualization"], height=400, scrolling=True)