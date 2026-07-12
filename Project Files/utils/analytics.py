import pandas as pd
import plotly.express as px
import streamlit as st


def show_analytics(history):

    if not history:
        return

    df = pd.DataFrame(history)

    tab1, tab2, tab3 = st.tabs(
        ["Emotions", "Fields", "Summary"]
    )

    with tab1:

        emotion_counts = df["emotion"].value_counts()

        fig = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="Emotion Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:

        fig = px.histogram(
            df,
            x="field",
            color="emotion",
            title="Emotions by Study Field"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:

        st.write("Total Interactions:", len(df))

        st.dataframe(
            df.tail(5),
            use_container_width=True
        )