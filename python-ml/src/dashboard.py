"""BizCharts Dashboard - Streamlit multi-page app."""

import streamlit as st

st.set_page_config(
    page_title="BizCharts",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Main dashboard entry point."""
    st.sidebar.title("BizCharts")
    st.sidebar.markdown("4chan /biz/ Market Sentiment")

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Main Dashboard", "Coin Explorer", "Historical Analysis", "Post Browser", "Data Explorer", "System Health"],
    )

    if page == "Main Dashboard":
        render_main_dashboard()
    elif page == "Coin Explorer":
        render_coin_explorer()
    elif page == "Historical Analysis":
        render_historical()
    elif page == "Post Browser":
        render_post_browser()
    elif page == "Data Explorer":
        render_data_explorer()
    elif page == "System Health":
        render_system_health()


def render_main_dashboard() -> None:
    """Main Fear/Greed dashboard."""
    st.title("Market Sentiment Overview")

    # Try to load data
    try:
        from .aggregator import SentimentAggregator
        aggregator = SentimentAggregator()
        sentiment = aggregator.get_current_sentiment()

        if sentiment:
            # Fear/Greed Gauge
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.subheader("Fear & Greed Index")
                # Gauge visualization
                import plotly.graph_objects as go

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment.fear_greed_index,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Current Sentiment"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 25], "color": "red"},
                            {"range": [25, 45], "color": "orange"},
                            {"range": [45, 55], "color": "yellow"},
                            {"range": [55, 75], "color": "lightgreen"},
                            {"range": [75, 100], "color": "green"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": sentiment.fear_greed_index,
                        },
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Composition")
                st.metric("Bullish", f"{sentiment.bullish_pct:.1f}%")
                st.metric("Bearish", f"{sentiment.bearish_pct:.1f}%")
                st.metric("Neutral", f"{sentiment.neutral_pct:.1f}%")

            with col3:
                st.subheader("Activity")
                st.metric("Total Posts", f"{sentiment.total_posts:,}")
                st.metric("Unique Threads", f"{sentiment.unique_threads:,}")
                st.metric("Avg Sentiment", f"{sentiment.avg_sentiment:.3f}")

        else:
            st.warning("No sentiment data available yet. Make sure the scraper and analyzer are running.")
            st.info("""
            **Getting Started:**
            1. Start the Rust scraper: `./rust-scraper/target/release/biz-scraper`
            2. Run the analyzer: `python -m src.aggregator`
            3. Refresh this page
            """)

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Database may not be initialized. Run the scraper first.")


def render_coin_explorer() -> None:
    """Coin sentiment explorer."""
    st.title("Coin Explorer")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Search")
        search = st.text_input("Coin Symbol", placeholder="BTC, ETH, SOL...")
        timeframe = st.selectbox("Timeframe", ["24 hours", "7 days", "30 days"])

    with col2:
        if search:
            st.subheader(f"Sentiment for {search.upper()}")
            try:
                from .aggregator import SentimentAggregator
                aggregator = SentimentAggregator()

                hours = {"24 hours": 24, "7 days": 168, "30 days": 720}[timeframe]
                data = aggregator.get_coin_sentiment(search.upper(), hours=hours)

                if data:
                    import plotly.express as px
                    import pandas as pd

                    df = pd.DataFrame([
                        {"timestamp": d.timestamp, "sentiment": d.avg_sentiment, "posts": d.post_count}
                        for d in data
                    ])

                    fig = px.line(df, x="timestamp", y="sentiment", title=f"{search.upper()} Sentiment Over Time")
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

                    # Stats
                    st.metric("Average Sentiment", f"{df['sentiment'].mean():.3f}")
                    st.metric("Total Mentions", f"{df['posts'].sum():,}")
                else:
                    st.info(f"No data found for {search.upper()}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Enter a coin symbol to view sentiment data")

            # Show leaderboard
            st.subheader("Most Mentioned Coins")
            st.info("Leaderboard will appear once data is collected")


def render_historical() -> None:
    """Historical analysis view."""
    st.title("Historical Analysis")

    timeframe = st.selectbox("Timeframe", ["24 hours", "7 days", "30 days", "All time"])

    st.subheader("Market Sentiment Over Time")

    try:
        import duckdb
        from pathlib import Path

        db_path = Path("data/analytics.duckdb")
        if db_path.exists():
            with duckdb.connect(str(db_path)) as conn:
                df = conn.execute("""
                    SELECT bucket_start, fear_greed_index, bullish_pct, bearish_pct, total_posts
                    FROM market_sentiment
                    WHERE bucket_size = 'hour'
                    ORDER BY bucket_start DESC
                    LIMIT 168
                """).fetchdf()

                if not df.empty:
                    import plotly.express as px

                    fig = px.line(df, x="bucket_start", y="fear_greed_index",
                                  title="Fear/Greed Index History")
                    fig.add_hline(y=50, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

                    # Volume chart
                    fig2 = px.bar(df, x="bucket_start", y="total_posts",
                                  title="Post Volume")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No historical data yet")
        else:
            st.warning("Analytics database not found")
    except Exception as e:
        st.error(f"Error: {e}")


def render_post_browser() -> None:
    """Browse raw posts."""
    st.title("Post Browser")

    col1, col2, col3 = st.columns(3)
    with col1:
        coin_filter = st.text_input("Filter by coin", placeholder="BTC")
    with col2:
        sentiment_filter = st.selectbox("Sentiment", ["All", "Bullish", "Bearish", "Neutral"])
    with col3:
        limit = st.slider("Posts to show", 10, 100, 25)

    try:
        import sqlite3
        from pathlib import Path

        db_path = Path("data/posts.db")
        if db_path.exists():
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Build query
                query = """
                    SELECT post_id, thread_id, timestamp, text, has_image
                    FROM posts
                    WHERE text IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                cursor = conn.execute(query, (limit,))
                posts = cursor.fetchall()

                if posts:
                    for post in posts:
                        with st.expander(f"Post #{post['post_id']} - Thread {post['thread_id']}"):
                            st.text(post['text'][:500] if post['text'] else "No text")
                            if post['has_image']:
                                st.caption("Has image")
                else:
                    st.info("No posts found")
        else:
            st.warning("Posts database not found. Start the scraper first.")
    except Exception as e:
        st.error(f"Error: {e}")


def render_data_explorer() -> None:
    """SQL query interface."""
    st.title("Data Explorer")

    st.subheader("Run SQL Query")

    # Pre-built queries
    preset = st.selectbox("Preset Queries", [
        "Custom",
        "Top mentioned coins (24h)",
        "Hourly sentiment average",
        "Most active threads",
    ])

    preset_queries = {
        "Top mentioned coins (24h)": """
SELECT coin_symbol, COUNT(*) as mentions
FROM coin_mentions cm
JOIN posts p ON cm.post_id = p.post_id
WHERE p.timestamp > strftime('%s', 'now', '-1 day')
GROUP BY coin_symbol
ORDER BY mentions DESC
LIMIT 20
""",
        "Hourly sentiment average": """
SELECT
    datetime(bucket_start) as time,
    fear_greed_index,
    total_posts
FROM market_sentiment
WHERE bucket_size = 'hour'
ORDER BY bucket_start DESC
LIMIT 24
""",
        "Most active threads": """
SELECT thread_id, COUNT(*) as posts
FROM posts
GROUP BY thread_id
ORDER BY posts DESC
LIMIT 20
""",
    }

    if preset != "Custom":
        query = st.text_area("Query", preset_queries.get(preset, ""), height=150)
    else:
        query = st.text_area("Query", "SELECT * FROM posts LIMIT 10", height=150)

    db_choice = st.radio("Database", ["SQLite (posts.db)", "DuckDB (analytics.duckdb)"])

    if st.button("Run Query"):
        try:
            from pathlib import Path

            if "SQLite" in db_choice:
                import sqlite3
                db_path = Path("data/posts.db")
                if db_path.exists():
                    with sqlite3.connect(db_path) as conn:
                        import pandas as pd
                        df = pd.read_sql_query(query, conn)
                        st.dataframe(df)

                        # Export option
                        csv = df.to_csv(index=False)
                        st.download_button("Download CSV", csv, "query_results.csv", "text/csv")
                else:
                    st.error("SQLite database not found")
            else:
                import duckdb
                db_path = Path("data/analytics.duckdb")
                if db_path.exists():
                    with duckdb.connect(str(db_path)) as conn:
                        df = conn.execute(query).fetchdf()
                        st.dataframe(df)

                        csv = df.to_csv(index=False)
                        st.download_button("Download CSV", csv, "query_results.csv", "text/csv")
                else:
                    st.error("DuckDB database not found")
        except Exception as e:
            st.error(f"Query error: {e}")


def render_system_health() -> None:
    """System health monitoring."""
    st.title("System Health")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scraper Status")
        try:
            import sqlite3
            from pathlib import Path
            from datetime import datetime

            db_path = Path("data/posts.db")
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    # Get stats
                    total_posts = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
                    total_threads = conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
                    latest = conn.execute("SELECT MAX(timestamp) FROM posts").fetchone()[0]

                    st.metric("Total Posts", f"{total_posts:,}")
                    st.metric("Total Threads", f"{total_threads:,}")

                    if latest:
                        latest_dt = datetime.fromtimestamp(latest)
                        st.metric("Latest Post", latest_dt.strftime("%Y-%m-%d %H:%M"))
                    else:
                        st.metric("Latest Post", "No data")
            else:
                st.warning("Database not found")
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.subheader("Analysis Status")
        try:
            import duckdb
            from pathlib import Path

            db_path = Path("data/analytics.duckdb")
            if db_path.exists():
                with duckdb.connect(str(db_path)) as conn:
                    analyzed = conn.execute("SELECT COUNT(*) FROM sentiment_scores").fetchone()[0]
                    st.metric("Analyzed Posts", f"{analyzed:,}")

                    latest = conn.execute("""
                        SELECT MAX(bucket_start) FROM market_sentiment
                        WHERE bucket_size = 'hour'
                    """).fetchone()[0]
                    if latest:
                        st.metric("Latest Aggregation", str(latest)[:16])
                    else:
                        st.metric("Latest Aggregation", "No data")
            else:
                st.warning("Analytics database not found")
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Database Files")
    from pathlib import Path

    for db_file in ["data/posts.db", "data/analytics.duckdb"]:
        path = Path(db_file)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            st.success(f"{db_file}: {size_mb:.2f} MB")
        else:
            st.warning(f"{db_file}: Not found")


if __name__ == "__main__":
    main()
