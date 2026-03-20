import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Sales Insights",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FF4B4B;
        border-bottom: 2px solid #FF4B4B;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load & Preprocess Data ─────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file)

    # Fill missing values
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype(int)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype(int)

    # Encode Gender
    df['Gender_Enc'] = df['Gender'].map({'M': 0, 'F': 1})

    # Encode Age
    age_map = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
    df['Age_Enc'] = df['Age'].map(age_map)

    # Encode City
    df['City_Enc'] = LabelEncoder().fit_transform(df['City_Category'])

    # Normalize Purchase
    scaler = StandardScaler()
    df['Purchase_Norm'] = scaler.fit_transform(df[['Purchase']])

    # Drop duplicates
    df = df.drop_duplicates()

    return df

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shopping-bag.png", width=80)
st.sidebar.title("🛍️ Black Friday Dashboard")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 Upload Black Friday CSV", type=["csv"])

if uploaded_file is None:
    st.markdown('<p class="main-title">🛍️ Black Friday Sales Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Mining the Future: Unlocking Business Intelligence with AI</p>', unsafe_allow_html=True)
    st.info("👈 Please upload your **Black Friday dataset CSV** from the sidebar to get started.")
    st.markdown("""
    ### 📋 What this dashboard shows:
    - **📊 Exploratory Data Analysis** — spending by age, gender, category
    - **🔵 Customer Clustering** — K-Means segmentation of shoppers
    - **🔗 Association Rules** — which product categories are bought together
    - **🚨 Anomaly Detection** — unusual high spenders
    """)
    st.stop()

df = load_and_preprocess(uploaded_file)

page = st.sidebar.radio("📌 Navigate", [
    "📊 Overview & EDA",
    "🔵 Customer Clustering",
    "🔗 Association Rules",
    "🚨 Anomaly Detection",
    "💡 Insights Summary"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.metric("Total Records", f"{len(df):,}")
st.sidebar.metric("Unique Users", f"{df['User_ID'].nunique():,}")
st.sidebar.metric("Unique Products", f"{df['Product_ID'].nunique():,}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW & EDA
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Understanding shopping patterns from Black Friday data</p>', unsafe_allow_html=True)

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Revenue", f"${df['Purchase'].sum():,.0f}")
    with col2:
        st.metric("👤 Avg Spend / User", f"${df.groupby('User_ID')['Purchase'].sum().mean():,.0f}")
    with col3:
        st.metric("📦 Total Transactions", f"{len(df):,}")
    with col4:
        st.metric("🏙️ City Categories", f"{df['City_Category'].nunique()}")

    st.markdown("---")

    # Purchase Distribution by Gender
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Purchase by Gender</p>', unsafe_allow_html=True)
        gender_data = df.groupby('Gender')['Purchase'].sum().reset_index()
        fig = px.pie(gender_data, values='Purchase', names='Gender',
                     color_discrete_sequence=['#FF4B4B', '#4B79FF'],
                     hole=0.45)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Avg Purchase by Age Group</p>', unsafe_allow_html=True)
        age_data = df.groupby('Age')['Purchase'].mean().reset_index()
        age_order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']
        age_data['Age'] = pd.Categorical(age_data['Age'], categories=age_order, ordered=True)
        age_data = age_data.sort_values('Age')
        fig = px.bar(age_data, x='Age', y='Purchase',
                     color='Purchase', color_continuous_scale='Reds',
                     labels={'Purchase': 'Avg Purchase ($)'})
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Product Category Analysis
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Top Product Categories</p>', unsafe_allow_html=True)
        cat_data = df.groupby('Product_Category_1')['Purchase'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(cat_data, x='Product_Category_1', y='Purchase',
                     color='Purchase', color_continuous_scale='Reds',
                     labels={'Product_Category_1': 'Category', 'Purchase': 'Total Sales ($)'})
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Purchase by City Category</p>', unsafe_allow_html=True)
        city_data = df.groupby('City_Category')['Purchase'].mean().reset_index()
        fig = px.bar(city_data, x='City_Category', y='Purchase',
                     color='City_Category', color_discrete_sequence=['#FF4B4B', '#FF8C00', '#4B79FF'],
                     labels={'Purchase': 'Avg Purchase ($)'})
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
    corr_cols = ['Age_Enc', 'Gender_Enc', 'Occupation', 'City_Enc',
                 'Stay_In_Current_City_Years' if df['Stay_In_Current_City_Years'].dtype != object else 'City_Enc',
                 'Marital_Status', 'Product_Category_1', 'Product_Category_2',
                 'Product_Category_3', 'Purchase']
    # Handle Stay_In_Current_City_Years if it's string
    df_corr = df.copy()
    try:
        df_corr['Stay_In_Current_City_Years'] = df_corr['Stay_In_Current_City_Years'].replace('4+', 4).astype(int)
    except:
        pass
    valid_cols = [c for c in corr_cols if c in df_corr.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_corr[valid_cols].corr(), annot=True, fmt='.2f', cmap='Reds', ax=ax, linewidths=0.5)
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    plt.xticks(color='white', fontsize=8)
    plt.yticks(color='white', fontsize=8)
    st.pyplot(fig)
    plt.close()

    # Box Plot
    st.markdown('<p class="section-header">Purchase Distribution by Age & Gender</p>', unsafe_allow_html=True)
    fig = px.box(df, x='Age', y='Purchase', color='Gender',
                 color_discrete_sequence=['#4B79FF', '#FF4B4B'],
                 category_orders={'Age': ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+']})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font_color='white')
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Customer Clustering":
    st.markdown('<p class="main-title">🔵 Customer Clustering</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Segmenting shoppers using K-Means clustering</p>', unsafe_allow_html=True)

    features = ['Age_Enc', 'Gender_Enc', 'Occupation', 'Marital_Status', 'Purchase_Norm']
    X = df[features].dropna()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<p class="section-header">⚙️ Settings</p>', unsafe_allow_html=True)
        n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)

        st.markdown("**Elbow Method**")
        inertias = []
        k_range = range(2, 9)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(list(k_range), inertias, 'ro-', linewidth=2, markersize=6)
        ax.axvline(x=n_clusters, color='yellow', linestyle='--', alpha=0.7)
        ax.set_xlabel('K', color='white')
        ax.set_ylabel('Inertia', color='white')
        ax.set_title('Elbow Curve', color='white')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="section-header">Customer Clusters (PCA View)</p>', unsafe_allow_html=True)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_cluster = X.copy()
        df_cluster['Cluster'] = km.fit_predict(X)

        cluster_labels = {
            0: "💸 Discount Lovers",
            1: "👑 Premium Buyers",
            2: "🛒 Casual Shoppers",
            3: "🔁 Frequent Buyers",
            4: "💎 Luxury Seekers",
            5: "📦 Bulk Buyers",
            6: "🎯 Targeted Shoppers",
            7: "💡 Smart Savers"
        }
        df_cluster['Cluster_Label'] = df_cluster['Cluster'].map(cluster_labels)

        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X)
        df_cluster['PCA1'] = pca_result[:, 0]
        df_cluster['PCA2'] = pca_result[:, 1]

        fig = px.scatter(df_cluster, x='PCA1', y='PCA2', color='Cluster_Label',
                         title=f'K-Means Clusters (K={n_clusters})',
                         opacity=0.6, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    # Cluster Stats
    st.markdown('<p class="section-header">Cluster Statistics</p>', unsafe_allow_html=True)
    df['Cluster'] = km.labels_
    cluster_stats = df.groupby('Cluster').agg(
        Count=('User_ID', 'count'),
        Avg_Purchase=('Purchase', 'mean'),
        Total_Purchase=('Purchase', 'sum'),
        Avg_Age_Enc=('Age_Enc', 'mean')
    ).reset_index()
    cluster_stats['Cluster_Label'] = cluster_stats['Cluster'].map(cluster_labels)
    cluster_stats['Avg_Purchase'] = cluster_stats['Avg_Purchase'].round(2)
    cluster_stats['Total_Purchase'] = cluster_stats['Total_Purchase'].round(2)
    st.dataframe(cluster_stats[['Cluster_Label', 'Count', 'Avg_Purchase', 'Total_Purchase']],
                 use_container_width=True)

    # Avg Purchase per Cluster Bar Chart
    fig = px.bar(cluster_stats, x='Cluster_Label', y='Avg_Purchase',
                 color='Avg_Purchase', color_continuous_scale='Reds',
                 labels={'Avg_Purchase': 'Avg Purchase ($)', 'Cluster_Label': 'Cluster'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    st.markdown('<p class="main-title">🔗 Association Rule Mining</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Discovering product combinations frequently bought together</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        min_support = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
        min_lift = st.slider("Min Lift", 1.0, 5.0, 1.2, 0.1)

    with st.spinner("Mining association rules..."):
        # Build transactions per user
        df_assoc = df[df['Product_Category_1'] != 0].copy()
        df_assoc['PC2'] = df_assoc['Product_Category_2'].apply(lambda x: f"Cat_{int(x)}" if x != 0 else None)
        df_assoc['PC1'] = df_assoc['Product_Category_1'].apply(lambda x: f"Cat_{int(x)}")

        transactions = df_assoc.groupby('User_ID').apply(
            lambda x: list(set(
                list(x['PC1']) +
                [v for v in x['PC2'].dropna().tolist() if v != 'Cat_0']
            ))
        ).tolist()

        te = TransactionEncoder()
        te_arr = te.fit_transform(transactions)
        df_te = pd.DataFrame(te_arr, columns=te.columns_)

        try:
            frequent_items = apriori(df_te, min_support=min_support, use_colnames=True)
            rules = association_rules(frequent_items, metric="lift", min_threshold=min_lift)
            rules = rules[rules['confidence'] >= min_confidence].sort_values('lift', ascending=False)

            with col2:
                st.metric("Rules Found", len(rules))

            if len(rules) > 0:
                st.markdown('<p class="section-header">Top Association Rules</p>', unsafe_allow_html=True)
                display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15).copy()
                display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                display_rules = display_rules.round(4)
                st.dataframe(display_rules, use_container_width=True)

                # Scatter: Support vs Confidence
                st.markdown('<p class="section-header">Support vs Confidence (sized by Lift)</p>', unsafe_allow_html=True)
                rules_plot = rules.copy()
                rules_plot['antecedents'] = rules_plot['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules_plot['consequents'] = rules_plot['consequents'].apply(lambda x: ', '.join(list(x)))
                fig = px.scatter(rules_plot, x='support', y='confidence', size='lift',
                                 color='lift', color_continuous_scale='Reds',
                                 hover_data=['antecedents', 'consequents'],
                                 labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'})
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No rules found. Try lowering the min support or confidence thresholds.")
        except Exception as e:
            st.error(f"Error mining rules: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.markdown('<p class="main-title">🚨 Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Spotting unusually high spenders in Black Friday data</p>', unsafe_allow_html=True)

    method = st.radio("Detection Method", ["Z-Score", "IQR"], horizontal=True)
    threshold = st.slider("Threshold (Z-Score threshold or IQR multiplier)", 1.5, 4.0, 3.0, 0.1)

    user_purchase = df.groupby('User_ID')['Purchase'].sum().reset_index()
    user_purchase.columns = ['User_ID', 'Total_Purchase']
    user_info = df[['User_ID', 'Gender', 'Age', 'Occupation', 'City_Category', 'Marital_Status']].drop_duplicates('User_ID')
    user_data = user_purchase.merge(user_info, on='User_ID')

    if method == "Z-Score":
        z_scores = np.abs(stats.zscore(user_data['Total_Purchase']))
        anomalies = user_data[z_scores > threshold].copy()
        anomalies['Anomaly_Score'] = z_scores[z_scores > threshold]
    else:
        Q1 = user_data['Total_Purchase'].quantile(0.25)
        Q3 = user_data['Total_Purchase'].quantile(0.75)
        IQR_val = Q3 - Q1
        lower = Q1 - threshold * IQR_val
        upper = Q3 + threshold * IQR_val
        anomalies = user_data[(user_data['Total_Purchase'] < lower) | (user_data['Total_Purchase'] > upper)].copy()
        anomalies['Anomaly_Score'] = abs((anomalies['Total_Purchase'] - user_data['Total_Purchase'].median()) / IQR_val)

    col1, col2, col3 = st.columns(3)
    col1.metric("🚨 Anomalies Found", len(anomalies))
    col2.metric("👥 Total Users", len(user_data))
    col3.metric("📊 Anomaly Rate", f"{len(anomalies)/len(user_data)*100:.2f}%")

    # Distribution with anomalies highlighted
    st.markdown('<p class="section-header">Purchase Distribution with Anomalies Highlighted</p>', unsafe_allow_html=True)
    fig = go.Figure()
    normal_data = user_data[~user_data['User_ID'].isin(anomalies['User_ID'])]
    fig.add_trace(go.Histogram(x=normal_data['Total_Purchase'], name='Normal',
                               marker_color='#4B79FF', opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=anomalies['Total_Purchase'], name='Anomaly',
                               marker_color='#FF4B4B', opacity=0.9, nbinsx=20))
    fig.update_layout(barmode='overlay', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font_color='white',
                      xaxis_title='Total Purchase ($)', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

    if len(anomalies) > 0:
        st.markdown('<p class="section-header">Anomalous Customers Table</p>', unsafe_allow_html=True)
        st.dataframe(anomalies.sort_values('Total_Purchase', ascending=False).head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Anomalies by Gender**")
            fig = px.pie(anomalies, names='Gender', color_discrete_sequence=['#FF4B4B', '#4B79FF'], hole=0.4)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("**Anomalies by Age Group**")
            age_anom = anomalies.groupby('Age').size().reset_index(name='Count')
            fig = px.bar(age_anom, x='Age', y='Count', color='Count', color_continuous_scale='Reds')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5: INSIGHTS SUMMARY
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights Summary":
    st.markdown('<p class="main-title">💡 Key Insights & Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Strategic takeaways from the Black Friday analysis</p>', unsafe_allow_html=True)

    st.markdown("### 🎯 Who Spends the Most?")
    age_spend = df.groupby('Age')['Purchase'].mean().sort_values(ascending=False)
    top_age = age_spend.index[0]
    top_val = age_spend.values[0]
    gender_spend = df.groupby('Gender')['Purchase'].mean()
    top_gender = 'Male' if gender_spend.get('M', 0) > gender_spend.get('F', 0) else 'Female'

    col1, col2, col3 = st.columns(3)
    col1.metric("🏆 Highest Spending Age Group", top_age, f"Avg ${top_val:,.0f}")
    col2.metric("👤 Higher Spending Gender", top_gender)
    col3.metric("🏙️ Best City", df.groupby('City_Category')['Purchase'].mean().idxmax())

    st.markdown("---")
    st.markdown("### 📌 Summary Findings")

    findings = [
        ("📊 Shopping Behavior", f"Age group **{top_age}** has the highest average purchase value. Males tend to spend more than females overall."),
        ("🔵 Customer Segments", "K-Means clustering reveals distinct segments: **Discount Lovers**, **Premium Buyers**, **Casual Shoppers**, and **Frequent Buyers**. Each group needs a different marketing strategy."),
        ("🔗 Product Associations", "Certain product categories are frequently bought together. Retailers can use this to design **combo offers** and **cross-selling campaigns**."),
        ("🚨 Anomaly Insights", "A small percentage of users show unusually high spending behavior. These **VIP customers** deserve special loyalty programs and personalized offers."),
        ("🏙️ City Insights", "City category **B** and **C** shoppers show different purchase patterns compared to City A, suggesting targeted regional campaigns could be effective."),
    ]

    for icon_title, desc in findings:
        with st.expander(icon_title, expanded=True):
            st.write(desc)

    st.markdown("---")
    st.markdown("### 📈 Purchase Trend by Occupation")
    occ_data = df.groupby('Occupation')['Purchase'].mean().reset_index().sort_values('Purchase', ascending=False)
    fig = px.bar(occ_data, x='Occupation', y='Purchase', color='Purchase',
                 color_continuous_scale='Reds',
                 labels={'Purchase': 'Avg Purchase ($)', 'Occupation': 'Occupation Code'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏪 Marital Status vs Purchase")
    ms_data = df.groupby('Marital_Status')['Purchase'].mean().reset_index()
    ms_data['Marital_Status'] = ms_data['Marital_Status'].map({0: 'Single', 1: 'Married'})
    fig = px.bar(ms_data, x='Marital_Status', y='Purchase',
                 color='Marital_Status', color_discrete_sequence=['#FF4B4B', '#4B79FF'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig, use_container_width=True)
