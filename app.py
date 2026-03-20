import streamlit as st
import pandas as pd
import numpy as np
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

st.markdown("""
<style>
    .main-title { font-size:2.4rem; font-weight:800; color:#FF4B4B; text-align:center; margin-bottom:0.2rem; }
    .sub-title  { font-size:1.1rem; color:#888; text-align:center; margin-bottom:2rem; }
    .section-header { font-size:1.3rem; font-weight:700; color:#FF4B4B;
                      border-bottom:2px solid #FF4B4B; padding-bottom:0.3rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')

# ─── Load & Preprocess ───────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype(int)
    df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype(int)
    df = df.drop_duplicates()
    df['Gender_Enc'] = df['Gender'].map({'M': 0, 'F': 1})
    age_map = {'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7}
    df['Age_Enc'] = df['Age'].map(age_map)
    df['City_Enc'] = LabelEncoder().fit_transform(df['City_Category'])
    try:
        df['Stay_Enc'] = df['Stay_In_Current_City_Years'].replace('4+', 4).astype(int)
    except:
        df['Stay_Enc'] = 0
    scaler = StandardScaler()
    df['Purchase_Norm'] = scaler.fit_transform(df[['Purchase']])
    return df

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🛍️ Black Friday Dashboard")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload Black Friday CSV", type=["csv"])

if uploaded_file is None:
    st.markdown('<p class="main-title">🛍️ Black Friday Sales Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Mining the Future: Unlocking Business Intelligence with AI</p>', unsafe_allow_html=True)
    st.info("👈 Upload your **Black Friday dataset CSV** from the sidebar to get started.")
    st.markdown("""
    ### 📋 Dashboard Pages:
    - **📊 Overview & EDA** — Spending by age, gender, category
    - **🔵 Customer Clustering** — K-Means segmentation
    - **🔗 Association Rules** — Product combinations
    - **🚨 Anomaly Detection** — Unusual spenders
    - **💡 Insights Summary** — Key findings
    """)
    st.stop()

df = load_data(uploaded_file)

page = st.sidebar.radio("📌 Navigate", [
    "📊 Overview & EDA",
    "🔵 Customer Clustering",
    "🔗 Association Rules",
    "🚨 Anomaly Detection",
    "💡 Insights Summary"
])
st.sidebar.markdown("---")
st.sidebar.metric("Total Records",   f"{len(df):,}")
st.sidebar.metric("Unique Users",    f"{df['User_ID'].nunique():,}")
st.sidebar.metric("Unique Products", f"{df['Product_ID'].nunique():,}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Understanding Black Friday shopping patterns</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue",    f"${df['Purchase'].sum():,.0f}")
    c2.metric("👤 Avg Spend/User",   f"${df.groupby('User_ID')['Purchase'].sum().mean():,.0f}")
    c3.metric("📦 Transactions",     f"{len(df):,}")
    c4.metric("🏙️ City Categories", f"{df['City_Category'].nunique()}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Purchase by Gender</p>', unsafe_allow_html=True)
        g = df.groupby('Gender')['Purchase'].sum().reset_index()
        fig = px.pie(g, values='Purchase', names='Gender', hole=0.45,
                     color_discrete_sequence=['#FF4B4B','#4B79FF'])
        fig.update_layout(**LAYOUT, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Avg Purchase by Age Group</p>', unsafe_allow_html=True)
        age_order = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']
        a = df.groupby('Age')['Purchase'].mean().reindex(age_order).reset_index()
        fig = px.bar(a, x='Age', y='Purchase', color='Purchase',
                     color_continuous_scale='Reds')
        fig.update_layout(**LAYOUT, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Top 10 Product Categories</p>', unsafe_allow_html=True)
        cat = df.groupby('Product_Category_1')['Purchase'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(cat, x='Product_Category_1', y='Purchase', color='Purchase',
                     color_continuous_scale='Reds',
                     labels={'Product_Category_1':'Category','Purchase':'Total Sales ($)'})
        fig.update_layout(**LAYOUT, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Avg Purchase by City</p>', unsafe_allow_html=True)
        city = df.groupby('City_Category')['Purchase'].mean().reset_index()
        fig = px.bar(city, x='City_Category', y='Purchase',
                     color='City_Category',
                     color_discrete_sequence=['#FF4B4B','#FF8C00','#4B79FF'])
        fig.update_layout(**LAYOUT, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Correlation Heatmap</p>', unsafe_allow_html=True)
    corr_cols = ['Age_Enc','Gender_Enc','Occupation','Marital_Status','Stay_Enc',
                 'Product_Category_1','Product_Category_2','Product_Category_3','Purchase']
    valid = [c for c in corr_cols if c in df.columns]
    corr = df[valid].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='Reds', aspect='auto')
    fig.update_layout(**LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Purchase Distribution by Age & Gender</p>', unsafe_allow_html=True)
    fig = px.box(df, x='Age', y='Purchase', color='Gender',
                 color_discrete_sequence=['#4B79FF','#FF4B4B'],
                 category_orders={'Age': age_order})
    fig.update_layout(**LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Customer Clustering":
    st.markdown('<p class="main-title">🔵 Customer Clustering</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Segmenting shoppers using K-Means</p>', unsafe_allow_html=True)

    features = ['Age_Enc','Gender_Enc','Occupation','Marital_Status','Purchase_Norm']
    X = df[features].dropna()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<p class="section-header">⚙️ Settings</p>', unsafe_allow_html=True)
        n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)

        st.markdown("**Elbow Method**")
        inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in range(2, 9)]
        fig = go.Figure(go.Scatter(x=list(range(2,9)), y=inertias, mode='lines+markers',
                                   line=dict(color='#FF4B4B', width=2),
                                   marker=dict(size=8, color='#FF4B4B')))
        fig.add_vline(x=n_clusters, line_dash='dash', line_color='yellow')
        fig.update_layout(**LAYOUT, xaxis_title='K', yaxis_title='Inertia',
                          title='Elbow Curve', margin=dict(t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Clusters (PCA View)</p>', unsafe_allow_html=True)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        cluster_names = {0:'💸 Discount Lovers',1:'👑 Premium Buyers',
                         2:'🛒 Casual Shoppers',3:'🔁 Frequent Buyers',
                         4:'💎 Luxury Seekers',5:'📦 Bulk Buyers',
                         6:'🎯 Targeted Shoppers',7:'💡 Smart Savers'}

        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X)
        df_plot = pd.DataFrame({'PCA1': pca_result[:,0], 'PCA2': pca_result[:,1],
                                 'Cluster': [cluster_names.get(l, str(l)) for l in labels]})
        fig = px.scatter(df_plot, x='PCA1', y='PCA2', color='Cluster', opacity=0.5,
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Cluster Statistics</p>', unsafe_allow_html=True)
    df_c = df.loc[X.index].copy()
    df_c['Cluster'] = labels
    df_c['Cluster_Label'] = df_c['Cluster'].map(cluster_names)
    stats_df = df_c.groupby('Cluster_Label').agg(
        Count=('User_ID','count'),
        Avg_Purchase=('Purchase','mean'),
        Total_Purchase=('Purchase','sum')
    ).round(2).reset_index()
    st.dataframe(stats_df, use_container_width=True)

    fig = px.bar(stats_df, x='Cluster_Label', y='Avg_Purchase', color='Avg_Purchase',
                 color_continuous_scale='Reds',
                 labels={'Avg_Purchase':'Avg Purchase ($)','Cluster_Label':'Cluster'})
    fig.update_layout(**LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ASSOCIATION RULES
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔗 Association Rules":
    st.markdown('<p class="main-title">🔗 Association Rule Mining</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Which product categories are bought together?</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    min_support    = col1.slider("Min Support",    0.01, 0.5,  0.05, 0.01)
    min_confidence = col2.slider("Min Confidence", 0.10, 1.0,  0.30, 0.05)
    min_lift       = col3.slider("Min Lift",       1.0,  5.0,  1.20, 0.10)

    with st.spinner("Mining rules..."):
        df_a = df[df['Product_Category_1'] != 0].copy()
        df_a['PC1'] = df_a['Product_Category_1'].apply(lambda x: f'Cat_{int(x)}')
        df_a['PC2'] = df_a['Product_Category_2'].apply(lambda x: f'Cat_{int(x)}' if x != 0 else None)

        transactions = df_a.groupby('User_ID').apply(
            lambda x: list(set(list(x['PC1']) + [v for v in x['PC2'].dropna() if v != 'Cat_0']))
        ).tolist()

        te = TransactionEncoder()
        df_te = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

        try:
            freq = apriori(df_te, min_support=min_support, use_colnames=True)
            rules = association_rules(freq, metric='lift', min_threshold=min_lift)
            rules = rules[rules['confidence'] >= min_confidence].sort_values('lift', ascending=False)

            st.metric("Rules Found", len(rules))

            if len(rules) > 0:
                st.markdown('<p class="section-header">Top Rules</p>', unsafe_allow_html=True)
                disp = rules[['antecedents','consequents','support','confidence','lift']].head(15).copy()
                disp['antecedents'] = disp['antecedents'].apply(lambda x: ', '.join(list(x)))
                disp['consequents'] = disp['consequents'].apply(lambda x: ', '.join(list(x)))
                st.dataframe(disp.round(4), use_container_width=True)

                r2 = rules.copy()
                r2['antecedents'] = r2['antecedents'].apply(lambda x: ', '.join(list(x)))
                r2['consequents'] = r2['consequents'].apply(lambda x: ', '.join(list(x)))
                fig = px.scatter(r2, x='support', y='confidence', size='lift',
                                 color='lift', color_continuous_scale='Reds',
                                 hover_data=['antecedents','consequents'])
                fig.update_layout(**LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No rules found — try lowering the thresholds.")
        except Exception as e:
            st.error(f"Error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.markdown('<p class="main-title">🚨 Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Spotting unusually high spenders</p>', unsafe_allow_html=True)

    method    = st.radio("Detection Method", ["Z-Score", "IQR"], horizontal=True)
    threshold = st.slider("Threshold", 1.5, 4.0, 3.0, 0.1)

    user_purchase = df.groupby('User_ID')['Purchase'].sum().reset_index()
    user_purchase.columns = ['User_ID','Total_Purchase']
    user_info = df[['User_ID','Gender','Age','Occupation','City_Category','Marital_Status']].drop_duplicates('User_ID')
    ud = user_purchase.merge(user_info, on='User_ID')

    if method == "Z-Score":
        z = np.abs(stats.zscore(ud['Total_Purchase']))
        anomalies = ud[z > threshold].copy()
    else:
        Q1, Q3 = ud['Total_Purchase'].quantile(0.25), ud['Total_Purchase'].quantile(0.75)
        IQR_v = Q3 - Q1
        anomalies = ud[(ud['Total_Purchase'] < Q1 - threshold*IQR_v) |
                       (ud['Total_Purchase'] > Q3 + threshold*IQR_v)].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("🚨 Anomalies",   len(anomalies))
    c2.metric("👥 Total Users", len(ud))
    c3.metric("📊 Rate",        f"{len(anomalies)/len(ud)*100:.2f}%")

    normal = ud[~ud['User_ID'].isin(anomalies['User_ID'])]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=normal['Total_Purchase'],    name='Normal',  marker_color='#4B79FF', opacity=0.7, nbinsx=50))
    fig.add_trace(go.Histogram(x=anomalies['Total_Purchase'], name='Anomaly', marker_color='#FF4B4B', opacity=0.9, nbinsx=20))
    fig.update_layout(**LAYOUT, barmode='overlay', xaxis_title='Total Purchase ($)', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)

    if len(anomalies) > 0:
        st.markdown('<p class="section-header">Anomalous Customers</p>', unsafe_allow_html=True)
        st.dataframe(anomalies.sort_values('Total_Purchase', ascending=False).head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(anomalies, names='Gender', hole=0.4, title='By Gender',
                         color_discrete_sequence=['#FF4B4B','#4B79FF'])
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ag = anomalies.groupby('Age').size().reset_index(name='Count')
            fig = px.bar(ag, x='Age', y='Count', color='Count', color_continuous_scale='Reds', title='By Age Group')
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INSIGHTS SUMMARY
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights Summary":
    st.markdown('<p class="main-title">💡 Key Insights & Recommendations</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Strategic takeaways from Black Friday data</p>', unsafe_allow_html=True)

    top_age  = df.groupby('Age')['Purchase'].mean().idxmax()
    top_val  = df.groupby('Age')['Purchase'].mean().max()
    top_city = df.groupby('City_Category')['Purchase'].mean().idxmax()
    g_avg    = df.groupby('Gender')['Purchase'].mean()
    top_gen  = 'Male' if g_avg.get('M', 0) > g_avg.get('F', 0) else 'Female'

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Top Age Group",    top_age,  f"Avg ${top_val:,.0f}")
    c2.metric("👤 Higher Spender",   top_gen)
    c3.metric("🏙️ Best City",        f"City {top_city}")

    st.markdown("---")
    for title, body in [
        ("📊 Shopping Behavior",  f"Age group **{top_age}** has the highest average purchase. {'Males' if top_gen=='Male' else 'Females'} spend more overall."),
        ("🔵 Customer Segments",  "K-Means reveals 4 distinct groups: Discount Lovers, Premium Buyers, Casual Shoppers, and Frequent Buyers — each needing a different marketing strategy."),
        ("🔗 Product Associations","Certain product categories are frequently co-purchased. Use these rules to design combo offers and cross-selling campaigns."),
        ("🚨 Anomaly Insights",   "A small group of VIP customers spend significantly more than average. Target them with exclusive loyalty programs and personalized offers."),
        ("🏙️ City Insights",      "City B and C shoppers behave differently from City A — regional campaigns can maximize engagement."),
    ]:
        with st.expander(title, expanded=True):
            st.write(body)

    st.markdown("---")
    st.markdown('<p class="section-header">Avg Purchase by Occupation</p>', unsafe_allow_html=True)
    occ = df.groupby('Occupation')['Purchase'].mean().reset_index().sort_values('Purchase', ascending=False)
    fig = px.bar(occ, x='Occupation', y='Purchase', color='Purchase', color_continuous_scale='Reds')
    fig.update_layout(**LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Marital Status vs Purchase</p>', unsafe_allow_html=True)
    ms = df.groupby('Marital_Status')['Purchase'].mean().reset_index()
    ms['Marital_Status'] = ms['Marital_Status'].map({0:'Single', 1:'Married'})
    fig = px.bar(ms, x='Marital_Status', y='Purchase', color='Marital_Status',
                 color_discrete_sequence=['#FF4B4B','#4B79FF'])
    fig.update_layout(**LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
