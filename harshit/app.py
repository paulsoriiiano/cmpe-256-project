import streamlit as st
import pandas as pd
import numpy as np
import os


st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)


RECS_PATH = 'data/final_recommendations.csv'
HISTORY_PATH = 'data/interim/interactions_full.csv'


def local_css():
    st.markdown("""
    <style>
        /* Global Reset & Dark Theme */
        .stApp {
            background: linear-gradient(180deg, #1c1c1e 0%, #000000 100%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #ffffff;
        }
        
        /* Hide Streamlit Header/Toolbar */
        header {visibility: hidden;}
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Compact Header */
        .header-container {
            padding: 0.5rem 0 1.5rem 0;
            text-align: center;
        }
        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            letter-spacing: -0.02em;
            color: #ffffff;
        }
        .subtitle {
            font-size: 1rem;
            color: #8e8e93;
            font-weight: 400;
            margin-bottom: 0.5rem;
        }
        
        /* Model Info Pill */
        .model-pill {
            display: inline-block;
            background: rgba(255, 255, 255, 0.08);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            color: #fa2d48;
            border: 1px solid rgba(250, 45, 72, 0.2);
            font-weight: 500;
        }

        /* Cards/Containers */
        .card {
            background-color: #1c1c1e;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            border: 1px solid #2c2c2e;
            height: 100%;
        }
        
        /* Section Titles */
        h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 0.8rem;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Inputs */
        .stTextInput > div > div > input {
            background-color: #2c2c2e;
            color: #ffffff;
            border: 1px solid #3a3a3c;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 0.95rem;
        }
        .stTextInput > div > div > input:focus {
            border-color: #fa2d48;
            box-shadow: 0 0 0 1px #fa2d48;
        }
        
        /* Primary Button */
        .stButton > button {
            background-color: #fa2d48;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1.2rem;
            width: 100%;
            transition: all 0.2s ease;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }
        .stButton > button:hover {
            background-color: #d41c36;
            transform: scale(1.01);
            box-shadow: 0 4px 12px rgba(250, 45, 72, 0.3);
        }
        
        /* Status Badges */
        .status-badge {
            background: rgba(48, 209, 88, 0.1);
            color: #30d158;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 0.8rem;
            text-align: center;
            border: 1px solid rgba(48, 209, 88, 0.2);
        }
        .error-badge {
            background: rgba(255, 69, 58, 0.1);
            color: #ff453a;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            margin-top: 0.8rem;
            text-align: center;
            border: 1px solid rgba(255, 69, 58, 0.2);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
            border-bottom: 1px solid #2c2c2e;
            padding-bottom: 0px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 36px;
            white-space: pre-wrap;
            background-color: transparent;
            border: none;
            color: #8e8e93;
            font-weight: 500;
            font-size: 0.9rem;
            padding-top: 0;
            padding-bottom: 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: transparent;
            color: #fa2d48;
            border-bottom: 2px solid #fa2d48;
            font-weight: 600;
        }
        
        /* Tables */
        .dataframe {
            font-size: 0.9rem !important;
        }
        thead tr th {
            background-color: #252527 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-bottom: 1px solid #3a3a3c !important;
            padding: 8px !important;
        }
        tbody tr {
            background-color: #1c1c1e !important;
            border-bottom: 1px solid #2c2c2e !important;
        }
        tbody tr:hover {
            background-color: #2c2c2e !important;
        }
        td {
            padding: 8px !important;
            color: #e0e0e0 !important;
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 3rem 1rem;
            color: #8e8e93;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
        }
        .empty-icon {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            opacity: 0.6;
        }
        
        /* Music Flair */
        .flair-icon {
            font-size: 1.2rem;
            vertical-align: middle;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()



@st.cache_data
def load_data():

    try:
        if not os.path.exists(RECS_PATH):
            return None, None, f"Error: {RECS_PATH} not found."
        
        recs_df = pd.read_csv(RECS_PATH)
        recs_df['user_id'] = recs_df['user_id'].astype(int)
        recs_df.set_index('user_id', inplace=True)
        
        if os.path.exists(HISTORY_PATH):
            history_df = pd.read_csv(HISTORY_PATH)
            history_map = history_df.groupby('user_id')['item_id'].apply(list).to_dict()
        else:
            history_map = {}
            
        return recs_df, history_map, "Success"
    except Exception as e:
        return None, None, str(e)

def get_recommendations(user_id, recs_df):

    if user_id not in recs_df.index:
        return None
    row = recs_df.loc[user_id]
    rec_list = []
    for i, item in enumerate(row):
        rec_list.append({'Rank': i+1, 'Track ID': item})
    return pd.DataFrame(rec_list)

def get_user_history(user_id, history_map):

    items = history_map.get(user_id, [])
    if not items:
        return None
    return pd.DataFrame({'Track ID': items[:50]})




st.markdown("""
    <div class="header-container">
        <h1>Music Recommender <span class="flair-icon">🎶</span></h1>
        <div class="subtitle">Discover Your Next Favorite Track</div>
        <div class="model-pill">ItemCF (k=5) • NDCG@20: 0.0568</div>
    </div>
""", unsafe_allow_html=True)


recs_df, history_map, status = load_data()

if status != "Success":
    st.error(f"Failed to load data: {status}")
    st.stop()


col1, col2 = st.columns([1, 2.5], gap="medium")


with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>🎧 Listener Panel</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8e8e93; font-size: 0.85rem; margin-bottom: 0.8rem;'>Enter a User ID to generate recommendations.</p>", unsafe_allow_html=True)
    
    user_input = st.text_input("User ID", value="0", help="Try IDs like 0, 10, 100...", label_visibility="collapsed", placeholder="Enter User ID")
    
    if st.button("Generate Recommendations"):
        st.session_state['selected_user'] = user_input
    

    if 'selected_user' in st.session_state:
        try:
            uid = int(st.session_state['selected_user'])
            if uid in recs_df.index:
                hist_count = len(history_map.get(uid, []))
                st.markdown(f"""
                    <div class="status-badge">
                        ✅ Loaded User {uid}<br>
                        <span style="font-size: 0.75rem; opacity: 0.8;">{hist_count} History Items</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="error-badge">
                        ❌ User {uid} not found
                    </div>
                """, unsafe_allow_html=True)
        except ValueError:
            st.markdown("""
                <div class="error-badge">
                    ⚠️ Invalid User ID
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)


with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    

    tab1, tab2 = st.tabs(["🔥 Recommendations", "📜 Listening History"])
    
    if 'selected_user' in st.session_state:
        try:
            user_id = int(st.session_state['selected_user'])
            
            with tab1:
                recs = get_recommendations(user_id, recs_df)
                if recs is not None:
                    st.markdown(f"### Top 20 Recommended Tracks for User {user_id}")
                    st.dataframe(
                        recs, 
                        hide_index=True,
                        use_container_width=True,
                        height=500
                    )
                else:
                    st.markdown("""
                        <div class="empty-state">
                            <div class="empty-icon">🔍</div>
                            <p>User not found in database.</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                hist = get_user_history(user_id, history_map)
                if hist is not None:
                    st.markdown(f"### History ({len(history_map[user_id])} tracks)")
                    st.dataframe(
                        hist, 
                        hide_index=True, 
                        use_container_width=True,
                        height=500
                    )
                else:
                    st.markdown("""
                        <div class="empty-state">
                            <div class="empty-icon">📭</div>
                            <p>No listening history found.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
        except ValueError:
            with tab1:
                 st.markdown("""
                    <div class="empty-state">
                        <div class="empty-icon">⚠️</div>
                        <p>Please enter a valid numeric User ID.</p>
                    </div>
                """, unsafe_allow_html=True)
    else:

        with tab1:
            st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">👋</div>
                    <h3>Welcome!</h3>
                    <p>Enter a User ID on the left to see personalized recommendations.</p>
                </div>
            """, unsafe_allow_html=True)
        with tab2:
             st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">📜</div>
                    <p>History will appear here after you select a user.</p>
                </div>
            """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
