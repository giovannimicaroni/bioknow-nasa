import streamlit as st
import base64

# =====================
# CONFIGURAÇÃO GERAL
# =====================
st.set_page_config(page_title="BioKnow", page_icon="🔍", layout="wide")

# =====================
# FUNÇÃO PARA INSERIR IMAGEM LOCAL COMO FUNDO
# =====================
def set_bg_from_local(image_path):
    """Lê imagem local e injeta no CSS como Base64."""
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ===== CHAME AQUI SUA IMAGEM =====
set_bg_from_local("/Users/laura/Documents/Others/NASA_2025/versions/biograph_2/assets/background.png")  # ← altere para o caminho da sua imagem

# Ocultar barra lateral
st.markdown("<style>[data-testid='stSidebarNav']{display:none;}</style>", unsafe_allow_html=True)

# ===== RESTANTE DO SEU CÓDIGO (idêntico) =====
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a1a1a;
        text-align: center;
        margin-top: 3rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box input {
        border: 2px solid #007bff !important;
        border-radius: 50px !important;
        padding: 0.8rem 1.5rem !important;
        font-size: 1rem !important;
    }
    .search-button button {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 50px !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.5rem !important;
    }
    .search-button button:hover {
        background-color: #0056b3 !important;
    }
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 2rem;
    }
    .nav-button button {
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        background-color: black !important;
        border: 1px solid #ccc !important;
    }
    .nav-button button:hover {
        background-color: #eef3ff !important;
        border-color: #007bff !important;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# CONTEÚDO PRINCIPAL
# =====================
st.markdown("<h1 class='main-title'>BioKnow</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>From search to insight</p>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    keyword = st.text_input("", placeholder="Search using keywords", label_visibility="collapsed", key="search_input")
    st.markdown("<div class='search-button' style='text-align:center;'>", unsafe_allow_html=True)
    if st.button("🔍 Search"):
        st.session_state.keyword = keyword
        st.success(f"Searching for: **{keyword}**")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; margin-top:1.5rem;'>", unsafe_allow_html=True)
if st.button("⚙️ Advanced Settings"):
    st.session_state.page = "config"
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Home", key="p1"):
        st.session_state.page = "page1"
        st.rerun()
with col_b:
    if st.button("View Graph", key="p2"):
        st.switch_page("pages/graph.py")
        st.session_state.page = "page2"
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "page1":
    st.title("📄 Page 1")
    st.write("Conteúdo da Página 1.")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

elif st.session_state.page == "page2":
    st.title("📊 Page 2")
    st.write("Conteúdo da Página 2.")
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

elif st.session_state.page == "config":
    st.title("⚙️ Advanced Settings")
    st.checkbox("Enable dark mode")
    st.checkbox("Save search history")
    st.slider("Detail level", 1, 10, 5)
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.rerun()
