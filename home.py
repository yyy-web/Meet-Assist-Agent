import streamlit as st

# ══ 第1步：读取页面参数 ══
page = st.query_params.get("page", "home")

# ══ 第2步：set_page_config ══
st.set_page_config(
    page_title="TalentTalk · AI 招聘助手",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed" if page == "home" else "expanded",
)

# ══ 第3步：主题配置 ══
THEMES = {
    "indigo": {
        "name": "靛蓝",
        "icon": "💜",
        "bg":        "#f8f7ff",
        "bg2":       "#ffffff",
        "primary":   "#6366f1",
        "primary2":  "#818cf8",
        "accent":    "#c084fc",
        "text":      "#1e1b4b",
        "text2":     "#4338ca",
        "muted":     "#6b7280",
        "border":    "rgba(99,102,241,.18)",
        "card_bg":   "rgba(99,102,241,.04)",
        "card_hover":"rgba(99,102,241,.08)",
        "grad":      "linear-gradient(125deg,#6366f1,#818cf8,#c084fc)",
        "glow":      "rgba(99,102,241,.15)",
    },
    "rose": {
        "name": "玫瑰",
        "icon": "🌸",
        "bg":        "#fff7f8",
        "bg2":       "#ffffff",
        "primary":   "#f43f5e",
        "primary2":  "#fb7185",
        "accent":    "#f472b6",
        "text":      "#4c0519",
        "text2":     "#be123c",
        "muted":     "#6b7280",
        "border":    "rgba(244,63,94,.18)",
        "card_bg":   "rgba(244,63,94,.04)",
        "card_hover":"rgba(244,63,94,.08)",
        "grad":      "linear-gradient(125deg,#f43f5e,#fb7185,#f472b6)",
        "glow":      "rgba(244,63,94,.15)",
    },
    "emerald": {
        "name": "翡翠",
        "icon": "🌿",
        "bg":        "#f0fdf4",
        "bg2":       "#ffffff",
        "primary":   "#10b981",
        "primary2":  "#34d399",
        "accent":    "#6ee7b7",
        "text":      "#064e3b",
        "text2":     "#047857",
        "muted":     "#6b7280",
        "border":    "rgba(16,185,129,.18)",
        "card_bg":   "rgba(16,185,129,.04)",
        "card_hover":"rgba(16,185,129,.08)",
        "grad":      "linear-gradient(125deg,#10b981,#34d399,#6ee7b7)",
        "glow":      "rgba(16,185,129,.15)",
    },
    "amber": {
        "name": "琥珀",
        "icon": "🌤️",
        "bg":        "#fffbeb",
        "bg2":       "#ffffff",
        "primary":   "#f59e0b",
        "primary2":  "#fbbf24",
        "accent":    "#fcd34d",
        "text":      "#451a03",
        "text2":     "#b45309",
        "muted":     "#6b7280",
        "border":    "rgba(245,158,11,.18)",
        "card_bg":   "rgba(245,158,11,.04)",
        "card_hover":"rgba(245,158,11,.08)",
        "grad":      "linear-gradient(125deg,#f59e0b,#fbbf24,#fcd34d)",
        "glow":      "rgba(245,158,11,.15)",
    },
    "sky": {
        "name": "天空",
        "icon": "🩵",
        "bg":        "#f0f9ff",
        "bg2":       "#ffffff",
        "primary":   "#0ea5e9",
        "primary2":  "#38bdf8",
        "accent":    "#7dd3fc",
        "text":      "#0c1a2e",
        "text2":     "#0369a1",
        "muted":     "#6b7280",
        "border":    "rgba(14,165,233,.18)",
        "card_bg":   "rgba(14,165,233,.04)",
        "card_hover":"rgba(14,165,233,.08)",
        "grad":      "linear-gradient(125deg,#0ea5e9,#38bdf8,#7dd3fc)",
        "glow":      "rgba(14,165,233,.15)",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "indigo"
T = THEMES[st.session_state.theme]

# ══ 第4步：全局 CSS（主题变量注入）══
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"] {{
    background: {T['bg']} !important;
    color: {T['text']};
    font-family: 'DM Sans', sans-serif;
}}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{ display: none !important; }}

.main .block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}
[data-testid="stAppViewContainer"] > .main > div:first-child {{
    padding-top: 0 !important;
}}

/* ── 主题切换器 ── */
.theme-bar {{
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    background: {T['bg2']};
    border-bottom: 1px solid {T['border']};
    backdrop-filter: blur(12px);
    padding: .55rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 12px {T['glow']};
}}
.theme-brand {{
    font-family: 'Syne', sans-serif;
    font-size: .95rem;
    font-weight: 800;
    background: {T['grad']};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -.02em;
}}
.theme-pills {{
    display: flex;
    gap: .35rem;
    align-items: center;
}}
.theme-pill {{
    font-size: .72rem;
    padding: .22rem .65rem;
    border-radius: 100px;
    border: 1px solid {T['border']};
    background: {T['card_bg']};
    color: {T['text2']};
    cursor: pointer;
    transition: all .18s ease;
    font-weight: 500;
}}
.theme-pill:hover, .theme-pill.active {{
    background: {T['primary']};
    color: #fff;
    border-color: {T['primary']};
}}

/* ── 英雄区 ── */
.hero-wrap {{
    position: relative;
    min-height: 58vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center;
    padding: 7rem 2rem 3.5rem;
    overflow: hidden;
    background: {T['bg']};
}}
.hero-wrap::before {{
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient({T['border']} 1px, transparent 1px),
        linear-gradient(90deg, {T['border']} 1px, transparent 1px);
    background-size: 52px 52px;
    pointer-events: none;
    opacity: .6;
}}
.hero-wrap::after {{
    content: '';
    position: absolute; top: -10%; left: 50%;
    transform: translateX(-50%);
    width: 800px; height: 600px;
    background: radial-gradient(ellipse, {T['glow']} 0%, transparent 68%);
    pointer-events: none;
}}
.hero-wrap > * {{ position: relative; z-index: 1; }}

.badge {{
    display: inline-flex; align-items: center; gap: .4rem;
    background: {T['card_bg']};
    border: 1px solid {T['border']};
    border-radius: 100px; padding: .25rem .85rem;
    font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
    color: {T['text2']}; margin-bottom: 1.4rem;
    font-weight: 500;
}}
.dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {T['primary']};
    box-shadow: 0 0 6px {T['primary']};
    animation: blink 2s ease-in-out infinite;
}}
@keyframes blink {{ 50% {{ opacity:.3; transform:scale(1.5); }} }}

.htitle {{
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.6rem, 5.5vw, 5rem);
    font-weight: 800; line-height: 1.06; letter-spacing: -.03em;
    color: {T['text']};
    margin-bottom: 1rem;
}}
.grad-text {{
    background: {T['grad']};
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hsub {{
    font-size: clamp(.88rem, 1.4vw, 1.05rem);
    font-weight: 300; color: {T['muted']};
    max-width: 460px; line-height: 1.8;
    margin-bottom: 2rem;
}}

/* 统计栏 */
.hero-stats {{
    display: flex; gap: 2.5rem; align-items: center;
    margin-bottom: .5rem;
}}
.stat-item {{ text-align: center; }}
.stat-num {{
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    background: {T['grad']};
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.stat-label {{ font-size: .68rem; color: {T['muted']}; letter-spacing: .04em; margin-top: .1rem; }}
.stat-div {{ width: 1px; height: 2rem; background: {T['border']}; }}

/* ── 分区标签 ── */
.section-row {{
    display: flex; align-items: center; gap: .8rem;
    padding: 0 3vw; max-width: 1280px; margin: 0 auto 1rem;
}}
.section-line {{
    flex: 1; height: 1px;
    background: linear-gradient(90deg, {T['primary']}40, transparent);
}}
.section-line.r {{
    background: linear-gradient(90deg, transparent, {T['primary']}40);
}}
.section-text {{
    font-size: .65rem; letter-spacing: .12em; text-transform: uppercase;
    color: {T['text2']}; font-weight: 600;
}}

/* ── 卡片 ── */
.cards-wrap {{
    padding: 0 3vw 4rem;
    max-width: 1280px;
    margin: 0 auto;
}}
.fcard {{
    position: relative; overflow: hidden;
    background: {T['bg2']};
    border: 1px solid {T['border']};
    border-radius: 16px;
    padding: 1.6rem 1.4rem 1.2rem;
    height: 100%;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,.04), 0 4px 16px {T['glow']};
}}
.fcard::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: {T['grad']};
    opacity: 0;
    transition: opacity .25s ease;
    border-radius: 16px 16px 0 0;
}}
.fcard:hover {{
    transform: translateY(-5px);
    box-shadow: 0 12px 40px {T['glow']}, 0 2px 8px rgba(0,0,0,.06);
    border-color: {T['primary']}55;
}}
.fcard:hover::before {{ opacity: 1; }}

.cnum {{
    position: absolute; top: 1rem; right: 1.2rem;
    font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800;
    color: {T['primary']}12; line-height: 1;
    pointer-events: none; user-select: none;
}}
.cicon {{
    width: 44px; height: 44px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; margin-bottom: .9rem;
    background: {T['card_bg']};
    border: 1px solid {T['border']};
}}
.ctitle {{
    font-family: 'Syne', sans-serif; font-size: .92rem; font-weight: 700;
    color: {T['text']}; margin-bottom: .4rem; letter-spacing: -.01em;
    line-height: 1.3;
}}
.cdesc {{
    font-size: .78rem; color: {T['muted']}; line-height: 1.7; margin-bottom: .9rem;
}}
.tags {{ display: flex; gap: .3rem; flex-wrap: wrap; margin-bottom: .5rem; }}
.tag {{
    font-size: .62rem; padding: .12rem .48rem; border-radius: 100px;
    font-weight: 500; letter-spacing: .03em;
    background: {T['card_bg']};
    color: {T['text2']};
    border: 1px solid {T['border']};
}}

/* ── 导航按钮 ── */
div[data-testid="stButton"] > button {{
    width: 100% !important;
    background: {T['card_bg']} !important;
    border: 1px solid {T['border']} !important;
    border-radius: 9px !important;
    color: {T['text2']} !important;
    font-size: .76rem !important;
    padding: .42rem 1rem !important;
    margin-top: .25rem !important;
    transition: all .2s ease !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}}
div[data-testid="stButton"] > button:hover {{
    background: {T['primary']} !important;
    border-color: {T['primary']} !important;
    color: #fff !important;
    transform: none !important;
}}

/* ── footer ── */
.site-footer {{
    border-top: 1px solid {T['border']};
    padding: 1.8rem 3vw;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 1rem;
    max-width: 1280px; margin: 0 auto;
    color: {T['muted']}; font-size: .75rem;
}}
.footer-brand {{
    font-family: 'Syne', sans-serif; font-size: .85rem; font-weight: 800;
    background: {T['grad']};
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* ── 子页面 sidebar ── */
[data-testid="stSidebar"] {{
    display: flex !important; visibility: visible !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    background: {T['bg2']} !important;
    border-right: 1px solid {T['border']} !important;
    padding: 1.5rem 1rem !important;
}}
[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
[data-testid="stSidebar"] label {{
    color: {T['text2']} !important; font-size: .83rem !important; font-weight: 500 !important;
}}
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stTextInput input {{
    background: {T['card_bg']} !important;
    border-color: {T['border']} !important;
    color: {T['text']} !important;
    border-radius: 8px !important;
}}
</style>
""", unsafe_allow_html=True)

# ══ 第5步：顶部导航栏（主题切换）══
if page == "home":
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    .main .block-container { padding: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # 主题切换按钮（用 Streamlit 原生放在顶部）
    top_cols = st.columns([2, 1, 1, 1, 1, 1, 2])
    labels = [("💜 靛蓝","indigo"),("🌸 玫瑰","rose"),("🌿 翡翠","emerald"),("🌤️ 琥珀","amber"),("🩵 天空","sky")]
    for col, (label, key) in zip(top_cols[1:6], labels):
        with col:
            active = "✦ " if st.session_state.theme == key else ""
            if st.button(f"{active}{label}", key=f"theme_{key}"):
                st.session_state.theme = key
                st.rerun()

    # 顶部品牌栏（HTML 装饰）
    st.markdown(f"""
    <div style="text-align:center;padding:.4rem 0 0;font-family:'Syne',sans-serif;
                font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;
                color:{T['text2']};opacity:.6;font-weight:600;">
        TalentTalk · AI Recruitment Platform
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .main .block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# ══ 第6步：子页面路由 ══
if page == "text_interview":
    from text_interview_app import run as _run
    _run(); st.stop()

elif page == "voice_whisper":
    from voice_interview_app_whisper import run as _run
    _run(); st.stop()

elif page == "voice_assemblyai":
    st.title("☁️ Voice Interview · AssemblyAI (云端)")
    st.info("将 voice_assemblyai_app.py 改造后在此 import")
    if st.button("🏠 返回首页"):
        st.query_params["page"] = "home"; st.rerun()
    st.stop()

elif page == "resume_optimizer":
    from resume_optimizer import run as _run
    _run(); st.stop()

elif page == "auto_apply":
    from auto import run as _run
    _run(); st.stop()

# ══ 第7步：首页内容 ══

# ── 英雄区 ──
st.markdown(f"""
<div class="hero-wrap">
  <div class="badge"><span class="dot"></span>AI-Powered Recruitment Platform</div>
  <h1 class="htitle">
    Land Your<br>
    <span class="grad-text">Dream Job</span>
  </h1>
  <p class="hsub">
    简历优化 · 智能投递 · AI 模拟面试<br>
    覆盖求职全链路，助你拿下心仪 Offer
  </p>
  <div class="hero-stats">
    <div class="stat-item">
      <div class="stat-num">5+</div>
      <div class="stat-label">AI 功能模块</div>
    </div>
    <div class="stat-div"></div>
    <div class="stat-item">
      <div class="stat-num">RAG</div>
      <div class="stat-label">知识增强检索</div>
    </div>
    <div class="stat-div"></div>
    <div class="stat-item">
      <div class="stat-num">Agent</div>
      <div class="stat-label">LangGraph 驱动</div>
    </div>
    <div class="stat-div"></div>
    <div class="stat-item">
      <div class="stat-num">Milvus</div>
      <div class="stat-label">向量语义匹配</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── section 分区标签 ──
st.markdown("""
<div class="section-row">
  <div class="section-line"></div>
  <span class="section-text">功能模块</span>
  <div class="section-line r"></div>
</div>
""", unsafe_allow_html=True)

# ── 卡片数据 ──
CARDS = [
    dict(page="resume_optimizer", icon="📄", num="01",
         title="简历优化 · Resume Optimizer",
         desc="上传简历 + 粘贴 JD，AI 分析匹配度、提炼关键词、重写亮点，精准命中 ATS 过滤器。",
         tags=["ATS 优化", "关键词提取", "差距分析"], btn="进入简历优化 →"),
    dict(page="auto_apply", icon="🚀", num="02",
         title="自动投递 · Auto Apply",
         desc="填写目标岗位，Milvus 余弦相似度语义匹配，生成定制求职信，一键批量投递。",
         tags=["语义匹配", "求职信生成", "Milvus"], btn="进入自动投递 →"),
    dict(page="text_interview", icon="💬", num="03",
         title="文字面试 · Text Interview",
         desc="AI 面试官根据目标岗位出题，支持多轮追问，结束后给出评分报告与改进建议。",
         tags=["多轮对话", "评分报告", "实时反馈"], btn="进入文字面试 →"),
    dict(page="voice_whisper", icon="🎙️", num="04",
         title="语音面试 · Whisper（本地）",
         desc="调用本地 Whisper 模型语音识别，数据不出本机，隐私优先，支持中英双语。",
         tags=["本地离线", "Whisper ASR", "隐私保护"], btn="进入语音面试(本地) →"),
    dict(page="voice_assemblyai", icon="☁️", num="05",
         title="语音面试 · AssemblyAI（云端）",
         desc="接入 AssemblyAI 云端转写，响应更快、准确率更高，支持情绪分析与语速检测。",
         tags=["云端转写", "情绪分析", "高精度 ASR"], btn="进入语音面试(云端) →"),
]

# ── 渲染卡片 ──
st.markdown("<div class='cards-wrap'>", unsafe_allow_html=True)

row1 = st.columns(3, gap="medium")
for col, card in zip(row1, CARDS[:3]):
    with col:
        st.markdown(f"""
        <div class="fcard">
          <span class="cnum">{card['num']}</span>
          <div class="cicon">{card['icon']}</div>
          <div class="ctitle">{card['title']}</div>
          <div class="cdesc">{card['desc']}</div>
          <div class="tags">{''.join(f'<span class="tag">{t}</span>' for t in card['tags'])}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(card["btn"], key=f"nav_{card['page']}"):
            st.query_params["page"] = card["page"]; st.rerun()

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

_, col4, col5, _ = st.columns([0.5, 1, 1, 0.5])
for col, card in zip([col4, col5], CARDS[3:]):
    with col:
        st.markdown(f"""
        <div class="fcard">
          <span class="cnum">{card['num']}</span>
          <div class="cicon">{card['icon']}</div>
          <div class="ctitle">{card['title']}</div>
          <div class="cdesc">{card['desc']}</div>
          <div class="tags">{''.join(f'<span class="tag">{t}</span>' for t in card['tags'])}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(card["btn"], key=f"nav_{card['page']}"):
            st.query_params["page"] = card["page"]; st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ── footer ──
st.markdown(f"""
<div style="padding:0 3vw;">
  <div class="site-footer">
    <span class="footer-brand">TalentTalk</span>
    <span>Built with Streamlit &amp; LangGraph · AI-Powered Interview System</span>
    <span style="color:{T['text2']};font-weight:500;">当前主题：{T['icon']} {T['name']}</span>
  </div>
</div>
""", unsafe_allow_html=True)