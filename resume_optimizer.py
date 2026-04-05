# resume_optimizer.py
from __future__ import annotations

import os
import re
from typing import TypedDict, Optional

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END


# ─────────────────────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────────────────────
class ResumeState(TypedDict):
    resume_text: str
    job_description: str
    mode: str
    parsed_info: str
    gap_analysis: str
    optimized_resume: str
    suggestions: str
    score: int
    score_reason: str
    retry_count: int

MAX_RETRY = 2


# ─────────────────────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────────────────────
def _make_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )


# ─────────────────────────────────────────────────────────────
#  节点
# ─────────────────────────────────────────────────────────────
def parse_node(state: ResumeState) -> ResumeState:
    llm = _make_llm(temperature=0.0)
    system = SystemMessage(content=(
        "你是一名资深 HR 顾问，擅长分析简历与岗位要求的匹配度。"
        "请严格按照 [结构化简历] / [JD关键词] / [差距分析] 三个 section 输出，不要多余解释。"
    ))
    user = HumanMessage(content=f"""
## 原始简历
{state["resume_text"]}

## 目标岗位描述（JD）
{state["job_description"]}

请输出：

### [结构化简历]
用 Markdown 列表列出简历中的：技能、工作经历（公司/职位/时间/职责）、教育背景、项目经历。

### [JD关键词]
从 JD 中提取：必备技能、加分技能、软素质要求、行业背景要求，各自列表。

### [差距分析]
对比上面两部分，列出：
1. 简历中缺失但 JD 要求的关键点
2. 简历中表述模糊需要强化的点
3. 简历中与 JD 无关可精简的内容
""")
    response = llm.invoke([system, user])
    content = response.content
    parsed_info = _extract_section(content, "[结构化简历]", "[JD关键词]")
    gap_analysis = _extract_section(content, "[差距分析]", None)
    return {**state, "parsed_info": parsed_info or content, "gap_analysis": gap_analysis or ""}


def optimize_node(state: ResumeState) -> ResumeState:
    llm = _make_llm(temperature=0.7)
    mode_prompt = {
        "concise":    "语言简洁，每条经历控制在 3 句以内，突出量化成果。",
        "detailed":   "内容详实，充分展开项目背景、技术方案和个人贡献。",
        "aggressive": "大胆强化关键词密度，主动匹配 JD 要求，使用与 JD 一致的措辞。",
    }.get(state.get("mode", "concise"), "语言简洁，突出量化成果。")
    retry_hint = ""
    if state.get("retry_count", 0) > 0 and state.get("score_reason"):
        retry_hint = f"\n\n⚠️ 上一版本评分不足，评审意见如下，请针对性改进：\n{state['score_reason']}"
    system = SystemMessage(content=(
        "你是一名专业简历优化师，能将普通简历改写成高竞争力的求职文档。"
        f"优化风格要求：{mode_prompt}"
        "请严格按照 [优化后简历] / [修改说明] 两个 section 输出。"
    ))
    user = HumanMessage(content=f"""
## 结构化简历与差距分析
{state["parsed_info"]}

{state["gap_analysis"]}

## JD 核心要求
{state["job_description"][:800]}

{retry_hint}

请输出：

### [优化后简历]
完整重写简历，Markdown 格式，包含：
- 个人简介（3-4 句，嵌入 JD 关键词）
- 核心技能（分类列表）
- 工作经历（STAR 法则，含量化数据）
- 项目经历
- 教育背景

### [修改说明]
用编号列表说明做了哪些主要改动及原因（≤10 条）。
""")
    response = llm.invoke([system, user])
    content = response.content
    optimized = _extract_section(content, "[优化后简历]", "[修改说明]")
    suggestions = _extract_section(content, "[修改说明]", None)
    return {**state, "optimized_resume": optimized or content, "suggestions": suggestions or ""}


def review_node(state: ResumeState) -> ResumeState:
    llm = _make_llm(temperature=0.0)
    system = SystemMessage(content=(
        "你是一名严格的简历质量审核员，请从匹配度、清晰度、量化成果、关键词覆盖四个维度综合评分。"
        "只输出 JSON，格式：{\"score\": <0-100整数>, \"reason\": \"<100字以内的评分说明>\"}"
    ))
    user = HumanMessage(content=f"""
## 目标 JD（摘要）
{state["job_description"][:600]}

## 优化后简历
{state["optimized_resume"][:2000]}

请评分并输出 JSON。
""")
    response = llm.invoke([system, user])
    raw = response.content.strip()
    score, reason = _parse_score_json(raw)
    return {**state, "score": score, "score_reason": reason, "retry_count": state.get("retry_count", 0)}


# ─────────────────────────────────────────────────────────────
#  路由
# ─────────────────────────────────────────────────────────────
def should_retry(state: ResumeState) -> str:
    if state["score"] < 75 and state.get("retry_count", 0) < MAX_RETRY:
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "optimize"
    return "end"


# ─────────────────────────────────────────────────────────────
#  Graph
# ─────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ResumeState)
    graph.add_node("parse",    parse_node) #节点
    graph.add_node("optimize", optimize_node)
    graph.add_node("review",   review_node)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "optimize") #边
    graph.add_edge("optimize", "review")
    # 条件边：如果评分不达标且未超过最大重试次数，则返回 optimize 继续优化；否则结束
    graph.add_conditional_edges("review", should_retry, {"optimize": "optimize", "end": END})
    return graph.compile()


# ─────────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────────
def _extract_section(text: str, start_marker: str, end_marker: Optional[str]) -> str:
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)
    if end_marker:
        end_idx = text.find(end_marker, start_idx)
        return text[start_idx:end_idx].strip() if end_idx != -1 else text[start_idx:].strip()
    return text[start_idx:].strip()


def _parse_score_json(raw: str) -> tuple[int, str]:
    try:
        import json
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = int(data.get("score", 70))
            reason = str(data.get("reason", ""))
            return max(0, min(100, score)), reason
    except Exception:
        pass
    nums = re.findall(r"\b(\d{2,3})\b", raw)
    score = int(nums[0]) if nums else 70
    return max(0, min(100, score)), raw[:200]


def extract_pdf_text(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)


# ─────────────────────────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────────────────────────
def run():
    # ── 只注入必要的补充样式，不影响 sidebar ──
    st.markdown("""
    <style>
    .score-badge {
        display: inline-block; padding: 0.3rem 0.8rem;
        border-radius: 999px; font-weight: 600; font-size: 1rem;
    }
    /* 返回首页按钮固定在右上角，但确保不遮挡 sidebar 按钮 */
    .home-btn-wrap {
        background: #f0f9ff;
        position: fixed; top: 0.8rem; right: 1.2rem; z-index: 9999;
    }
    .home-btn-wrap a {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(15,17,30,0.9);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px; padding: 0.4rem 0.9rem;
        color: #94a3b8 !important; font-size: 0.82rem;
        text-decoration: none !important; transition: all 0.2s ease;
        backdrop-filter: blur(5px);
    }
    .home-btn-wrap a:hover {
        background: #f0f9ff;
        border-color: rgba(99,102,241,0.4);
        color: #a5b4fc !important;
    }
    
    /* 主内容区标题、文字颜色修复 - 但不影响 sidebar */
    .main h1, .main h2, .main h3 { color: #f0f4ff !important; }
    .main p, .main label, .main span:not([data-testid="stSidebar"] *) { 
        color: #dde1ee; 
    }
    
    /* 确保主内容区不会覆盖 sidebar */
    .main > div {
        margin-left: 0 !important;
    }
    
    /* 修复 tabs 样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        color: #a5b4fc !important;
    }
    </style>
    <div class="home-btn-wrap">
        <a href="?page=home" target="_self">🏠 返回首页</a>
    </div>
    """, unsafe_allow_html=True)

    st.title("📄 简历优化 Agent")
    st.caption("上传简历 PDF + 粘贴 JD，AI 自动分析差距并重写简历")

    # ── Sidebar ──
    st.sidebar.header("⚙️ 优化配置")
    resume_file = st.sidebar.file_uploader("上传简历（PDF）", type=["pdf"], key="ro_resume")
    job_desc = st.sidebar.text_area(
        "粘贴目标岗位 JD", height=200, key="ro_jd",
        placeholder="将招聘网站上的岗位描述粘贴到这里..."
    )
    mode = st.sidebar.selectbox(
        "优化风格",
        options=["concise", "detailed", "aggressive"],
        format_func=lambda x: {
            "concise":    "简洁型 — 突出数据，去除冗余",
            "detailed":   "详实型 — 充分展开背景与贡献",
            "aggressive": "激进型 — 最大化关键词匹配",
        }[x],
        key="ro_mode",
    )
    run_btn = st.sidebar.button("🚀 开始优化", use_container_width=True)

    # ── 主区域 ──
    if "ro_result" not in st.session_state:
        st.session_state.ro_result = None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始简历")
        if resume_file:
            st.success(f"已上传：{resume_file.name}")
        else:
            st.info("请在左侧上传简历 PDF")
    with col2:
        st.subheader("目标岗位 JD")
        if job_desc.strip():
            st.text_area("JD 预览", job_desc[:400] + ("..." if len(job_desc) > 400 else ""),
                         height=120, disabled=True, key="ro_jd_preview")
        else:
            st.info("请在左侧粘贴 JD")

    st.divider()

    if run_btn:
        if not resume_file:
            st.error("请先上传简历 PDF"); st.stop()
        if not job_desc.strip():
            st.error("请先粘贴目标岗位 JD"); st.stop()

        resume_dir = "./uploaded_resumes"
        os.makedirs(resume_dir, exist_ok=True)
        resume_path = os.path.join(resume_dir, resume_file.name)
        with open(resume_path, "wb") as f:
            f.write(resume_file.getbuffer())

        with st.spinner("正在读取简历..."):
            resume_text = extract_pdf_text(resume_path)
            print("Extracted resume: {resume_text} ", resume_text)

        if not resume_text.strip():
            st.error("无法从 PDF 中提取文字，请确认文件非扫描版。"); st.stop()

        initial_state: ResumeState = {
            "resume_text": resume_text, "job_description": job_desc, "mode": mode,
            "parsed_info": "", "gap_analysis": "", "optimized_resume": "",
            "suggestions": "", "score": 0, "score_reason": "", "retry_count": 0,
        }

        graph = build_graph()
        progress_placeholder = st.empty()

        with st.spinner("Agent 运行中，请稍候..."):
            steps_log = []
            final_state = initial_state
            for step in graph.stream(initial_state):
                node_name = list(step.keys())[0]
                final_state = {**final_state, **step[node_name]}
                label = {
                    "parse":    "✅ Node 1 完成：简历解析 + 差距分析",
                    "optimize": f"✅ Node 2 完成：简历优化（第 {final_state.get('retry_count', 0) + 1} 次）",
                    "review":   f"✅ Node 3 完成：质量审核 → 评分 {final_state.get('score', '?')}",
                }.get(node_name, f"✅ {node_name} 完成")
                steps_log.append(label)
                progress_placeholder.markdown("\n\n".join(steps_log))

        st.session_state.ro_result = final_state

    result = st.session_state.ro_result
    if result:
        score = result.get("score", 0)
        score_color = "#22c55e" if score >= 75 else "#f59e0b" if score >= 60 else "#ef4444"
        st.subheader("📊 优化结果")
        st.markdown(
            f'<span class="score-badge" style="background:{score_color}20;color:{score_color};'
            f'border:1px solid {score_color}40">质量评分：{score} / 100</span>',
            unsafe_allow_html=True,
        )
        if result.get("score_reason"):
            st.caption(f"评审说明：{result['score_reason']}")
        if result.get("retry_count", 0) > 0:
            st.caption(f"经过 {result['retry_count']} 次自动迭代优化")
        st.divider()

        tab1, tab2, tab3 = st.tabs(["📝 优化后简历", "🔍 差距分析", "💡 修改说明"])
        with tab1:
            optimized = result.get("optimized_resume", "")
            st.markdown(optimized)
            st.download_button("⬇️ 下载优化简历（Markdown）",
                               data=optimized.encode("utf-8"),
                               file_name="optimized_resume.md", mime="text/markdown")
        with tab2:
            st.markdown("**结构化提取**")
            st.markdown(result.get("parsed_info", ""))
            st.divider()
            st.markdown("**差距分析**")
            st.markdown(result.get("gap_analysis", ""))
        with tab3:
            st.markdown(result.get("suggestions", ""))


if __name__ == "__main__":
    run()