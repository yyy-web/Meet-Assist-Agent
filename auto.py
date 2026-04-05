# ─────────────────────────────────────────────────────────────────────────────
#  auto_apply.py  ·  自动投递 Agent（LangGraph + Milvus 余弦相似度 + Streamlit）
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import os
import re
from dotenv import load_dotenv
load_dotenv()
import json
import time
import uuid
import hashlib
import logging
from typing import TypedDict, Annotated, Optional
from datetime import datetime

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  常量配置
# ─────────────────────────────────────────────────────────────

MILVUS_HOST        = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT        = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME    = "job_postings"
EMBEDDING_DIM      = 1024
MATCH_THRESHOLD    = 0.72
TOP_K              = 20
MAX_APPLY          = 10

USE_REAL_SCRAPER  = os.getenv("BOSS_USE_REAL_SCRAPER", "0") == "1"
SCRAPER_MAX_PAGES = int(os.getenv("BOSS_MAX_PAGES", "1"))
SCRAPER_HEADLESS  = os.getenv("BOSS_HEADLESS", "1") == "1"
SCRAPER_DETAIL    = os.getenv("BOSS_FETCH_DETAIL", "1") == "1"

APPLY_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apply_log.json")


# ─────────────────────────────────────────────────────────────
#  1.  State 定义
# ─────────────────────────────────────────────────────────────

class JobInfo(TypedDict):
    job_id:       str
    title:        str
    company:      str
    location:     str
    salary:       str
    description:  str
    url:          str
    score:        float
    cover_letter: str
    applied:      bool


class AutoApplyState(TypedDict):
    resume_text:    str
    target_title:   str
    target_city:    str
    threshold:      float
    max_apply:      int
    resume_vector:  list[float]
    raw_jobs:       list[JobInfo]
    matched_jobs:   list[JobInfo]
    filtered_jobs:  list[JobInfo]
    confirmed_jobs: list[JobInfo]
    applied_jobs:   list[JobInfo]
    apply_stats:    dict
    error:          str
    _resume_name:   str
    _use_real:      bool


# ─────────────────────────────────────────────────────────────
#  2.  工具函数
# ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts  = [p.extract_text() for p in reader.pages if p.extract_text()]
        text   = "\n\n".join(parts).strip()
        if text:
            return text, "pypdf"
    except Exception:
        pass
    return "", "error"


def clean_resume_text(raw_text: str) -> str:
    text = re.sub(r"(?m)^[\s\-–—]*\d+[\s\-–—]*$", "", raw_text)
    text = re.sub(r"(?im)^page\s*\d+.*$", "", text)
    text = re.sub(r"(?m)^第\s*\d+\s*页.*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def save_apply_log(
    applied_jobs: list[JobInfo],
    target_title: str,
    target_city:  str,
    resume_name:  str = "（未知）",
    apply_stats:  dict = None,
) -> str:
    session = {
        "session_id":   datetime.now().strftime("%Y%m%d%H%M%S"),
        "time":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date":         datetime.now().strftime("%Y-%m-%d"),
        "target_title": target_title,
        "target_city":  target_city,
        "resume_name":  resume_name,
        "stats":        apply_stats or {},
        "jobs": [
            {
                "title":        j["title"],
                "company":      j["company"],
                "salary":       j["salary"],
                "location":     j["location"],
                "score":        j["score"],
                "url":          j["url"],
                "applied":      j["applied"],
                "status":       j.get("apply_status", "applied" if j["applied"] else "pending"),
                "cover_letter": j.get("cover_letter", ""),
            }
            for j in applied_jobs
        ],
    }

    existing = []
    if os.path.exists(APPLY_LOG_PATH):
        try:
            with open(APPLY_LOG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.append(session)
    with open(APPLY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    return APPLY_LOG_PATH


def load_apply_log() -> list[dict]:
    if not os.path.exists(APPLY_LOG_PATH):
        return []
    try:
        with open(APPLY_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(reversed(data))
    except Exception:
        return []


def _make_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model="qwen-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=temperature,
    )


def _get_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )


def _connect_milvus():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception:
        pass


def _get_collection_dim(col: Collection) -> int:
    for field in col.schema.fields:
        if field.name == "embedding":
            return field.params.get("dim", -1)
    return -1


def _ensure_collection() -> Collection:
    _connect_milvus()
    if utility.has_collection(COLLECTION_NAME):
        col = Collection(COLLECTION_NAME)
        if _get_collection_dim(col) != EMBEDDING_DIM:
            col.drop()
        else:
            col.load()
            return col

    fields = [
        FieldSchema(name="id",          dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="title",       dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="company",     dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="location",    dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="salary",      dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="url",         dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding",   dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description=f"BOSS 直聘岗位向量库 dim={EMBEDDING_DIM}")
    col = Collection(COLLECTION_NAME, schema)
    col.create_index("embedding", {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    col.load()
    return col


def _normalize(vector: list[float]) -> list[float]:
    import math
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def _upsert_jobs_to_milvus(col: Collection, jobs: list[JobInfo], embedder):
    descs     = [j["description"][:1000] for j in jobs]
    norm_vecs = [_normalize(v) for v in embedder.embed_documents(descs)]
    col.insert([
        [j["job_id"]             for j in jobs],
        [j["title"]              for j in jobs],
        [j["company"]            for j in jobs],
        [j["location"]           for j in jobs],
        [j["salary"]             for j in jobs],
        [j["description"][:4000] for j in jobs],
        [j["url"]                for j in jobs],
        norm_vecs,
    ])
    col.flush()


def _mock_boss_jobs(target_title: str, target_city: str) -> list[JobInfo]:
    base_jobs = [
        {"title": f"高级{target_title}",      "company": "字节跳动",    "salary": "25-45K",
         "desc": f"负责{target_title}相关业务开发，3年以上经验，熟悉Python/Java，了解AI/大模型优先。"},
        {"title": f"{target_title}（AI方向）", "company": "阿里云",      "salary": "20-40K",
         "desc": f"参与{target_title}工作，结合LLM和Agent技术，需熟悉LangChain/LangGraph框架。"},
        {"title": f"资深{target_title}",      "company": "腾讯",        "salary": "30-50K",
         "desc": f"主导{target_title}架构设计，要求5年以上经验，熟悉微服务、分布式系统。"},
        {"title": target_title,               "company": "美团",        "salary": "18-30K",
         "desc": f"加入我们的{target_title}团队，负责日常开发迭代，要求掌握主流技术栈。"},
        {"title": f"初级{target_title}",      "company": "滴滴出行",    "salary": "12-20K",
         "desc": f"适合应届生或1-2年{target_title}经验，参与产品功能开发和测试。"},
        {"title": "财务分析师",               "company": "某金融公司",   "salary": "15-25K",
         "desc": "负责财务报表分析、预算管理、成本核算，要求CPA证书，精通Excel财务建模。"},
        {"title": "市场营销专员",             "company": "某消费品公司", "salary": "10-18K",
         "desc": "负责品牌推广、社交媒体运营、线下活动策划，要求有互联网营销经验。"},
    ]
    jobs: list[JobInfo] = []
    for j in base_jobs:
        job_id = hashlib.md5(f"{j['title']}{j['company']}".encode()).hexdigest()[:16]
        jobs.append(JobInfo(
            job_id=job_id, title=j["title"], company=j["company"],
            location=target_city, salary=j["salary"], description=j["desc"],
            url=f"https://www.zhipin.com/job_detail/{job_id}.html",
            score=0.0, cover_letter="", applied=False,
        ))
    return jobs


# ─────────────────────────────────────────────────────────────
#  3.  LangGraph 节点
# ─────────────────────────────────────────────────────────────

def embed_resume_node(state: AutoApplyState) -> AutoApplyState:
    try:
        embedder = _get_embeddings()
        raw_vec  = embedder.embed_query(state["resume_text"][:2000])
        return {**state, "resume_vector": _normalize(raw_vec), "error": ""}
    except Exception as e:
        return {**state, "resume_vector": [], "error": f"向量化失败：{e}"}


def fetch_jobs_node(state: AutoApplyState) -> AutoApplyState:
    """
    不在此节点内调用爬虫。
    真实模式：直接从 session_state 读取预爬取数据。
    模拟模式：生成 mock 数据。
    """
    if state.get("error"):
        return state
    try:
        if state.get("_use_real"):
            prefetched = st.session_state.get("prefetched_jobs", [])
            if not prefetched:
                return {**state, "raw_jobs": [], "error": "未找到预爬取数据，请重新点击搜索"}
            jobs = [JobInfo(**d) for d in prefetched]
        else:
            jobs = _mock_boss_jobs(state["target_title"], state["target_city"])

        if not jobs:
            return {**state, "raw_jobs": [], "error": "未获取到任何岗位"}

        embedder = _get_embeddings()
        col      = _ensure_collection()
        _upsert_jobs_to_milvus(col, jobs, embedder)
        return {**state, "raw_jobs": jobs, "error": ""}
    except Exception as e:
        return {**state, "raw_jobs": [], "error": f"岗位获取失败：{e}"}


def match_node(state: AutoApplyState) -> AutoApplyState:
    if state.get("error") or not state.get("resume_vector"):
        return state
    try:
        col     = _ensure_collection()
        results = col.search(
            data=[state["resume_vector"]],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=TOP_K,
            output_fields=["id", "title", "company", "location", "salary", "description", "url"],
        )
        matched = []
        for hit in results[0]:
            matched.append(JobInfo(
                job_id=hit.entity.get("id"),       title=hit.entity.get("title"),
                company=hit.entity.get("company"), location=hit.entity.get("location"),
                salary=hit.entity.get("salary"),   description=hit.entity.get("description"),
                url=hit.entity.get("url"),          score=round(float(hit.score), 4),
                cover_letter="", applied=False,
            ))
        matched.sort(key=lambda x: x["score"], reverse=True)
        return {**state, "matched_jobs": matched, "error": ""}
    except Exception as e:
        return {**state, "matched_jobs": [], "error": f"相似度匹配失败：{e}"}


def filter_node(state: AutoApplyState) -> AutoApplyState:
    if state.get("error"):
        return state
    threshold = state.get("threshold", MATCH_THRESHOLD)
    max_apply = state.get("max_apply", MAX_APPLY)
    filtered  = [j for j in state.get("matched_jobs", []) if j["score"] >= threshold][:max_apply]
    return {**state, "filtered_jobs": filtered}


def generate_letter_node(state: AutoApplyState) -> AutoApplyState:
    if state.get("error") or not state.get("confirmed_jobs"):
        return state

    llm          = _make_llm(temperature=0.75)
    updated_jobs = []

    for job in state["confirmed_jobs"]:
        system = SystemMessage(content=(
            "你是一位专业的求职信撰写顾问。"
            "根据候选人简历和目标岗位，生成一封简洁有力的求职信（50-100字），"
            "重点突出技能匹配点，语气专业诚恳，避免空话套话。"
        ))
        user = HumanMessage(content=f"""
## 候选人简历（摘要）
{state["resume_text"][:800]}

## 目标岗位
公司：{job["company"]}
职位：{job["title"]}
岗位要求：{job["description"][:500]}

请生成求职信：
""")
        try:
            resp         = llm.invoke([system, user])
            cover_letter = resp.content.strip()
        except Exception as e:
            cover_letter = f"（求职信生成失败：{e}）"

        updated_jobs.append({**job, "cover_letter": cover_letter})

    return {**state, "confirmed_jobs": updated_jobs}


def apply_node(state: AutoApplyState) -> AutoApplyState:
    if state.get("error") or not state.get("confirmed_jobs"):
        return state

    applied = [{**job, "applied": True, "apply_status": "applied"} for job in state["confirmed_jobs"]]
    stats   = {"success": len(applied), "failed": 0, "already": 0, "skipped": 0}

    try:
        save_apply_log(
            applied_jobs=applied,
            target_title=state.get("target_title", ""),
            target_city=state.get("target_city", ""),
            resume_name=state.get("_resume_name", "（未知）"),
            apply_stats=stats,
        )
    except Exception:
        pass

    return {**state, "applied_jobs": applied, "apply_stats": stats}


def real_apply_node(state: AutoApplyState) -> AutoApplyState:
    if state.get("error") or not state.get("confirmed_jobs"):
        return state

    try:
        from boss_scraper import BossScraper, BossJob

        scraper   = BossScraper(headless=False)
        logged_in = scraper.login_check()
        if not logged_in:
            return {**state, "error": "BOSS 直聘登录失败，无法投递"}

        boss_jobs: list[BossJob] = []
        for j in state["confirmed_jobs"]:
            bj = BossJob(
                job_id=j["job_id"], title=j["title"],
                company=j["company"], location=j["location"],
                salary=j["salary"],   description=j["description"],
                url=j["url"],
            )
            bj.cover_letter = j.get("cover_letter", "")  # type: ignore
            boss_jobs.append(bj)

        result_jobs = scraper.apply_jobs(
            boss_jobs,
            cover_letter=None, #求职信
            daily_limit=state.get("max_apply", MAX_APPLY),
        )
        scraper.quit()

        applied = []
        stats   = {"success": 0, "failed": 0, "already": 0, "skipped": 0}
        for orig, res in zip(state["confirmed_jobs"], result_jobs):
            status = res.apply_status
            row    = {**orig, "applied": status in ("applied", "already_applied"), "apply_status": status}
            applied.append(row)
            if status == "applied":
                stats["success"] += 1
            elif status == "already_applied":
                stats["already"] += 1
            elif status == "skipped_daily_limit":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

        try:
            save_apply_log(
                applied_jobs=[j for j in applied if j["applied"]],
                target_title=state.get("target_title", ""),
                target_city=state.get("target_city", ""),
                resume_name=state.get("_resume_name", "（未知）"),
                apply_stats=stats,
            )
        except Exception:
            pass

        return {**state, "applied_jobs": applied, "apply_stats": stats}

    except ImportError:
        return {**state, "error": "boss_scraper.py 未找到"}
    except Exception as e:
        return {**state, "error": f"真实投递失败：{e}"}


# ─────────────────────────────────────────────────────────────
#  4.  构建 LangGraph
# ─────────────────────────────────────────────────────────────

def build_search_graph():
    graph = StateGraph(AutoApplyState)
    graph.add_node("embed_resume", embed_resume_node)
    graph.add_node("fetch_jobs",   fetch_jobs_node)
    graph.add_node("match",        match_node)
    graph.add_node("filter",       filter_node)
    graph.set_entry_point("embed_resume")
    graph.add_edge("embed_resume", "fetch_jobs")
    graph.add_edge("fetch_jobs",   "match")
    graph.add_edge("match",        "filter")
    graph.add_edge("filter",       END)
    return graph.compile()


def build_apply_graph(use_real: bool):
    graph = StateGraph(AutoApplyState)
    graph.add_node("generate_letter", generate_letter_node)
    graph.add_node("apply", real_apply_node if use_real else apply_node)
    graph.set_entry_point("generate_letter")
    graph.add_edge("generate_letter", "apply")
    graph.add_edge("apply",           END)
    return graph.compile()


# ─────────────────────────────────────────────────────────────
#  5.  Streamlit UI
# ─────────────────────────────────────────────────────────────

CSS = """
<style>
.block-container { padding: 2rem !important; }
.home-btn-container {
    position: fixed; top: 0.8rem; right: 1rem; z-index: 9999;
}
.home-btn-container a {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #FFFFFF; border: 1px solid rgba(0,0,0,0.10);
    border-radius: 8px; padding: 0.4rem 0.9rem;
    color: #64748b !important; font-size: 0.82rem;
    text-decoration: none !important;
}
.mode-card {
    border: 2px solid transparent; border-radius: 12px;
    padding: 1rem 1.2rem; cursor: pointer;
    transition: all 0.2s ease; margin-bottom: 0.5rem;
}
.mode-card.mock  { background: #f0fdf4; border-color: #86efac; }
.mode-card.real  { background: #eff6ff; border-color: #93c5fd; }
.job-card {
    border: 1px solid rgba(99,102,241,0.2); border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    background: rgba(99,102,241,0.03);
}
.job-card.selected { border-color: #6366f1; background: rgba(99,102,241,0.07); }
.progress-row {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.5rem 0.8rem; border-radius: 8px;
    margin-bottom: 0.4rem; font-size: 0.9rem;
}
.progress-row.success { background: #f0fdf4; }
.progress-row.failed  { background: #fef2f2; }
.progress-row.already { background: #fffbeb; }
.progress-row.pending { background: #f8fafc; }
.stat-box {
    text-align: center; border-radius: 10px; padding: 1rem;
}
.stat-box .num { font-size: 2rem; font-weight: 700; }
.stat-box .lab { font-size: 0.8rem; color: #64748b; }
</style>
<div class="home-btn-container">
    <a href="?page=home">🏠 返回首页</a>
</div>
"""


def _score_bar(score: float) -> str:
    filled = int(score * 20)
    bar    = "█" * filled + "░" * (20 - filled)
    color  = "green" if score >= 0.8 else "orange" if score >= 0.72 else "red"
    return f":{color}[{bar}] {score:.2%}"


def _status_icon(status: str) -> str:
    return {
        "applied":             "✅ 投递成功",
        "already_applied":     "⚠️ 已投递过",
        "skipped_daily_limit": "⏭️ 超出上限",
        "failed_no_btn":       "❌ 未找到按钮",
        "failed_no_url":       "❌ 无URL",
        "failed_exception":    "❌ 异常",
        "pending":             "⏳ 待处理",
    }.get(status, f"❓ {status}")


# ─────────────────────────────────────────────────────────────
#  阶段零：爬取状态机
#
#  ★ 设计原则：
#    - scrape_sub = "login_pending"  → 显示引导页，等待用户点击「启动」按钮
#    - scrape_sub = "login_waiting"  → 浏览器已开，等待用户扫码后点击「已登录」
#    - scrape_sub = "scraping_ready" → 用户确认登录，显示「开始爬取」按钮
#    - scrape_sub = "scraping_running"→ 用户点击后，在同一次 script run 内
#                                       同步完成登录验证 + search_jobs，
#                                       不再有任何 rerun，彻底避免实例丢失
#
#  ★ 关键：_scraper_instance 不跨 rerun 传递（跨线程会丢失）。
#    每个需要浏览器的子阶段都在当前 script run 内用完即丢或存 cookie 后重建。
# ─────────────────────────────────────────────────────────────

def _run_scrape_if_needed(
    use_real: bool,
    target_title: str,
    target_city: str,
) -> str:
    """
    返回值：
      "done"   — 爬取完成，可进入下一阶段
      "wait"   — 等待用户点击按钮，当前 run 结束
      "failed" — 失败，终止流程
    """
    # ── 模拟模式：直接跳过 ──────────────────────────────────
    if not use_real:
        st.session_state.prefetched_jobs = []
        st.session_state.scrape_done     = True
        return "done"

    # ── 已完成爬取：跳过 ────────────────────────────────────
    if st.session_state.get("scrape_done") and st.session_state.get("prefetched_jobs") is not None:
        return "done"

    scrape_sub = st.session_state.get("scrape_sub", "login_pending")

    # ══════════════════════════════════════════════════════════
    # 子阶段一：引导页
    #   显示说明文字 + 「启动浏览器」按钮
    #   用户点击后：启动浏览器，检测 Cookie，决定跳转到哪个下一步
    # ══════════════════════════════════════════════════════════
    if scrape_sub == "login_pending":
        st.info("🔐 准备爬取 BOSS 直聘", icon="🖥️")
        st.markdown("""
        点击「启动浏览器」后将自动：
        - 若本地有有效 Cookie → 自动登录，直接开始爬取
        - 若 Cookie 失效或不存在 → 打开登录页，请在浏览器中扫码

        ⚠️ 爬取过程中**请勿关闭**弹出的浏览器窗口
        """)

        if st.button("🚀 启动浏览器", type="primary", key="btn_open_browser"):
            # ── 在本次 script run 内同步执行，不 rerun ─────
            try:
                from boss_scraper import BossScraper, COOKIE_PATH

                with st.status("⚙️ 启动浏览器...", expanded=True) as launch_status:
                    scraper = BossScraper(headless=False)
                    scraper.page.get("https://www.zhipin.com")
                    time.sleep(1.5)

                    # 尝试 Cookie 登录
                    cookie_ok = False
                    if COOKIE_PATH.exists():
                        try:
                            cookies = json.loads(COOKIE_PATH.read_text(encoding="utf-8"))
                            for c in cookies:
                                scraper.page.set.cookies(c)
                            scraper.page.refresh()
                            time.sleep(2)
                            if scraper._is_logged_in():
                                cookie_ok = True
                        except Exception:
                            pass

                    if cookie_ok:
                        # Cookie 有效 → 直接爬取，不需要扫码
                        launch_status.update(label="✅ Cookie 自动登录成功，开始爬取...", state="running")
                        result = _do_scrape_with_scraper(scraper, target_title, target_city, launch_status)
                        scraper.quit()
                        if result == "done":
                            return "done"
                        else:
                            st.session_state.scrape_sub = "login_pending"
                            return "failed"
                    else:
                        # Cookie 无效 → 打开登录页，等用户扫码
                        scraper.page.get(scraper.LOGIN_URL)
                        launch_status.update(label="✅ 浏览器已打开登录页，请扫码", state="complete")
                        # ★ 把 scraper 序列化为可跨 rerun 的数据：只保存 Cookie 路径标记
                        #   实际实例不存 session_state，下一阶段重建
                        st.session_state.scrape_sub = "login_waiting"
                        # 延迟一下让 status 显示出来
                        time.sleep(0.5)

            except Exception as e:
                st.error(f"启动浏览器失败：{e}")
                return "failed"

            st.rerun()

        return "wait"

    # ══════════════════════════════════════════════════════════
    # 子阶段二：等待扫码
    #   用户在浏览器里扫码登录后，点击「我已登录」
    #   → 重建 scraper 实例，用 Cookie 验证登录，验证通过则开始爬取
    # ══════════════════════════════════════════════════════════
    if scrape_sub == "login_waiting":
        st.success("✅ 浏览器已打开 BOSS 直聘登录页")
        st.info("""
        **请完成以下步骤：**
        1. 在弹出的 Chrome 窗口中，用手机扫描二维码
        2. 在 BOSS 直聘 APP 中确认登录
        3. 看到浏览器跳转到主页后，点击下方「我已登录，开始爬取」
        """, icon="📱")
        st.warning("提示：登录后浏览器窗口会保持打开，爬取完毕后自动关闭", icon="ℹ️")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ 我已登录，开始爬取", type="primary", key="btn_login_done"):
                # ── 重建 scraper，加载 Cookie，验证登录，同步爬取 ──
                try:
                    from boss_scraper import BossScraper, COOKIE_PATH

                    with st.status("🔐 验证登录状态...", expanded=True) as verify_status:
                        scraper = BossScraper(headless=False)
                        scraper.page.get("https://www.zhipin.com")
                        time.sleep(1.5)

                        # 加载 Cookie
                        loaded = False
                        if COOKIE_PATH.exists():
                            try:
                                cookies = json.loads(COOKIE_PATH.read_text(encoding="utf-8"))
                                for c in cookies:
                                    scraper.page.set.cookies(c)
                                scraper.page.refresh()
                                time.sleep(2)
                                loaded = scraper._is_logged_in()
                            except Exception:
                                pass

                        if not loaded:
                            # Cookie 还没保存（用户可能刚扫码），等一下再检测
                            verify_status.update(label="⏳ Cookie 尚未就绪，等待 5 秒后重试...")
                            time.sleep(5)
                            scraper.page.refresh()
                            time.sleep(2)
                            loaded = scraper._is_logged_in()

                        if not loaded:
                            verify_status.update(label="❌ 未检测到登录状态", state="error")
                            scraper.quit()
                            st.error("未检测到登录状态，请确认已在浏览器中完成扫码后再试")
                            return "wait"

                        # 保存 Cookie
                        try:
                            scraper._save_cookies()
                        except Exception:
                            pass

                        verify_status.update(label="✅ 登录成功，开始爬取...", state="running")
                        result = _do_scrape_with_scraper(scraper, target_title, target_city, verify_status)
                        scraper.quit()

                    if result == "done":
                        return "done"
                    else:
                        st.session_state.scrape_sub = "login_pending"
                        return "failed"

                except Exception as e:
                    st.error(f"操作失败：{e}")
                    return "failed"

        with col2:
            if st.button("🔄 重新开始", key="btn_restart"):
                st.session_state.scrape_sub = "login_pending"
                st.rerun()

        return "wait"

    return "wait"


def _do_scrape_with_scraper(scraper, target_title: str, target_city: str, status_ctx) -> str:
    """
    在已登录的 scraper 实例上执行爬取，直接操作传入的 st.status 上下文。
    返回 "done" 或 "failed"。
    ★ 此函数在同一次 script run 内被调用，不涉及任何 rerun。
    """
    try:
        status_ctx.update(label=f"🔍 搜索岗位：{target_title}，共 {SCRAPER_MAX_PAGES} 页...")
        st.write(f"📄 正在爬取第 1 页...")

        jobs = scraper.search_jobs(
            keyword=target_title,
            city=target_city,
            max_pages=SCRAPER_MAX_PAGES,
        )

        if SCRAPER_DETAIL and jobs:
            st.write(f"📋 补全 {len(jobs)} 个岗位详情...")
            status_ctx.update(label=f"📋 补全岗位详情（共 {len(jobs)} 个）...")
            jobs = scraper.fetch_all_details(jobs)

        raw_dicts = [{
            "job_id":       j.job_id,
            "title":        j.title,
            "company":      j.company,
            "location":     j.location,
            "salary":       j.salary,
            "description":  j.description or f"{j.title}，{j.company}，{'，'.join(j.tags)}",
            "url":          j.url,
            "score":        0.0,
            "cover_letter": "",
            "applied":      False,
        } for j in jobs]

        if not raw_dicts:
            status_ctx.update(label="⚠️ 未获取到任何岗位", state="error")
            st.warning("未获取到任何岗位，请检查关键词或降低页数")
            return "failed"

        st.session_state.prefetched_jobs = raw_dicts
        st.session_state.scrape_done     = True
        st.session_state.scrape_sub      = "done"
        status_ctx.update(
            label=f"✅ 爬取完成，共获取 {len(raw_dicts)} 个岗位",
            state="complete",
        )
        st.write(f"✅ 共获取 {len(raw_dicts)} 个岗位，即将进入匹配分析...")
        time.sleep(1)
        return "done"

    except Exception as e:
        status_ctx.update(label=f"❌ 爬取失败：{e}", state="error")
        st.error(f"爬取异常详情：{e}")
        return "failed"


# ─────────────────────────────────────────────────────────────
#  阶段一：搜索匹配 UI
# ─────────────────────────────────────────────────────────────

def _render_search_phase(
    resume_text: str, resume_name: str,
    target_title: str, target_city: str,
    threshold: float, max_apply: int,
    use_real: bool,
):
    st.subheader("🔍 Step 1-4：搜索 & 匹配")

    initial_state = AutoApplyState(
        resume_text=resume_text, target_title=target_title,
        target_city=target_city, threshold=threshold, max_apply=max_apply,
        resume_vector=[], raw_jobs=[], matched_jobs=[], filtered_jobs=[],
        confirmed_jobs=[], applied_jobs=[], apply_stats={},
        error="", _resume_name=resume_name, _use_real=use_real,
    )

    graph       = build_search_graph()
    final_state = initial_state
    steps_log   = []
    log_ph      = st.empty()

    NODE_LABELS = {
        "embed_resume": "✅ Step 1 — 简历向量化",
        "fetch_jobs":   f"✅ Step 2 — {'读取预爬取数据' if use_real else '模拟数据'} + 写入 Milvus",
        "match":        "✅ Step 3 — 余弦相似度检索",
        "filter":       None,
    }

    with st.spinner("正在分析匹配..."):
        for step in graph.stream(initial_state):
            node_name   = list(step.keys())[0]
            final_state = {**final_state, **step[node_name]}
            if final_state.get("error"):
                steps_log.append(f"❌ {node_name}：{final_state['error']}")
            else:
                if node_name == "filter":
                    label = (
                        f"✅ Step 4 — 阈值过滤（≥{threshold:.0%}），"
                        f"剩余 **{len(final_state.get('filtered_jobs', []))}** 个岗位"
                    )
                else:
                    label = NODE_LABELS.get(node_name, node_name)
                steps_log.append(label)
            log_ph.markdown("\n\n".join(steps_log))

    if final_state.get("error"):
        st.error(f"运行出错：{final_state['error']}")
        return None

    return final_state


# ─────────────────────────────────────────────────────────────
#  阶段二：预览确认 UI
# ─────────────────────────────────────────────────────────────

def _render_confirm_phase(search_state: dict) -> list[JobInfo] | None:
    filtered = search_state.get("filtered_jobs", [])
    matched  = search_state.get("matched_jobs", [])

    if not filtered:
        st.warning("没有达到相似度阈值的岗位，建议降低阈值或更换关键词。")
        _render_all_matched(matched, search_state.get("threshold", MATCH_THRESHOLD))
        return None

    st.subheader(f"📋 Step 5：投递前预览确认（共 {len(filtered)} 个岗位）")
    st.info("请勾选要投递的岗位，取消勾选可跳过。确认后点击「开始投递」。")

    selected_ids = []
    for i, job in enumerate(filtered):
        col_check, col_info = st.columns([0.08, 0.92])
        with col_check:
            # ★ 修复：label 不能为空，用 label_visibility 隐藏
            checked = st.checkbox(
                label=f"选择岗位 {i + 1}",
                value=True,
                key=f"confirm_{job['job_id']}_{i}",
                label_visibility="hidden",
            )
        with col_info:
            card_cls = "job-card selected" if checked else "job-card"
            st.markdown(f"""
            <div class="{card_cls}">
                <b>{job['title']}</b> · {job['company']} ·
                <code>{job['salary']}</code> · {job['location']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"余弦相似度：{_score_bar(job['score'])}")
        if checked:
            selected_ids.append(job["job_id"])

    st.caption(f"已选 **{len(selected_ids)}** / {len(filtered)} 个岗位")

    with st.expander(f"📊 查看全部匹配岗位（召回 {len(matched)} 个）"):
        _render_all_matched(matched, search_state.get("threshold", MATCH_THRESHOLD))

    col_a, col_b = st.columns([1, 3])
    with col_a:
        confirm_btn = st.button(
            f"🚀 确认投递 {len(selected_ids)} 个",
            type="primary", width="stretch",
            disabled=len(selected_ids) == 0,
        )
    with col_b:
        st.caption("点击后将逐一生成求职信并自动投递")

    if confirm_btn:
        return [j for j in filtered if j["job_id"] in selected_ids]

    return None


def _render_all_matched(matched: list[JobInfo], threshold: float):
    for job in matched:
        passed = job["score"] >= threshold
        icon   = "✅" if passed else "🔸"
        with st.expander(f"{icon} {job['title']} · {job['company']} · {job['score']:.2%}"):
            st.write(f"**薪资：** {job['salary']}　**地点：** {job['location']}")
            st.info(job["description"])
            if not passed:
                st.caption(f"⚠️ 低于阈值 {threshold:.0%}，已过滤")


# ─────────────────────────────────────────────────────────────
#  阶段三：投递执行 UI
# ─────────────────────────────────────────────────────────────

def _render_apply_phase(
    search_state: dict,
    confirmed_jobs: list[JobInfo],
    use_real: bool,
):
    st.subheader("📨 Step 6-7：生成求职信 & 投递")

    if use_real:
        st.info(
            "⚠️ **安全验证提示**：若 BOSS 直聘检测到异常访问，会跳转安全验证页。"
            "爬虫将自动等待验证通过（通常 5-30 秒），无需手动操作。",
            icon="🛡️",
        )

    apply_state = {
        **search_state,
        "confirmed_jobs": confirmed_jobs,
        "applied_jobs":   [],
        "apply_stats":    {},
    }

    graph       = build_apply_graph(use_real)
    final_state = apply_state
    progress_ph = st.empty()

    with progress_ph.container():
        for job in confirmed_jobs:
            st.markdown(f"""
            <div class="progress-row pending">
                ⏳ <b>{job['title']}</b> · {job['company']} · 待投递
            </div>
            """, unsafe_allow_html=True)

    step_log = []
    log_ph   = st.empty()

    with st.spinner("投递中..."):
        for step in graph.stream(apply_state):
            node_name   = list(step.keys())[0]
            final_state = {**final_state, **step[node_name]}
            if node_name == "generate_letter":
                step_log.append("✅ 求职信生成完毕")
            elif node_name == "apply":
                step_log.append("✅ 投递执行完毕")
            log_ph.markdown("\n\n".join(step_log))

    if final_state.get("error"):
        st.error(f"投递出错：{final_state['error']}")
        return final_state

    applied_jobs = final_state.get("applied_jobs", [])
    with progress_ph.container():
        st.markdown("**投递进度：**")
        for job in applied_jobs:
            status = job.get("apply_status", "applied" if job["applied"] else "failed")
            css    = "success" if job["applied"] else ("already" if "already" in status else "failed")
            label  = _status_icon(status)
            st.markdown(f"""
            <div class="progress-row {css}">
                {label} &nbsp;
                <b>{job['title']}</b> · {job['company']} · <code>{job['salary']}</code>
            </div>
            """, unsafe_allow_html=True)

    stats = final_state.get("apply_stats", {})
    _render_stats_panel(stats, applied_jobs)

    return final_state


def _render_stats_panel(stats: dict, applied_jobs: list[JobInfo]):
    st.divider()
    st.subheader("📊 投递结果统计")

    s = stats.get("success", 0)
    a = stats.get("already", 0)
    f = stats.get("failed",  0)
    k = stats.get("skipped", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="stat-box" style="background:#f0fdf4"><div class="num" style="color:#16a34a">{s}</div><div class="lab">✅ 投递成功</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-box" style="background:#fffbeb"><div class="num" style="color:#d97706">{a}</div><div class="lab">⚠️ 已投递过</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-box" style="background:#fef2f2"><div class="num" style="color:#dc2626">{f}</div><div class="lab">❌ 投递失败</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="stat-box" style="background:#f8fafc"><div class="num" style="color:#64748b">{k}</div><div class="lab">⏭️ 超出上限</div></div>', unsafe_allow_html=True)

    if applied_jobs:
        st.divider()
        st.subheader("📝 求职信详情")
        for job in applied_jobs:
            if job.get("cover_letter"):
                with st.expander(f"{job['title']} — {job['company']}"):
                    st.markdown(job["cover_letter"])
                    st.code(job["cover_letter"], language=None)
                    _, cb = st.columns([3, 1])
                    with cb:
                        st.link_button("查看岗位", job["url"])


# ─────────────────────────────────────────────────────────────
#  投递历史 UI
# ─────────────────────────────────────────────────────────────

def _render_history_tab():
    sessions = load_apply_log()

    if not sessions:
        st.info("📭 暂无投递记录，完成第一次投递后自动保存。")
        return

    st.caption(f"共 **{len(sessions)}** 次投递记录（最新在前）")

    all_dates  = sorted({s["date"] for s in sessions}, reverse=True)
    all_titles = sorted({s["target_title"] for s in sessions})
    all_cities = sorted({s["target_city"] for s in sessions})

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_date  = st.selectbox("📅 日期", ["全部"] + all_dates,  key="hist_date")
    with fc2:
        sel_title = st.selectbox("💼 岗位", ["全部"] + all_titles, key="hist_title")
    with fc3:
        sel_city  = st.selectbox("📍 城市", ["全部"] + all_cities, key="hist_city")

    filtered_sessions = [
        s for s in sessions
        if (sel_date  == "全部" or s["date"]          == sel_date)
        and (sel_title == "全部" or s["target_title"] == sel_title)
        and (sel_city  == "全部" or s["target_city"]  == sel_city)
    ]

    st.caption(f"筛选结果：**{len(filtered_sessions)}** 次投递")
    st.divider()

    for sess in filtered_sessions:
        stats     = sess.get("stats", {})
        jobs      = sess.get("jobs", [])
        success   = stats.get("success", sum(1 for j in jobs if j.get("applied")))
        total     = len(jobs)
        avg_score = (sum(j["score"] for j in jobs) / total) if total else 0

        with st.expander(
            f"📅 {sess['time']}  |  {sess['target_title']} · {sess['target_city']}"
            f"  |  ✅ {success}/{total} 已投递  |  均分 {avg_score:.1%}",
            expanded=False,
        ):
            col_s = st.columns(4)
            col_s[0].metric("投递成功", stats.get("success", "—"))
            col_s[1].metric("已投递过", stats.get("already", "—"))
            col_s[2].metric("投递失败", stats.get("failed",  "—"))
            col_s[3].metric("简历来源", sess.get("resume_name", "—"))
            st.divider()

            import pandas as pd
            df = pd.DataFrame([
                {
                    "职位":   j["title"],
                    "公司":   j["company"],
                    "薪资":   j["salary"],
                    "地点":   j["location"],
                    "匹配度": f"{j['score']:.2%}",
                    "状态":   _status_icon(j.get("status", "applied" if j.get("applied") else "pending")),
                    "链接":   j["url"],
                }
                for j in jobs
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)  # dataframe 本身不受此警告影响

    st.divider()
    download_data = json.dumps(sessions, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label="⬇️ 下载完整日志 (.json)",
        data=download_data,
        file_name=f"apply_log_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
    )


# ─────────────────────────────────────────────────────────────
#  主入口
# ─────────────────────────────────────────────────────────────

def run():
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🚀 自动投递 Agent")
    st.caption("基于 Milvus 余弦相似度语义匹配，自动筛选 BOSS 直聘岗位并生成定制求职信")

    # ── Session State 初始化 ─────────────────────────────────
    defaults = {
        "phase":           "search",
        "search_state":    None,
        "apply_result":    None,
        "confirmed_jobs":  None,
        "prefetched_jobs": None,
        "scrape_done":     False,
        "scrape_sub":      "login_pending",
        "search_params":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ════════════════════════════════════════════════════════
    #  Sidebar
    # ════════════════════════════════════════════════════════
    st.sidebar.header("⚙️ 投递配置")

    target_title = st.sidebar.text_input("目标岗位", value="Python 后端工程师", key="aa_title")
    target_city  = st.sidebar.selectbox(
        "目标城市",
        ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"],
        key="aa_city",
    )
    threshold = st.sidebar.slider(
        "相似度阈值", min_value=0.4, max_value=0.95, value=MATCH_THRESHOLD, step=0.01,
        help="余弦相似度 ≥ 此值才进入投递列表", key="aa_threshold",
    )
    max_apply = st.sidebar.slider("最多投递数量", 1, 20, 5, key="aa_max")

    st.sidebar.divider()

    st.sidebar.subheader("🕷️ 数据来源")
    mode = st.sidebar.radio(
        "选择模式",
        options=["mock", "real"],
        format_func=lambda x: "🧪 模拟数据（调试）" if x == "mock" else "🌐 真实爬取 BOSS 直聘",
        key="aa_mode",
    )
    use_real = (mode == "real")

    if use_real:
        st.sidebar.markdown("""
        <div class="mode-card real">
            🌐 <b>真实模式</b><br>
            <small>首次运行需在弹出浏览器中扫码登录，Cookie 保存后后续自动登录</small>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.caption(
            f"爬取 {SCRAPER_MAX_PAGES} 页 · 补全详情：{'是' if SCRAPER_DETAIL else '否'}"
        )
    else:
        st.sidebar.markdown("""
        <div class="mode-card mock">
            🧪 <b>模拟模式</b><br>
            <small>使用内置样例数据，无需登录，适合本地调试</small>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.divider()

    st.sidebar.subheader("📄 简历上传")
    uploaded_pdf = st.sidebar.file_uploader(
        "上传 PDF 格式简历", type=["pdf"], key="aa_resume_pdf",
    )

    resume_text = ""
    resume_name = "（未知）"

    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.read()
        with st.sidebar.status("📖 正在解析 PDF...", expanded=False) as pdf_status:
            extracted, method = extract_text_from_pdf(pdf_bytes)
            cleaned           = clean_resume_text(extracted)

        if cleaned:
            pdf_status.update(label=f"✅ 解析成功（{method}）", state="complete")
            st.sidebar.caption(f"📊 提取字符数：{len(cleaned):,}")
            with st.expander("🔍 简历解析预览（可手动修正）", expanded=False):
                st.caption(f"引擎：`{method}` · 文件：`{uploaded_pdf.name}` · 字符：`{len(cleaned):,}`")
                resume_text = st.text_area(
                    "解析内容（可编辑）", value=cleaned, height=300, key="aa_resume_edited"
                )
            resume_name = uploaded_pdf.name
        else:
            pdf_status.update(label="❌ PDF 解析失败", state="error")
            st.sidebar.error("无法提取文字（可能是扫描版）")
            resume_text = st.sidebar.text_area(
                "手动粘贴简历文本", height=200, placeholder="粘贴到这里...", key="aa_resume_fallback"
            )
    else:
        st.sidebar.caption("或直接粘贴文本：")
        resume_text = st.sidebar.text_area(
            "粘贴简历文本", height=200, placeholder="粘贴简历内容或上传 PDF ↑", key="aa_resume_manual"
        )

    search_btn = st.sidebar.button(
        "🔍 搜索匹配岗位", width="stretch", type="primary", key="aa_search_btn",
    )

    # ════════════════════════════════════════════════════════
    #  主区域
    # ════════════════════════════════════════════════════════
    tab_main, tab_history = st.tabs(["📋 投递流程", "📜 投递历史"])

    with tab_main:

        # ── 搜索按钮触发：校验 + 全量重置 ───────────────────
        if search_btn:
            if not resume_text.strip():
                st.error("请先上传 PDF 简历或粘贴简历文本")
                st.stop()
            if not target_title.strip():
                st.error("请输入目标岗位名称")
                st.stop()

            st.session_state.phase           = "scraping"
            st.session_state.search_state    = None
            st.session_state.apply_result    = None
            st.session_state.confirmed_jobs  = None
            st.session_state.prefetched_jobs = None
            st.session_state.scrape_done     = False
            st.session_state.scrape_sub      = "login_pending"
            st.session_state.search_params   = {
                "resume_text":  resume_text,
                "resume_name":  resume_name,
                "target_title": target_title,
                "target_city":  target_city,
                "threshold":    threshold,
                "max_apply":    max_apply,
                "use_real":     use_real,
            }
            st.rerun()

        # ── scraping 阶段 ────────────────────────────────────
        if st.session_state.phase == "scraping":
            params = st.session_state.get("search_params", {})

            result = _run_scrape_if_needed(
                use_real=params.get("use_real", False),
                target_title=params.get("target_title", ""),
                target_city=params.get("target_city", ""),
            )

            if result == "done":
                st.session_state.phase = "search"
                st.rerun()
            elif result == "failed":
                st.session_state.scrape_sub  = "login_pending"
                st.session_state.phase       = "search"
                st.session_state.search_params = None
                st.stop()
            else:
                # "wait"：留在当前页面，等用户点击按钮
                st.stop()

        # ── search 阶段：LangGraph embed→fetch→match→filter ──
        if st.session_state.phase == "search" and st.session_state.get("search_params"):
            params = st.session_state.search_params
            result = _render_search_phase(
                resume_text=params["resume_text"],
                resume_name=params["resume_name"],
                target_title=params["target_title"],
                target_city=params["target_city"],
                threshold=params["threshold"],
                max_apply=params["max_apply"],
                use_real=params["use_real"],
            )
            if result:
                st.session_state.search_state = result
                st.session_state.phase        = "confirm"
                st.rerun()

        # ── confirm 阶段：预览确认 ───────────────────────────
        if st.session_state.phase == "confirm" and st.session_state.search_state:
            confirmed = _render_confirm_phase(st.session_state.search_state)
            if confirmed is not None:
                st.session_state.confirmed_jobs = confirmed
                st.session_state.phase          = "apply"
                st.rerun()

        # ── apply 阶段：生成求职信 + 投递 ────────────────────
        if st.session_state.phase == "apply" and st.session_state.confirmed_jobs:
            params = st.session_state.get("search_params", {})
            result = _render_apply_phase(
                search_state=st.session_state.search_state,
                confirmed_jobs=st.session_state.confirmed_jobs,
                use_real=params.get("use_real", False),
            )
            if result:
                st.session_state.apply_result = result
                st.session_state.phase        = "done"
                st.rerun()

        # ── done 阶段：展示最终结果 ───────────────────────────
        if st.session_state.phase == "done" and st.session_state.apply_result:
            result = st.session_state.apply_result
            if not result.get("error"):
                _render_stats_panel(
                    result.get("apply_stats", {}),
                    result.get("applied_jobs", []),
                )
            if st.button("🔄 重新搜索", width="content"):
                for k in ["search_state", "apply_result", "confirmed_jobs",
                          "prefetched_jobs", "search_params"]:
                    st.session_state[k] = None
                st.session_state.scrape_done = False
                st.session_state.scrape_sub  = "login_pending"
                st.session_state.phase       = "search"
                st.rerun()

        # ── 空态引导 ──────────────────────────────────────────
        if st.session_state.phase == "search" and not st.session_state.get("search_params"):
            st.markdown("""
            <div style="text-align:center; padding: 4rem 2rem; color: #94a3b8;">
                <div style="font-size:3rem">🎯</div>
                <h3 style="color:#64748b">准备好了吗？</h3>
                <p>在左侧填写配置，上传简历，点击「搜索匹配岗位」开始</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_history:
        _render_history_tab()


if __name__ == "__main__":
    run()