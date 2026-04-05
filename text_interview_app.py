# text_interview_app.py
import streamlit as st
from src.dynamic_workflow import build_workflow, AgentState
from langchain_core.messages import HumanMessage, AIMessage
import os

def run():
    # ── 恢复 sidebar（首页CSS把它隐藏了，这里覆盖回来）──
    # st.markdown("""
    # <style>
    # [data-testid="stSidebar"] { display: flex !important; }
    # </style>
    # """, unsafe_allow_html=True)

    # ── 返回首页按钮（右上角）──────────────────────────────────────────────
    st.markdown("""
    <style>
    .home-btn-container {
        position: fixed;
        top: 0.8rem;
        right: 1rem;
        z-index: 9999;
    }
    .home-btn-container a {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #FFFFFF;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 0.4rem 0.9rem;
        color: #94a3b8 !important;
        font-size: 0.82rem;
        font-family: 'DM Sans', sans-serif;
        text-decoration: none !important;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .home-btn-container a:hover {
        background: #FFFFFF;
        border-color: rgba(99,102,241,0.4);
        color: #a5b4fc !important;
    }
    </style>
    <div class="home-btn-container">
        <a href="?page=home">🏠 返回首页</a>
    </div>
    """, unsafe_allow_html=True)

    # ── 页面标题 ──────────────────────────────────────────────────────────────
    st.title("Talent Talk")
    st.markdown("*AI-powered technical interviews with dynamic resume analysis*")

    # ── Sidebar: Interview Setup ──────────────────────────────────────────────
    st.sidebar.header("Interview Setup")

    # 注意：session_state key 统一改为 ti_state，避免与其他模块冲突
    current_mode     = st.session_state.ti_state.get("mode", "friendly")            if "ti_state" in st.session_state else "friendly"
    current_position = st.session_state.ti_state.get("position", "AI Developer")    if "ti_state" in st.session_state else "AI Developer"
    current_company  = st.session_state.ti_state.get("company_name", "Tech Innovators Inc.") if "ti_state" in st.session_state else "Tech Innovators Inc."
    current_num_q    = st.session_state.ti_state.get("num_of_q", 2)                 if "ti_state" in st.session_state else 2
    current_num_f    = st.session_state.ti_state.get("num_of_follow_up", 1)         if "ti_state" in st.session_state else 1

    mode         = st.sidebar.selectbox("Interviewer Mode", ["friendly", "formal", "technical"],
                       index=["friendly", "formal", "technical"].index(current_mode))
    position     = st.sidebar.text_input("Position",     value=current_position)
    company      = st.sidebar.text_input("Company Name", value=current_company)
    num_of_q     = st.sidebar.number_input("Number of Technical Questions", min_value=1, max_value=10, value=current_num_q)
    num_of_f     = st.sidebar.number_input("Number of Follow-up Questions",  min_value=0, max_value=3,  value=current_num_f)

    params_changed = (
        "ti_state" in st.session_state and (
            mode     != st.session_state.ti_state.get("mode") or
            position != st.session_state.ti_state.get("position") or
            company  != st.session_state.ti_state.get("company_name") or
            num_of_q != st.session_state.ti_state.get("num_of_q") or
            num_of_f != st.session_state.ti_state.get("num_of_follow_up")
        )
    )
    if params_changed:
        st.sidebar.warning("Interview parameters have changed. Click 'Update Parameters' to apply changes.")

    if st.sidebar.button("Update Parameters") and "ti_state" in st.session_state:
        st.session_state.ti_state.update({
            "mode": mode, "position": position, "company_name": company,
            "num_of_q": num_of_q, "num_of_follow_up": num_of_f
        })
        st.sidebar.success("Interview parameters updated!")

    # ── 文件上传 ──────────────────────────────────────────────────────────────
    st.sidebar.header("Upload Files")

    st.sidebar.subheader("Candidate Resume")
    resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"], key="ti_resume_uploader")
    resume_path = None
    if resume_file:
        resume_dir = "./uploaded_resumes"
        os.makedirs(resume_dir, exist_ok=True)
        resume_path = os.path.join(resume_dir, resume_file.name)
        with open(resume_path, "wb") as f:
            f.write(resume_file.read())
        st.sidebar.success(f"Resume uploaded: {resume_file.name}")

    st.sidebar.subheader("Interview Questions")
    questions_file = st.sidebar.file_uploader("Upload Questions (PDF)", type=["pdf"], key="ti_questions_uploader")
    questions_path = None
    if questions_file:
        questions_dir = "./uploaded_questions"
        os.makedirs(questions_dir, exist_ok=True)
        questions_path = os.path.join(questions_dir, questions_file.name)
        with open(questions_path, "wb") as f:
            f.write(questions_file.read())
        st.sidebar.success(f"Questions uploaded: {questions_file.name}")

    # ── 工作流初始化 ──────────────────────────────────────────────────────────
    workflow = build_workflow()

    # ── Session State 初始化 ──────────────────────────────────────────────────
    if "ti_state" not in st.session_state:
        st.session_state.ti_state = AgentState(
            mode=mode, num_of_q=num_of_q, num_of_follow_up=num_of_f,
            position=position, evaluation_result="", company_name=company,
            messages=[], report="", pdf_path=None,
            resume_path=resume_path, questions_path=questions_path
        )
    else:
        if resume_path and st.session_state.ti_state.get("resume_path") != resume_path:
            st.session_state.ti_state["resume_path"] = resume_path
            st.sidebar.info("Resume updated. The interviewer will use your new resume.")
        if questions_path and st.session_state.ti_state.get("questions_path") != questions_path:
            st.session_state.ti_state["questions_path"] = questions_path
            st.sidebar.info("Interview questions updated. The interviewer will use your new questions.")

    # ── 面试主循环 ────────────────────────────────────────────────────────────
    st.header("Interview")
    user_input = st.text_input("Your answer (as candidate):", "", key="ti_user_input")

    if st.button("Send", key="ti_send") and user_input:
        st.session_state.ti_state["messages"].append(HumanMessage(content=user_input))

        interview_ended = False
        for msg in reversed(st.session_state.ti_state["messages"]):
            if isinstance(msg, AIMessage) and "that's it for today" in msg.content.lower():
                interview_ended = True
                st.info("Interview has ended. Generating evaluation and report...")
                break

        current_state = AgentState(
            mode=st.session_state.ti_state["mode"],
            num_of_q=st.session_state.ti_state["num_of_q"],
            num_of_follow_up=st.session_state.ti_state["num_of_follow_up"],
            position=st.session_state.ti_state["position"],
            company_name=st.session_state.ti_state["company_name"],
            messages=st.session_state.ti_state["messages"],
            evaluation_result="" if interview_ended else st.session_state.ti_state.get("evaluation_result", ""),
            report=""            if interview_ended else st.session_state.ti_state.get("report", ""),
            pdf_path=st.session_state.ti_state.get("pdf_path"),
            resume_path=st.session_state.ti_state.get("resume_path"),
            questions_path=st.session_state.ti_state.get("questions_path")
        )

        try:
            result = workflow.invoke(current_state)
            for key, value in result.items():
                if key == "evaluation_result" and interview_ended:
                    st.session_state.ti_state["evaluation_result"] = value
                elif key == "report" and value:
                    st.session_state.ti_state["report"] = value
                elif key == "pdf_path" and value:
                    st.session_state.ti_state["pdf_path"] = value
                elif key == "messages":
                    st.session_state.ti_state["messages"] = value
                else:
                    st.session_state.ti_state[key] = value
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # ── 对话记录 ──────────────────────────────────────────────────────────────
    st.subheader("Transcript")
    for m in st.session_state.ti_state["messages"]:
        if isinstance(m, HumanMessage):
            st.markdown(f"**Candidate:** {m.content}")
        elif isinstance(m, AIMessage):
            st.markdown(f"**AI Recruiter:** {m.content}")

    # ── 面试结束：生成评估报告 ────────────────────────────────────────────────
    interview_ended = False
    for msg in reversed(st.session_state.ti_state.get("messages", [])):
        if isinstance(msg, AIMessage) and "that's it for today" in msg.content.lower():
            interview_ended = True
            break

    if interview_ended and not st.session_state.ti_state.get("evaluation_result"):
        st.warning("Interview has ended. Click the button below to generate evaluation and report.")
        if st.button("Generate Evaluation and Report", key="ti_gen_report"):
            st.info("Generating evaluation and report... This may take a moment.")
            try:
                current_state = AgentState(
                    mode=st.session_state.ti_state["mode"],
                    num_of_q=st.session_state.ti_state["num_of_q"],
                    num_of_follow_up=st.session_state.ti_state["num_of_follow_up"],
                    position=st.session_state.ti_state["position"],
                    company_name=st.session_state.ti_state["company_name"],
                    messages=st.session_state.ti_state["messages"],
                    evaluation_result="", report="", pdf_path=None,
                    resume_path=st.session_state.ti_state.get("resume_path"),
                    questions_path=st.session_state.ti_state.get("questions_path")
                )

                from src.dynamic_workflow import evaluator, report_writer, pdf_generator_node
                import time

                with st.spinner("Generating evaluation..."):
                    for attempt in range(3):
                        try:
                            if not any(isinstance(m, HumanMessage) for m in current_state["messages"]):
                                current_state["messages"].append(HumanMessage(content="Please evaluate the interview."))
                            eval_result = evaluator(current_state)
                            st.session_state.ti_state["evaluation_result"] = eval_result["evaluation_result"]
                            break
                        except Exception as e:
                            if attempt == 2: raise
                            st.warning(f"Retry {attempt+1}/3: {str(e)}")
                            time.sleep(2)

                with st.spinner("Generating report..."):
                    current_state["evaluation_result"] = st.session_state.ti_state["evaluation_result"]
                    for attempt in range(3):
                        try:
                            report_result = report_writer(current_state)
                            st.session_state.ti_state["report"] = report_result["report"]
                            break
                        except Exception as e:
                            if attempt == 2: raise
                            st.warning(f"Retry {attempt+1}/3: {str(e)}")
                            time.sleep(2)

                with st.spinner("Generating PDF..."):
                    current_state["report"] = st.session_state.ti_state["report"]
                    pdf_result = pdf_generator_node(current_state)
                    st.session_state.ti_state["pdf_path"] = pdf_result["pdf_path"]

                st.success("Evaluation and report generated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    # ── 评估结果展示 ──────────────────────────────────────────────────────────
    if st.session_state.ti_state.get("evaluation_result"):
        st.subheader("Evaluation Result")
        st.markdown(st.session_state.ti_state["evaluation_result"])

    # ── 报告 & PDF 下载 ───────────────────────────────────────────────────────
    if st.session_state.ti_state.get("report"):
        st.subheader("HR Report")
        st.markdown(st.session_state.ti_state["report"])
        pdf_path = st.session_state.ti_state.get("pdf_path")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF Report", f,
                    file_name=os.path.basename(pdf_path),
                    key="ti_dl_pdf"
                )

    # ── App State 调试（侧边栏）──────────────────────────────────────────────
    st.sidebar.markdown("---")
    with st.expander("App State", expanded=False):
        state_copy = dict(st.session_state.ti_state)
        if "messages" in state_copy:
            state_copy["messages"] = [
                f"{type(m).__name__}: {m.content if hasattr(m, 'content') else str(m)}"
                for m in state_copy["messages"]
            ]
        st.code(str(state_copy), language="python")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    **Talent Talk** | Revolutionizing technical interviews with AI

    **Powered by:**
    - **LangGraph** - AI Agent
    - **Google Generative AI** - Language Model
    - **Streamlit** - Web Interface
    """)