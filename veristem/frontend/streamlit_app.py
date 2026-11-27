import json
import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("VERISTEM_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("VERISTEM_API_KEY", "")

EXAMPLES = {
    "Calculus — Definite Integral": "Evaluate the definite integral of x^2 * exp(-x) from x = 0 to x = 2 and verify numerically.",
    "Physics — Projectile Kinematics": "A projectile is launched from ground level at 45 m/s at 35 degrees above horizontal (g = 9.81 m/s^2). Compute maximum height, flight time, and horizontal range with SI units.",
    "Quantitative Finance — Black-Scholes Call": "Price a European call option with S = 100, K = 105, r = 0.05, sigma = 0.20, T = 1.0 year, and verify against Monte Carlo simulation.",
    "Linear Algebra — System & Conditioning": "Solve the linear system [[4, 1], [2, 3]] * x = [9, 8], compute eigenvalues, and verify residual norm.",
}


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


st.set_page_config(page_title="VeriSTEM Workbench", layout="wide")
st.title("VeriSTEM — Verifiable STEM Reasoning Workbench")
st.caption("Every numerical and symbolic claim is independently verified before acceptance.")

if "history" not in st.session_state:
    st.session_state.history = []
if "active_problem_id" not in st.session_state:
    st.session_state.active_problem_id = None

with st.sidebar:
    st.subheader("Configuration")
    domain_choice = st.selectbox(
        "Domain",
        options=["auto", "mathematics", "physics", "quantitative_finance", "general_stem"],
    )
    max_attempts = st.slider("Max Self-Correction Attempts", min_value=1, max_value=5, value=3)
    example_key = st.selectbox("Load Example Problem", options=["Custom", *EXAMPLES.keys()])
    st.divider()
    st.subheader("Session Problem History")
    for pid in reversed(st.session_state.history[-10:]):
        if st.button(pid, key=f"hist_{pid}"):
            st.session_state.active_problem_id = pid

default_question = "" if example_key == "Custom" else EXAMPLES[example_key]
question = st.text_area("STEM Problem Statement", value=default_question, height=130)

if st.button("Submit & Verify", type="primary", disabled=not question.strip()):
    payload: dict[str, object] = {"question": question.strip()}
    if domain_choice != "auto":
        payload["domain_hint"] = domain_choice

    with httpx.Client(timeout=30.0) as client:
        create_res = client.post(f"{API_URL}/api/v1/problems", json=payload, headers=_headers())
        create_res.raise_for_status()
        problem_id = create_res.json()["problem_id"]
        st.session_state.active_problem_id = problem_id
        if problem_id not in st.session_state.history:
            st.session_state.history.append(problem_id)

        solve_res = client.post(
            f"{API_URL}/api/v1/problems/{problem_id}/solve",
            json={"max_correction_attempts": max_attempts, "run_async": True},
            headers=_headers(),
        )
        solve_res.raise_for_status()

active_id = st.session_state.active_problem_id
if active_id:
    with httpx.Client(timeout=30.0) as client:
        detail_res = client.get(f"{API_URL}/api/v1/problems/{active_id}", headers=_headers())
        detail_res.raise_for_status()
        detail = detail_res.json()

    status_val = detail["status"]
    st.subheader(f"Execution: `{active_id}` — Status: `{status_val}` (Stage: `{detail['current_stage']}`)")

    if status_val == "RUNNING":
        st.info("Multi-agent verification workflow is running...")
        time.sleep(1.5)
        st.rerun()

    if status_val in ("NEEDS_CLARIFICATION", "HUMAN_REVIEW"):
        st.warning(f"Workflow paused at {status_val}. Provide input to resume from LangGraph checkpoint.")
        with st.form("resume_form"):
            action = st.selectbox("Action", ["clarify", "approve", "override", "reject"])
            clarification = st.text_area("Clarification / Reviewer Notes")
            override_val = st.text_input("Override Answer (only for override)")
            submitted = st.form_submit_button("Resume Workflow")
            if submitted:
                with httpx.Client(timeout=30.0) as client:
                    client.post(
                        f"{API_URL}/api/v1/problems/{active_id}/resume",
                        json={
                            "action": action,
                            "clarification_response": clarification or None,
                            "notes": clarification,
                            "override_answer": override_val or None,
                        },
                        headers=_headers(),
                    ).raise_for_status()
                st.rerun()

    col_left, col_right = st.columns([1.2, 1.0])
    with col_left:
        st.markdown("### Agent Execution Timeline")
        for step in detail.get("agent_timeline", []):
            with st.expander(f"{step['stage']} · {step['agent_name']} ({step['duration_ms']} ms)", expanded=False):
                st.write(step["summary"])
                st.json(step["structured_output"])

        st.markdown("### Tool Execution Evidence")
        for tool in detail.get("tool_records", []):
            badge = "PASS" if tool["success"] else "FAIL"
            with st.expander(f"[{badge}] {tool['tool_name']} ({tool['duration_ms']} ms)"):
                st.json({"input": tool["input_payload"], "output": tool["output_payload"], "error": tool["error_message"]})

    with col_right:
        verif = detail.get("verification_report")
        if verif:
            st.markdown("### Independent Verification")
            if verif["status"] == "PASS":
                st.success(f"Verification Status: PASS — {verif['evidence_summary']}")
            elif verif["status"] == "FAIL":
                st.error(f"Verification Status: FAIL — {verif['evidence_summary']}")
            else:
                st.warning(f"Verification Status: INCONCLUSIVE — {verif['evidence_summary']}")
            for chk in verif.get("checks", []):
                st.write(f"- **{'PASS' if chk['passed'] else 'FAIL'}** `{chk['check_name']}` ({chk['method']}): {chk['details']}")

        if detail.get("correction_history"):
            st.markdown("### Self-Correction Loop History")
            for corr in detail["correction_history"]:
                st.write(f"Attempt {corr['attempt_number']} ({corr['error_category']}): {corr['root_cause']}")

    report = detail.get("final_report")
    if report:
        st.divider()
        st.markdown("### Final Verified Report")
        st.metric("Final Answer", f"{report['final_answer']} ({report['units']})")
        st.markdown(report["explanation_markdown"])

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download JSON Report",
                data=json.dumps(report, indent=2),
                file_name=f"{active_id}_report.json",
                mime="application/json",
            )
        with dl2:
            st.download_button(
                "Download Markdown Report",
                data=report["explanation_markdown"],
                file_name=f"{active_id}_report.md",
                mime="text/markdown",
            )
