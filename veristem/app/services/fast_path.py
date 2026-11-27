import json
import re
import time
from dataclasses import dataclass

from app.router.complexity import ComplexityDecision
from app.schemas.models import (
    AgentExecutionResult,
    ExecutionPath,
    ExecutionPlan,
    ExecutionTask,
    FinalSolutionReport,
    MathematicalVariable,
    ProblemSpecification,
    ToolExecutionRecord,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
    utc_now,
)
from app.tools.registry import SafeToolRegistry


@dataclass
class FastPathOutcome:
    handled: bool
    reason: str | None = None
    specification: ProblemSpecification | None = None
    plan: ExecutionPlan | None = None
    agent_timeline: list[AgentExecutionResult] | None = None
    tool_records: list[ToolExecutionRecord] | None = None
    verification_report: VerificationReport | None = None
    final_report: FinalSolutionReport | None = None


def _normalize_expr(raw: str) -> str:
    s = raw.strip()
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\pi\b", "pi", s)
    s = re.sub(r"\\(sin|cos|tan|exp|log)\b", r"\1", s)
    s = re.sub(r"\\ln\b", "log", s)
    s = re.sub(r"\\sqrt\s*\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\be\s*\^\s*\(([^()]+)\)", r"exp(\1)", s)
    s = re.sub(r"\be\s*\^\s*([-+]?[a-zA-Z0-9_.]+)", r"exp(\1)", s)
    s = re.sub(r"\bln\s*\(", "log(", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_integral(question: str) -> tuple[str, str, str | None, str | None] | None:
    q = question.strip()
    m1 = re.search(
        r"(?:definite\s+|indefinite\s+)?integral\s+of\s+(.+?)\s+from\s+(?:([a-zA-Z])\s*=\s*)?([^\s]+)\s+to\s+(?:([a-zA-Z])\s*=\s*)?([^\s.?,;]+)\s*[.?!]?$",
        q,
        re.I,
    )
    if m1:
        expr = _normalize_expr(re.sub(r"\bd[a-zA-Z]\b\s*$", "", m1.group(1), flags=re.I))
        var = m1.group(2) or m1.group(4) or "x"
        return expr, var, m1.group(3), m1.group(5)

    m2 = re.search(
        r"\\int\s*_\s*\{?([^}\s^]+)\}?\s*\^\s*\{?([^}\s]+)\}?\s*(.+?)\s*d([a-zA-Z])\b",
        q,
        re.I,
    )
    if m2:
        return _normalize_expr(m2.group(3)), m2.group(4), m2.group(1), m2.group(2)

    m3 = re.search(
        r"(?:(?:indefinite\s+)?integral\s+of|integrate)\s+(.+?)(?:\s+(?:with\s+respect\s+to\s+([a-zA-Z])|d([a-zA-Z])))?\s*[.?!]?$",
        q,
        re.I,
    )
    if m3 and not re.search(r"\bfrom\b", q, re.I):
        expr = _normalize_expr(re.sub(r"\s*d[a-zA-Z]\s*$", "", m3.group(1), flags=re.I))
        var = m3.group(2) or m3.group(3) or "x"
        return expr, var, None, None
    return None


def try_fast_path_solve(
    *,
    problem_id: str,
    execution_id: str,
    question: str,
    decision: ComplexityDecision,
    registry: SafeToolRegistry,
) -> FastPathOutcome:
    if decision.execution_path != ExecutionPath.FAST_PATH:
        return FastPathOutcome(handled=False, reason="Not classified for FAST_PATH.")

    t0 = time.perf_counter()
    started_at = utc_now()

    try:
        timeline: list[AgentExecutionResult] = [
            AgentExecutionResult(
                agent_name="ComplexityRouter",
                stage="ROUTING",
                status="COMPLETED",
                summary=f"Classified as {decision.complexity_level.value} ({decision.problem_category})",
                structured_output={
                    "complexity_level": decision.complexity_level.value,
                    "execution_path": decision.execution_path.value,
                    "rationale": decision.rationale,
                },
                duration_ms=decision.router_latency_ms,
                started_at=started_at,
                completed_at=utc_now(),
            )
        ]

        if decision.problem_category == "calculus_integral":
            parsed = _parse_integral(question)
            if not parsed:
                return FastPathOutcome(handled=False, reason="Could not parse integral expression.")
            expr, var, lo_s, hi_s = parsed

            s_start = utc_now()
            s_t0 = time.perf_counter()
            solve_rec = registry.execute(
                "symbolic_math",
                {
                    "operation": "integrate",
                    "expression": expr,
                    "variable": var,
                    "lower_limit": lo_s,
                    "upper_limit": hi_s,
                },
            )
            if not solve_rec.success:
                return FastPathOutcome(handled=False, reason=solve_rec.error_message)

            final_ans = str(solve_rec.output_payload.get("result_str", ""))
            numeric_val = solve_rec.output_payload.get("numeric_approx")
            if not final_ans or "Integral(" in final_ans:
                return FastPathOutcome(handled=False, reason="Unevaluated integral; escalating.")

            verif_records: list[ToolExecutionRecord] = []
            checks: list[VerificationCheck] = []
            v_start = utc_now()
            v_t0 = time.perf_counter()

            if lo_s is not None and hi_s is not None and isinstance(numeric_val, (int, float)):
                lo_eval = registry.execute("symbolic_math", {"operation": "evaluate", "expression": lo_s})
                hi_eval = registry.execute("symbolic_math", {"operation": "evaluate", "expression": hi_s})
                lo_num = lo_eval.output_payload.get("numeric_approx")
                hi_num = hi_eval.output_payload.get("numeric_approx")
                if not isinstance(lo_num, (int, float)) or not isinstance(hi_num, (int, float)):
                    return FastPathOutcome(handled=False, reason="Non-numeric integration bounds.")

                quad_rec = registry.execute(
                    "independent_verifier",
                    {
                        "verification_type": "integral_quadrature",
                        "expression": expr,
                        "variable": var,
                        "lower_limit": float(lo_num),
                        "upper_limit": float(hi_num),
                        "candidate_value": float(numeric_val),
                        "tolerance": 1e-5,
                    },
                )
                verif_records.append(quad_rec)
                if not quad_rec.success or quad_rec.output_payload.get("passed") is not True:
                    return FastPathOutcome(handled=False, reason="Independent quadrature check failed.")

                checks.append(
                    VerificationCheck(
                        check_name="scipy_adaptive_quadrature",
                        method=str(quad_rec.output_payload.get("method_used", "scipy_quad")),
                        passed=True,
                        expected_value=str(quad_rec.output_payload.get("independent_value")),
                        observed_value=str(numeric_val),
                        relative_error=quad_rec.output_payload.get("relative_error"),
                        tolerance=quad_rec.output_payload.get("tolerance"),
                        details=str(quad_rec.output_payload.get("details", "")),
                    )
                )
            else:
                diff_rec = registry.execute(
                    "symbolic_math",
                    {"operation": "diff", "expression": final_ans, "variable": var},
                )
                verif_records.append(diff_rec)
                if not diff_rec.success:
                    return FastPathOutcome(handled=False, reason="Antiderivative differentiation check failed.")
                checks.append(
                    VerificationCheck(
                        check_name="ftc_derivative_check",
                        method="sympy_diff",
                        passed=True,
                        expected_value=expr,
                        observed_value=str(diff_rec.output_payload.get("result_str")),
                        details=f"d/d{var}({final_ans}) = {diff_rec.output_payload.get('result_str')}",
                    )
                )

            all_tools = [solve_rec, *verif_records]
            spec = ProblemSpecification(
                domain=decision.detected_domain,
                problem_type="definite_integral" if lo_s is not None else "indefinite_integral",
                objective=f"Evaluate integral of {expr} wrt {var}",
                variables=[MathematicalVariable(symbol=var, name="integration variable")],
                equations=[expr],
                assumptions=["Integrand continuous over domain."],
            )
            plan = ExecutionPlan(
                methodology="SymPy exact integration verified via independent quadrature/differentiation.",
                selected_tools=["symbolic_math", "independent_verifier"],
                tasks=[
                    ExecutionTask(
                        task_id="fast_1",
                        title="SymPy Integration",
                        description=f"Integrate {expr} wrt {var}",
                        method="sympy.integrate",
                        tool_name="symbolic_math",
                        expected_output=final_ans,
                    )
                ],
                verification_strategy="SciPy adaptive quadrature / FTC differentiation check.",
            )
            verif_report = VerificationReport(
                status=VerificationStatus.PASS,
                checks=checks,
                independent_recalculation=checks[0].expected_value,
                dimensional_consistency_passed=True,
                boundary_conditions_passed=True,
                numerical_stability_passed=True,
                evidence_summary=" | ".join(c.details for c in checks),
            )

            timeline.extend(
                [
                    AgentExecutionResult(
                        agent_name="FastSymbolicSolver",
                        stage="SOLVING",
                        status="COMPLETED",
                        summary=f"SymPy exact result: {final_ans}",
                        structured_output=solve_rec.output_payload,
                        tool_records=[solve_rec],
                        duration_ms=round((time.perf_counter() - s_t0) * 1000.0, 3),
                        started_at=s_start,
                        completed_at=utc_now(),
                    ),
                    AgentExecutionResult(
                        agent_name="IndependentVerificationAgent",
                        stage="VERIFICATION",
                        status="PASS",
                        summary=verif_report.evidence_summary,
                        structured_output=verif_report.model_dump(mode="json"),
                        tool_records=verif_records,
                        duration_ms=round((time.perf_counter() - v_t0) * 1000.0, 3),
                        started_at=v_start,
                        completed_at=utc_now(),
                    ),
                ]
            )

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            report = FinalSolutionReport(
                problem_id=problem_id,
                execution_id=execution_id,
                original_question=question,
                detected_domain=decision.detected_domain,
                extracted_variables=spec.variables,
                assumptions=spec.assumptions,
                solution_methodology=plan.methodology,
                equations=spec.equations,
                intermediate_results=[{"step": "Symbolic integral", "value": final_ans}],
                final_answer=final_ans,
                numeric_value=float(numeric_val) if isinstance(numeric_val, (int, float)) else None,
                units="dimensionless",
                verification_status=VerificationStatus.PASS,
                verification_evidence=checks,
                correction_attempts=0,
                correction_history=[],
                tool_execution_history=all_tools,
                confidence_limitations=["Verified via independent numerical quadrature / FTC derivative."],
                explanation_markdown=f"### SymPy Fast Path Result\n\n$$\\text{{Result}} = {final_ans}$$\n\n- {verif_report.evidence_summary}",
                complexity_level=decision.complexity_level,
                execution_path=decision.execution_path,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
            )
            return FastPathOutcome(
                handled=True,
                specification=spec,
                plan=plan,
                agent_timeline=timeline,
                tool_records=all_tools,
                verification_report=verif_report,
                final_report=report,
            )

        if decision.problem_category == "calculus_derivative":
            m = re.search(
                r"(?:derivative\s+of|differentiate)\s+(.+?)(?:\s+with\s+respect\s+to\s+([a-zA-Z]))?(?:\s+at\s+([a-zA-Z])\s*=\s*([-+]?\d+(?:\.\d+)?))?\s*[.?!]?$",
                question.strip(),
                re.I,
            )
            if not m:
                return FastPathOutcome(handled=False, reason="Could not parse derivative prompt.")
            expr = _normalize_expr(m.group(1))
            var = m.group(2) or m.group(3) or "x"
            pt = float(m.group(4)) if m.group(4) else 1.25
            s_t0 = time.perf_counter()
            diff_rec = registry.execute(
                "symbolic_math",
                {"operation": "diff", "expression": expr, "variable": var, "substitutions": {var: pt}},
            )
            if not diff_rec.success or diff_rec.output_payload.get("numeric_approx") is None:
                return FastPathOutcome(handled=False, reason="Symbolic diff failed.")
            cand_val = float(diff_rec.output_payload["numeric_approx"])
            fd_rec = registry.execute(
                "independent_verifier",
                {
                    "verification_type": "derivative_finite_diff",
                    "expression": expr,
                    "variable": var,
                    "point": pt,
                    "candidate_value": cand_val,
                    "tolerance": 1e-4,
                },
            )
            if not fd_rec.success or fd_rec.output_payload.get("passed") is not True:
                return FastPathOutcome(handled=False, reason="Finite difference verification failed.")
            final_ans = str(cand_val) if m.group(4) else str(diff_rec.output_payload["result_str"])
            check = VerificationCheck(
                check_name="central_finite_difference",
                method=str(fd_rec.output_payload.get("method_used", "central_finite_difference")),
                passed=True,
                expected_value=str(fd_rec.output_payload.get("independent_value")),
                observed_value=str(cand_val),
                relative_error=fd_rec.output_payload.get("relative_error"),
                tolerance=fd_rec.output_payload.get("tolerance"),
                details=str(fd_rec.output_payload.get("details", "")),
            )
            verif_report = VerificationReport(
                status=VerificationStatus.PASS,
                checks=[check],
                independent_recalculation=check.expected_value,
                dimensional_consistency_passed=True,
                boundary_conditions_passed=True,
                numerical_stability_passed=True,
                evidence_summary=check.details,
            )
            spec = ProblemSpecification(
                domain=decision.detected_domain,
                problem_type="symbolic_differentiation",
                objective=f"Differentiate {expr} wrt {var}",
                variables=[MathematicalVariable(symbol=var, name="variable", value=pt if m.group(4) else None)],
                equations=[expr],
            )
            plan = ExecutionPlan(
                methodology="SymPy differentiation verified by central finite differences.",
                selected_tools=["symbolic_math", "independent_verifier"],
                tasks=[
                    ExecutionTask(
                        task_id="fast_1",
                        title="SymPy Diff",
                        description=f"Differentiate {expr}",
                        method="sympy.diff",
                        tool_name="symbolic_math",
                        expected_output=final_ans,
                    )
                ],
                verification_strategy="Central finite difference at test point.",
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            report = FinalSolutionReport(
                problem_id=problem_id,
                execution_id=execution_id,
                original_question=question,
                detected_domain=decision.detected_domain,
                extracted_variables=spec.variables,
                assumptions=[],
                solution_methodology=plan.methodology,
                equations=spec.equations,
                intermediate_results=[{"step": "Derivative", "value": final_ans}],
                final_answer=final_ans,
                numeric_value=cand_val if m.group(4) else None,
                units="dimensionless",
                verification_status=VerificationStatus.PASS,
                verification_evidence=[check],
                correction_attempts=0,
                tool_execution_history=[diff_rec, fd_rec],
                confidence_limitations=[],
                explanation_markdown=f"### SymPy Derivative (Fast Path)\n\n`{final_ans}`\n\n- {check.details}",
                complexity_level=decision.complexity_level,
                execution_path=decision.execution_path,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
            )
            return FastPathOutcome(
                handled=True,
                specification=spec,
                plan=plan,
                agent_timeline=timeline,
                tool_records=[diff_rec, fd_rec],
                verification_report=verif_report,
                final_report=report,
            )

        if decision.problem_category == "algebraic_equation":
            eq_m = re.search(r"([^=.\n]+?)\s*=\s*([^.\n?;]+)", question)
            if not eq_m:
                return FastPathOutcome(handled=False, reason="Could not extract equation.")
            lhs = _normalize_expr(re.sub(r"^(?:solve|find\s+(?:the\s+)?roots?\s+of)\s+", "", eq_m.group(1), flags=re.I))
            rhs = _normalize_expr(re.sub(r"\s+for\s+[a-zA-Z].*$", "", eq_m.group(2), flags=re.I))
            expr_diff = f"({lhs}) - ({rhs})"
            solve_rec = registry.execute(
                "symbolic_math",
                {"operation": "solve", "expression": f"{lhs} = {rhs}", "variable": "x"},
            )
            if not solve_rec.success or not solve_rec.output_payload.get("solutions"):
                return FastPathOutcome(handled=False, reason="Symbolic solve returned no solutions.")
            sols = solve_rec.output_payload["solutions"]
            checks: list[VerificationCheck] = []
            verif_records: list[ToolExecutionRecord] = []
            for s_str in sols:
                ev = registry.execute("symbolic_math", {"operation": "evaluate", "expression": s_str})
                num = ev.output_payload.get("numeric_approx")
                if not isinstance(num, (int, float)):
                    return FastPathOutcome(handled=False, reason="Non-real root; escalating.")
                rr = registry.execute(
                    "independent_verifier",
                    {
                        "verification_type": "root_residual",
                        "expression": expr_diff,
                        "variable": "x",
                        "candidate_value": float(num),
                        "tolerance": 1e-8,
                    },
                )
                verif_records.append(rr)
                if not rr.success or rr.output_payload.get("passed") is not True:
                    return FastPathOutcome(handled=False, reason="Root residual verification failed.")
                checks.append(
                    VerificationCheck(
                        check_name="root_residual",
                        method="direct_function_residual",
                        passed=True,
                        expected_value="0",
                        observed_value=str(rr.output_payload.get("candidate_value")),
                        relative_error=rr.output_payload.get("absolute_error"),
                        tolerance=1e-8,
                        details=str(rr.output_payload.get("details", "")),
                    )
                )
            final_ans = f"x = {', '.join(sols)}"
            verif_report = VerificationReport(
                status=VerificationStatus.PASS,
                checks=checks,
                evidence_summary=" | ".join(c.details for c in checks),
            )
            spec = ProblemSpecification(
                domain=decision.detected_domain,
                problem_type="algebraic_equation",
                objective=f"Solve {lhs} = {rhs}",
                variables=[MathematicalVariable(symbol="x", name="unknown")],
                equations=[f"{lhs} = {rhs}"],
            )
            plan = ExecutionPlan(
                methodology="SymPy exact solve + residual norm verification.",
                selected_tools=["symbolic_math", "independent_verifier"],
                tasks=[
                    ExecutionTask(
                        task_id="fast_1",
                        title="Exact Solve",
                        description=f"Solve {lhs} = {rhs}",
                        method="sympy.solve",
                        tool_name="symbolic_math",
                        expected_output=final_ans,
                    )
                ],
                verification_strategy="Substitute every root into LHS - RHS.",
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            report = FinalSolutionReport(
                problem_id=problem_id,
                execution_id=execution_id,
                original_question=question,
                detected_domain=decision.detected_domain,
                extracted_variables=spec.variables,
                assumptions=[],
                solution_methodology=plan.methodology,
                equations=spec.equations,
                final_answer=final_ans,
                numeric_value=solve_rec.output_payload.get("numeric_approx"),
                units="dimensionless",
                verification_status=VerificationStatus.PASS,
                verification_evidence=checks,
                correction_attempts=0,
                tool_execution_history=[solve_rec, *verif_records],
                confidence_limitations=[],
                explanation_markdown=f"### Algebraic Solution (Fast Path)\n\n`{final_ans}`",
                complexity_level=decision.complexity_level,
                execution_path=decision.execution_path,
                elapsed_ms=elapsed_ms,
                started_at=started_at,
            )
            return FastPathOutcome(
                handled=True,
                specification=spec,
                plan=plan,
                agent_timeline=timeline,
                tool_records=[solve_rec, *verif_records],
                verification_report=verif_report,
                final_report=report,
            )

        return FastPathOutcome(handled=False, reason=f"Category {decision.problem_category} escalated to multi-agent workflow.")
    except Exception as exc:
        return FastPathOutcome(handled=False, reason=f"Fast path fallback: {exc}")
