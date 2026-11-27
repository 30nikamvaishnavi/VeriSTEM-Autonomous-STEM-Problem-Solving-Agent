import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import type { CorrectionRequest, VerificationReport } from '../types/veristem';

interface VerificationPanelProps {
  verification: VerificationReport | null;
  corrections: CorrectionRequest[];
}

export function VerificationPanel({ verification, corrections }: VerificationPanelProps) {
  if (!verification) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h3 className="text-base font-semibold text-slate-900">Independent Verification</h3>
        <p className="text-sm text-slate-500 mt-2">
          No verification report recorded yet. Verification executes automatically after the STEM Solver completes tool calculations.
        </p>
      </div>
    );
  }

  const isPass = verification.status === 'PASS';
  const isFail = verification.status === 'FAIL';
  const Icon = isPass ? CheckCircle2 : isFail ? XCircle : AlertTriangle;
  const tone = isPass
    ? 'text-emerald-700 bg-emerald-50 border-emerald-200'
    : isFail
      ? 'text-rose-700 bg-rose-50 border-rose-200'
      : 'text-amber-700 bg-amber-50 border-amber-200';

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h3 className="text-base font-semibold text-slate-900">Independent Verification Evidence</h3>
          <p className="text-sm text-slate-600 mt-1">{verification.evidence_summary}</p>
        </div>
        <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-md border text-xs font-semibold ${tone}`}>
          <Icon className="w-4 h-4 shrink-0" />
          <span>{verification.status}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-3 border-t border-slate-100 text-xs">
        <div>
          <span className="text-slate-500">Dimensional Consistency</span>
          <div className="font-mono font-medium text-slate-900 mt-1">
            {verification.dimensional_consistency_passed === null
              ? 'Not applicable'
              : verification.dimensional_consistency_passed
                ? 'Passed'
                : 'Failed'}
          </div>
        </div>
        <div>
          <span className="text-slate-500">Boundary Conditions</span>
          <div className="font-mono font-medium text-slate-900 mt-1">
            {verification.boundary_conditions_passed === null
              ? 'Not applicable'
              : verification.boundary_conditions_passed
                ? 'Passed'
                : 'Failed'}
          </div>
        </div>
        <div>
          <span className="text-slate-500">Numerical Stability</span>
          <div className="font-mono font-medium text-slate-900 mt-1">
            {verification.numerical_stability_passed === null
              ? 'Not applicable'
              : verification.numerical_stability_passed
                ? 'Passed'
                : 'Failed'}
          </div>
        </div>
      </div>

      {verification.checks.length > 0 && (
        <div className="overflow-x-auto border-t border-slate-100 pt-4">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500">
                <th className="py-2 pr-3 font-medium">Check</th>
                <th className="py-2 pr-3 font-medium">Method</th>
                <th className="py-2 pr-3 font-medium">Status</th>
                <th className="py-2 pr-3 font-medium text-right">Rel. Error</th>
                <th className="py-2 font-medium">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {verification.checks.map((chk, idx) => (
                <tr key={`${chk.check_name}_${idx}`} className="hover:bg-slate-50">
                  <td className="py-2.5 pr-3 font-medium text-slate-900">{chk.check_name}</td>
                  <td className="py-2.5 pr-3 font-mono text-slate-600">{chk.method}</td>
                  <td className="py-2.5 pr-3 font-medium">
                    <span className={chk.passed ? 'text-emerald-700' : 'text-rose-700'}>
                      {chk.passed ? 'PASS' : 'FAIL'}
                    </span>
                  </td>
                  <td className="py-2.5 pr-3 font-mono tabular-nums text-right text-slate-700">
                    {chk.relative_error !== null ? chk.relative_error.toExponential(2) : '—'}
                  </td>
                  <td className="py-2.5 text-slate-600">{chk.details}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {verification.failure_reasons.length > 0 && (
        <div className="border-t border-slate-100 pt-3">
          <h4 className="text-xs font-semibold text-rose-700 mb-1.5">Failure Reasons</h4>
          <ul className="space-y-1 text-xs text-rose-700">
            {verification.failure_reasons.map((reason, i) => (
              <li key={i}>• {reason}</li>
            ))}
          </ul>
        </div>
      )}

      {corrections.length > 0 && (
        <div className="border-t border-slate-100 pt-4 space-y-2">
          <h4 className="text-xs font-semibold text-slate-900">Self-Correction Loop History</h4>
          <div className="space-y-2">
            {corrections.map((corr) => (
              <div key={corr.attempt_number} className="p-3 bg-slate-50 border border-slate-200 rounded-md text-xs space-y-1">
                <div className="flex items-center justify-between font-medium text-slate-900">
                  <span>
                    Attempt {corr.attempt_number} · {corr.error_category}
                  </span>
                  <span className="font-mono text-slate-500">
                    {corr.tools_to_reexecute.join(', ') || 're-solve'}
                  </span>
                </div>
                <p className="text-slate-700">{corr.root_cause}</p>
                <p className="text-slate-500">{corr.corrective_instructions}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
