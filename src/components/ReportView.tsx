import { Download } from 'lucide-react';
import type { FinalSolutionReport } from '../types/veristem';
import { MathText } from './MathText';

interface ReportViewProps {
  report: FinalSolutionReport | null;
}

function triggerDownload(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function ReportView({ report }: ReportViewProps) {
  if (!report) {
    return null;
  }

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-6 space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4 pb-5 border-b border-slate-200">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 font-mono">
            <span>{report.problem_id}</span>
            <span aria-hidden="true">·</span>
            <span>{report.detected_domain.replace('_', ' ')}</span>
            {report.execution_path && (
              <>
                <span aria-hidden="true">·</span>
                <span className="text-slate-800">{report.execution_path}</span>
              </>
            )}
            <span aria-hidden="true">·</span>
            <span>Verified {report.verification_status}</span>
          </div>
          <h3 className="text-xl font-semibold text-slate-900 mt-1">Final Verified Solution Report</h3>
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          <button
            type="button"
            onClick={() =>
              triggerDownload(
                `${report.problem_id}_report.json`,
                JSON.stringify(report, null, 2),
                'application/json',
              )
            }
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download JSON</span>
          </button>
          <button
            type="button"
            onClick={() =>
              triggerDownload(
                `${report.problem_id}_report.md`,
                report.explanation_markdown,
                'text/markdown',
              )
            }
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download Markdown</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pb-5 border-b border-slate-100">
        <div className="md:col-span-2">
          <div className="text-xs font-medium text-slate-500">Verified Final Answer</div>
          <div className="mt-1 flex items-baseline gap-2 flex-wrap">
            <span className="text-2xl font-mono font-semibold tabular-nums text-slate-900">
              {report.final_answer}
            </span>
            {report.units && report.units !== 'dimensionless' && (
              <span className="text-xs font-mono text-slate-500">{report.units}</span>
            )}
          </div>
          <p className="text-xs text-slate-600 mt-2">{report.solution_methodology}</p>
        </div>
        <div className="space-y-1.5 text-xs font-mono tabular-nums">
          <div className="flex justify-between">
            <span className="text-slate-500">Numeric Value</span>
            <span className="text-slate-900">
              {report.numeric_value !== null ? report.numeric_value : 'Symbolic'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Complexity</span>
            <span className="text-slate-900">{report.complexity_level ?? 'STANDARD'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Elapsed Time</span>
            <span className="text-slate-900">
              {report.elapsed_ms !== undefined
                ? report.elapsed_ms >= 1000
                  ? `${(report.elapsed_ms / 1000).toFixed(2)} s`
                  : `${report.elapsed_ms.toFixed(1)} ms`
                : '—'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Correction Loops</span>
            <span className="text-slate-900">{report.correction_attempts}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Tool Calls</span>
            <span className="text-slate-900">{report.tool_execution_history.length}</span>
          </div>
        </div>
      </div>

      {report.extracted_variables.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-semibold text-slate-900">Extracted Variables</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3 font-medium">Symbol</th>
                  <th className="py-2 pr-3 font-medium">Quantity</th>
                  <th className="py-2 pr-3 font-medium">Value</th>
                  <th className="py-2 font-medium">Unit</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {report.extracted_variables.map((v, i) => (
                  <tr key={`${v.symbol}_${i}`}>
                    <td className="py-2 pr-3 font-mono font-medium text-slate-900">{v.symbol}</td>
                    <td className="py-2 pr-3 text-slate-700">{v.name}</td>
                    <td className="py-2 pr-3 font-mono tabular-nums text-slate-900">
                      {v.value !== null ? String(v.value) : 'unknown'}
                    </td>
                    <td className="py-2 font-mono text-slate-600">{v.unit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="space-y-2">
        <h4 className="text-sm font-semibold text-slate-900">Scientific Derivation & Explanation</h4>
        <MathText content={report.explanation_markdown} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-100">
        <div>
          <h4 className="text-xs font-semibold text-slate-900 mb-2">Modeling Assumptions</h4>
          {report.assumptions.length > 0 ? (
            <ul className="space-y-1 text-xs text-slate-600">
              {report.assumptions.map((a, i) => (
                <li key={i}>• {a}</li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-slate-500">None beyond standard domain axioms.</p>
          )}
        </div>
        <div>
          <h4 className="text-xs font-semibold text-slate-900 mb-2">Confidence Limitations</h4>
          {report.confidence_limitations.length > 0 ? (
            <ul className="space-y-1 text-xs text-slate-600">
              {report.confidence_limitations.map((lim, i) => (
                <li key={i}>• {lim}</li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-slate-500">Verified within numerical solver tolerances.</p>
          )}
        </div>
      </div>
    </div>
  );
}
