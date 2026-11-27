import { useState } from 'react';
import { ChevronDown, ChevronRight, Terminal } from 'lucide-react';
import type { ToolExecutionRecord } from '../types/veristem';

interface ToolEvidenceListProps {
  records: ToolExecutionRecord[];
}

export function ToolEvidenceList({ records }: ToolEvidenceListProps) {
  const [openIds, setOpenIds] = useState<Record<string, boolean>>({});

  const toggle = (id: string) => {
    setOpenIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  if (records.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg p-5">
        <h3 className="text-base font-semibold text-slate-900">Computational Tool Evidence</h3>
        <p className="text-sm text-slate-500 mt-2">
          Tool execution payloads and isolated Python sandbox outputs will appear here once execution begins.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5">
      <div className="flex items-center justify-between pb-3 border-b border-slate-200">
        <h3 className="text-base font-semibold text-slate-900">Computational Tool Evidence</h3>
        <span className="text-xs font-mono tabular-nums text-slate-500">
          {records.length} execution{records.length === 1 ? '' : 's'}
        </span>
      </div>

      <div className="divide-y divide-slate-100 mt-2">
        {records.map((rec) => {
          const isOpen = Boolean(openIds[rec.tool_call_id]);
          return (
            <div key={rec.tool_call_id} className="py-2.5">
              <button
                type="button"
                onClick={() => toggle(rec.tool_call_id)}
                className="w-full flex items-center justify-between gap-3 text-left py-1 hover:bg-slate-50 rounded px-1.5 transition-colors"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  {isOpen ? (
                    <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-slate-400 shrink-0" />
                  )}
                  <Terminal className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                  <span className="font-mono text-xs font-medium text-slate-900 truncate">
                    {rec.tool_name}
                  </span>
                  <span className="text-xs text-slate-400 font-mono">{rec.tool_call_id}</span>
                </div>
                <div className="flex items-center gap-3 shrink-0 text-xs font-mono tabular-nums">
                  <span className={rec.success ? 'text-emerald-700 font-medium' : 'text-rose-700 font-medium'}>
                    {rec.success ? 'OK' : 'ERROR'}
                  </span>
                  <span className="text-slate-500">{rec.duration_ms.toFixed(1)} ms</span>
                </div>
              </button>

              {isOpen && (
                <div className="mt-2 ml-7 space-y-2 text-xs">
                  <div>
                    <div className="text-slate-500 font-medium mb-1">Input Payload</div>
                    <pre className="p-2.5 bg-slate-50 border border-slate-200 rounded font-mono text-[11px] text-slate-800 overflow-x-auto">
                      {JSON.stringify(rec.input_payload, null, 2)}
                    </pre>
                  </div>
                  <div>
                    <div className="text-slate-500 font-medium mb-1">
                      {rec.success ? 'Output Payload' : 'Error'}
                    </div>
                    <pre className="p-2.5 bg-slate-50 border border-slate-200 rounded font-mono text-[11px] text-slate-800 overflow-x-auto">
                      {rec.success
                        ? JSON.stringify(rec.output_payload, null, 2)
                        : rec.error_message}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
