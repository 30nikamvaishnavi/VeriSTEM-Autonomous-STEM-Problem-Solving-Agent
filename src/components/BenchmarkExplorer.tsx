import { useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import type { BenchmarkCase, ProblemDomain } from '../types/veristem';

interface BenchmarkExplorerProps {
  benchmarks: BenchmarkCase[];
  onLoadProblem: (question: string, domain: ProblemDomain) => void;
}

const DOMAIN_FILTERS: Array<{ key: 'all' | ProblemDomain; label: string }> = [
  { key: 'all', label: 'All Domains' },
  { key: 'mathematics', label: 'Mathematics' },
  { key: 'physics', label: 'Physics' },
  { key: 'quantitative_finance', label: 'Quantitative Finance' },
];

export function BenchmarkExplorer({ benchmarks, onLoadProblem }: BenchmarkExplorerProps) {
  const [domainFilter, setDomainFilter] = useState<'all' | ProblemDomain>('all');

  const filtered =
    domainFilter === 'all'
      ? benchmarks
      : benchmarks.filter((b) => b.domain === domainFilter);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-900" style={{ textWrap: 'balance' }}>
            32-Problem Ground-Truth Benchmark Suite
          </h2>
          <p className="text-sm text-slate-600 mt-1">
            Independently validated reference problems across Mathematics, Physics, and Quantitative Finance.
          </p>
        </div>

        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
          {DOMAIN_FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              onClick={() => setDomainFilter(f.key)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                domainFilter === f.key
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-500">
                <th className="py-3 px-4 font-medium">ID</th>
                <th className="py-3 px-4 font-medium">Domain · Category</th>
                <th className="py-3 px-4 font-medium">Problem Statement</th>
                <th className="py-3 px-4 font-medium text-right">Ground Truth</th>
                <th className="py-3 px-4 font-medium">Verification Method</th>
                <th className="py-3 px-4 font-medium text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filtered.map((item) => (
                <tr key={item.id} className="hover:bg-slate-50/70">
                  <td className="py-3 px-4 font-mono font-medium text-slate-900 whitespace-nowrap">
                    {item.id}
                  </td>
                  <td className="py-3 px-4 text-slate-600 whitespace-nowrap">
                    {item.domain.replace('_', ' ')} · {item.category.replace('_', ' ')}
                  </td>
                  <td className="py-3 px-4 text-slate-800 max-w-md">{item.question}</td>
                  <td className="py-3 px-4 font-mono tabular-nums text-right text-slate-900 whitespace-nowrap">
                    {item.expected_answer} {item.expected_unit !== 'dimensionless' ? item.expected_unit : ''}
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-600 whitespace-nowrap">
                    {item.verification_method}
                  </td>
                  <td className="py-3 px-4 text-right whitespace-nowrap">
                    <button
                      type="button"
                      onClick={() => onLoadProblem(item.question, item.domain)}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded transition-colors whitespace-nowrap shrink-0"
                    >
                      <span>Solve</span>
                      <ArrowUpRight className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
