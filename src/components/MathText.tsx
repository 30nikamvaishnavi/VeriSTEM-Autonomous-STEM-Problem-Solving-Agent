import React from 'react';
import katex from 'katex';

interface MathTextProps {
  content: string;
  className?: string;
}

function renderLatexSegment(latex: string, displayMode: boolean, key: string): React.ReactNode {
  const html = katex.renderToString(latex.trim(), {
    displayMode,
    throwOnError: false,
    trust: false,
    strict: false,
  });
  if (displayMode) {
    return (
      <div
        key={key}
        className="my-3 overflow-x-auto py-2 px-3 bg-slate-50 border border-slate-200 rounded-md text-slate-900"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }
  return (
    <span
      key={key}
      className="inline-block align-middle mx-0.5"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function renderInlineLine(line: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  // Match inline $...$ or **bold** or `code`
  const tokenRegex = /(\$\$[^$]+\$\$|\$[^$\n]+\$|\*\*[^*]+\*\*|`[^`]+`)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let idx = 0;

  while ((match = tokenRegex.exec(line)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(line.slice(lastIndex, match.index));
    }
    const token = match[0];
    const key = `${keyPrefix}_${idx++}`;
    if (token.startsWith('$$') && token.endsWith('$$')) {
      nodes.push(renderLatexSegment(token.slice(2, -2), true, key));
    } else if (token.startsWith('$') && token.endsWith('$')) {
      nodes.push(renderLatexSegment(token.slice(1, -1), false, key));
    } else if (token.startsWith('**') && token.endsWith('**')) {
      nodes.push(
        <strong key={key} className="font-semibold text-slate-900">
          {token.slice(2, -2)}
        </strong>,
      );
    } else if (token.startsWith('`') && token.endsWith('`')) {
      nodes.push(
        <code
          key={key}
          className="font-mono text-[13px] px-1.5 py-0.5 bg-slate-100 text-slate-800 rounded"
        >
          {token.slice(1, -1)}
        </code>,
      );
    }
    lastIndex = match.index + token.length;
  }
  if (lastIndex < line.length) {
    nodes.push(line.slice(lastIndex));
  }
  return nodes;
}

export function MathText({ content, className = '' }: MathTextProps) {
  // Split display math blocks $$...$$ that may span multiple lines first
  const blocks = content.split(/(\$\$[\s\S]*?\$\$)/g);

  return (
    <div className={`space-y-2.5 text-[15px] leading-relaxed text-slate-700 ${className}`}>
      {blocks.map((block, bIdx) => {
        if (block.startsWith('$$') && block.endsWith('$$')) {
          return renderLatexSegment(block.slice(2, -2), true, `disp_${bIdx}`);
        }
        const lines = block.split('\n');
        return lines.map((rawLine, lIdx) => {
          const line = rawLine.trim();
          if (!line) return null;
          const key = `line_${bIdx}_${lIdx}`;
          if (line.startsWith('### ')) {
            return (
              <h4 key={key} className="text-base font-semibold text-slate-900 pt-2">
                {renderInlineLine(line.slice(4), key)}
              </h4>
            );
          }
          if (line.startsWith('## ')) {
            return (
              <h3 key={key} className="text-lg font-semibold text-slate-900 pt-3">
                {renderInlineLine(line.slice(3), key)}
              </h3>
            );
          }
          if (line.startsWith('# ')) {
            return (
              <h2 key={key} className="text-xl font-semibold text-slate-900 pt-3">
                {renderInlineLine(line.slice(2), key)}
              </h2>
            );
          }
          if (line.startsWith('- ') || line.startsWith('* ')) {
            return (
              <div key={key} className="flex items-start gap-2.5 pl-1">
                <span className="text-slate-400 select-none">•</span>
                <div className="flex-1">{renderInlineLine(line.slice(2), key)}</div>
              </div>
            );
          }
          return <p key={key}>{renderInlineLine(line, key)}</p>;
        });
      })}
    </div>
  );
}
