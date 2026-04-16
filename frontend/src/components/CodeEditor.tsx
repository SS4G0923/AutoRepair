import { useEffect, useRef } from "react";

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}

export function CodeEditor({ value, onChange, placeholder }: CodeEditorProps) {
  const lineCount = Math.max(value.split("\n").length, 12);
  const gutterRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    const gutter = gutterRef.current;
    if (!textarea || !gutter) {
      return;
    }
    const syncScroll = () => {
      gutter.scrollTop = textarea.scrollTop;
    };
    textarea.addEventListener("scroll", syncScroll);
    return () => textarea.removeEventListener("scroll", syncScroll);
  }, []);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-[28px] border border-white/10 bg-ink-900/95">
      <div className="flex shrink-0 items-center justify-between border-b border-white/10 px-5 py-3">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full bg-ember" />
          <span className="h-3 w-3 rounded-full bg-amber-300" />
          <span className="h-3 w-3 rounded-full bg-aurora" />
        </div>
        <div className="font-mono text-xs uppercase tracking-[0.28em] text-white/55">editor</div>
      </div>
      <div className="flex min-h-0 flex-1 bg-[radial-gradient(circle_at_top,_rgba(109,211,206,0.12),_transparent_35%),linear-gradient(180deg,rgba(255,255,255,0.03),transparent)]">
        <div
          ref={gutterRef}
          className="hidden h-full min-h-0 w-16 shrink-0 overflow-hidden border-r border-white/6 bg-white/5 px-4 py-5 text-right font-mono text-xs leading-6 text-white md:block"
        >
          {Array.from({ length: lineCount }, (_, index) => (
            <div key={index} className="h-6 leading-6">
              {index + 1}
            </div>
          ))}
        </div>
        <textarea
          ref={textareaRef}
          spellCheck={false}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          className="min-h-0 w-full flex-1 resize-none overflow-y-auto bg-transparent px-5 py-5 font-mono text-[14px] leading-6 text-white outline-none placeholder:text-white/25"
        />
      </div>
    </div>
  );
}
