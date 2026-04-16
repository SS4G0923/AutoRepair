import { useState, useRef, useEffect } from "react";

interface DropdownOption {
  label: string;
  value: string;
}

interface DropdownProps {
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
  menuClassName?: string;
  triggerLabelClassName?: string;
  optionLabelClassName?: string;
  placeholder?: string;
}

export function Dropdown({
  value,
  options,
  onChange,
  disabled,
  className = "",
  menuClassName = "",
  triggerLabelClassName = "",
  optionLabelClassName = "",
  placeholder,
}: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const selectedOption = options.find((o) => o.value === value);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className={`relative min-w-0 ${className}`} ref={containerRef}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full min-w-0 items-center justify-between gap-2 rounded-full border border-black/10 bg-white/50 px-3 py-1.5 text-sm text-slate-900 transition hover:bg-white/80 disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-white/5 dark:text-white dark:hover:bg-white/10"
      >
        <span className={`min-w-0 flex-1 ${triggerLabelClassName || "truncate"}`}>
          {selectedOption ? selectedOption.label : placeholder || value}
        </span>
        <svg
          className={`h-4 w-4 shrink-0 text-slate-500 transition-transform dark:text-slate-400 ${isOpen ? "rotate-180" : ""}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && !disabled && (
        <div
          className={`absolute left-0 top-[calc(100%+4px)] z-50 max-h-60 min-w-full max-w-[min(18rem,calc(100vw-2rem))] overflow-y-auto rounded-2xl border border-black/10 bg-white p-1 shadow-lg [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden dark:border-white/10 dark:bg-slate-900 ${menuClassName}`.trim()}
        >
          {options.length > 0 ? (
            options.map((option) => (
              <button
                key={option.value}
                type="button"
                onClick={() => {
                  onChange(option.value);
                  setIsOpen(false);
                }}
                className={`w-full rounded-xl px-3 py-2 text-left text-sm transition ${
                  value === option.value
                    ? "bg-slate-100 text-slate-900 dark:bg-white/10 dark:text-white"
                    : "text-slate-700 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-white/5 dark:hover:text-white"
                }`}
              >
                <span className={`block ${optionLabelClassName || "whitespace-normal break-words"}`}>{option.label}</span>
              </button>
            ))
          ) : (
            <div className="px-3 py-2 text-sm text-slate-500 dark:text-slate-400">
              {placeholder || "No options"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
