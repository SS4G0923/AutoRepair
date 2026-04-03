export function ThinkingDots() {
  return (
    <div className="inline-flex items-center gap-1 text-sm text-white/55">
      <span className="inline-block h-2 w-2 animate-bounce rounded-full bg-current [animation-delay:-0.2s]" />
      <span className="inline-block h-2 w-2 animate-bounce rounded-full bg-current [animation-delay:-0.1s]" />
      <span className="inline-block h-2 w-2 animate-bounce rounded-full bg-current" />
    </div>
  );
}
