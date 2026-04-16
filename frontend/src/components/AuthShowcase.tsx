import { useEffect, useRef } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  hue: number;
  alpha: number;
}

const FEATURES_ZH = ["智能代码分析", "流式推理展示", "一键自动修复", "多模型支持"];
const FEATURES_EN = ["Smart Code Analysis", "Streaming Reasoning", "One-Click Repair", "Multi-Model Support"];

export function AuthShowcase({ locale }: { locale: "zh" | "en" }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = 0;
    let height = 0;
    const particles: Particle[] = [];
    const CONNECTION_DIST = 130;
    const PARTICLE_COUNT = 60;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas!.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas!.width = width * dpr;
      canvas!.height = height * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function seed() {
      particles.length = 0;
      for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.45,
          vy: (Math.random() - 0.5) * 0.45,
          r: Math.random() * 2 + 1,
          hue: Math.random() * 40 + 15,
          alpha: Math.random() * 0.5 + 0.35,
        });
      }
    }

    resize();
    seed();

    let time = 0;

    function draw() {
      time += 0.003;
      ctx!.clearRect(0, 0, width, height);

      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;
      }

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECTION_DIST) {
            const opacity = (1 - dist / CONNECTION_DIST) * 0.18;
            ctx!.strokeStyle = `rgba(255,160,80,${opacity})`;
            ctx!.lineWidth = 0.6;
            ctx!.beginPath();
            ctx!.moveTo(particles[i].x, particles[i].y);
            ctx!.lineTo(particles[j].x, particles[j].y);
            ctx!.stroke();
          }
        }
      }

      for (const p of particles) {
        const glow = Math.sin(time * 2 + p.hue) * 0.15 + 0.85;
        ctx!.beginPath();
        ctx!.arc(p.x, p.y, p.r * glow, 0, Math.PI * 2);
        ctx!.fillStyle = `hsla(${p.hue}, 90%, 68%, ${p.alpha * glow})`;
        ctx!.fill();
      }

      const cx = width * 0.5;
      const cy = height * 0.45;
      const pulseR = 45 + Math.sin(time * 1.5) * 6;
      const grad = ctx!.createRadialGradient(cx, cy, 0, cx, cy, pulseR * 3);
      grad.addColorStop(0, "rgba(255,133,82,0.18)");
      grad.addColorStop(0.5, "rgba(39,111,191,0.08)");
      grad.addColorStop(1, "rgba(109,211,206,0)");
      ctx!.fillStyle = grad;
      ctx!.beginPath();
      ctx!.arc(cx, cy, pulseR * 3, 0, Math.PI * 2);
      ctx!.fill();

      const orbR = pulseR;
      for (let k = 0; k < 3; k++) {
        const angle = time * (0.8 + k * 0.3) + (k * Math.PI * 2) / 3;
        const ox = cx + Math.cos(angle) * orbR * (1.6 + k * 0.3);
        const oy = cy + Math.sin(angle) * orbR * (1.0 + k * 0.15);
        const dotGrad = ctx!.createRadialGradient(ox, oy, 0, ox, oy, 5);
        const hues = [25, 200, 165];
        dotGrad.addColorStop(0, `hsla(${hues[k]}, 90%, 72%, 0.9)`);
        dotGrad.addColorStop(1, `hsla(${hues[k]}, 90%, 72%, 0)`);
        ctx!.fillStyle = dotGrad;
        ctx!.beginPath();
        ctx!.arc(ox, oy, 5, 0, Math.PI * 2);
        ctx!.fill();
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    rafRef.current = requestAnimationFrame(draw);

    const ro = new ResizeObserver(() => {
      resize();
      seed();
    });
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(rafRef.current);
      ro.disconnect();
    };
  }, []);

  const features = locale === "zh" ? FEATURES_ZH : FEATURES_EN;

  return (
    <section className="relative flex h-full flex-col items-center justify-center overflow-hidden rounded-[36px] border border-black/5 bg-slate-950 dark:border-white/10">
      <canvas
        ref={canvasRef}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />

      <div className="relative z-10 flex flex-col items-center px-6 text-center">
        <div className="font-display text-3xl font-bold tracking-tight text-white xl:text-4xl">
          AutoRepair
        </div>
        <div className="mt-1.5 text-sm tracking-[0.35em] uppercase text-white/40">
          Studio
        </div>

        <div className="mt-10 grid grid-cols-2 gap-3">
          {features.map((text, i) => (
            <div
              key={text}
              className="group rounded-2xl border border-white/[0.08] bg-white/[0.04] px-4 py-3 backdrop-blur-sm transition-all hover:border-white/20 hover:bg-white/[0.08]"
              style={{ animationDelay: `${i * 120}ms` }}
            >
              <div className="flex items-center gap-2.5">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{
                    background: ["#ff8552", "#276fbf", "#6dd3ce", "#fbbf24"][i],
                    boxShadow: `0 0 8px ${["#ff855280", "#276fbf80", "#6dd3ce80", "#fbbf2480"][i]}`,
                  }}
                />
                <span className="text-xs font-medium text-white/75">{text}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="pointer-events-none absolute inset-0 rounded-[36px] ring-1 ring-inset ring-white/[0.06]" />
    </section>
  );
}
