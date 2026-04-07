import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  DEFAULT_SIDEBAR_WIDTH,
  MAX_SIDEBAR_WIDTH,
  MIN_SIDEBAR_WIDTH,
  SIDEBAR_WIDTH_STORAGE_KEY,
} from "./utils";

export function useSidebarLayout() {
  const [sidebarWidthPx, setSidebarWidthPx] = useState(DEFAULT_SIDEBAR_WIDTH);
  const [sidebarResizing, setSidebarResizing] = useState(false);
  const [isDesktopLayout, setIsDesktopLayout] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(min-width: 1024px)").matches,
  );

  const sidebarWidthRef = useRef(DEFAULT_SIDEBAR_WIDTH);
  const sidebarResizeStartRef = useRef({ x: 0, w: DEFAULT_SIDEBAR_WIDTH });

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY);
      if (!raw) {
        return;
      }
      const width = Number(raw);
      if (Number.isNaN(width)) {
        return;
      }
      const clamped = Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, width));
      setSidebarWidthPx(clamped);
      sidebarWidthRef.current = clamped;
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    sidebarWidthRef.current = sidebarWidthPx;
  }, [sidebarWidthPx]);

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    setIsDesktopLayout(mq.matches);
    const onChange = () => setIsDesktopLayout(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  useEffect(() => {
    if (!sidebarResizing) {
      return;
    }

    const onMove = (event: PointerEvent) => {
      const maxWidth = Math.min(MAX_SIDEBAR_WIDTH, Math.floor(window.innerWidth * 0.55));
      const nextWidth = Math.min(
        maxWidth,
        Math.max(
          MIN_SIDEBAR_WIDTH,
          sidebarResizeStartRef.current.w + (event.clientX - sidebarResizeStartRef.current.x),
        ),
      );
      setSidebarWidthPx(nextWidth);
      sidebarWidthRef.current = nextWidth;
    };

    const onUp = () => {
      setSidebarResizing(false);
      try {
        window.localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(sidebarWidthRef.current));
      } catch {
        /* ignore */
      }
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [sidebarResizing]);

  useEffect(() => {
    if (!sidebarResizing) {
      return;
    }
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [sidebarResizing]);

  function handleSidebarResizePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!isDesktopLayout) {
      return;
    }
    event.preventDefault();
    sidebarResizeStartRef.current = { x: event.clientX, w: sidebarWidthPx };
    sidebarWidthRef.current = sidebarWidthPx;
    setSidebarResizing(true);
  }

  function handleSidebarResizeReset() {
    setSidebarWidthPx(DEFAULT_SIDEBAR_WIDTH);
    sidebarWidthRef.current = DEFAULT_SIDEBAR_WIDTH;
    try {
      window.localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(DEFAULT_SIDEBAR_WIDTH));
    } catch {
      /* ignore */
    }
  }

  return {
    sidebarWidthPx,
    sidebarResizing,
    isDesktopLayout,
    handleSidebarResizePointerDown,
    handleSidebarResizeReset,
  };
}
