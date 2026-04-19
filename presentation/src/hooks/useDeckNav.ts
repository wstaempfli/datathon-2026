import { useCallback, useEffect, useState } from "react";

export function useDeckNav(total: number) {
  const [index, setIndex] = useState(0);

  const next = useCallback(
    () => setIndex((i) => Math.min(total - 1, i + 1)),
    [total],
  );
  const prev = useCallback(
    () => setIndex((i) => Math.max(0, i - 1)),
    [],
  );
  const jumpTo = useCallback(
    (i: number) => setIndex(Math.max(0, Math.min(total - 1, i))),
    [total],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const forward = ["ArrowRight", "ArrowDown", " ", "PageDown", "j", "J"];
      const back = ["ArrowLeft", "ArrowUp", "PageUp", "k", "K"];
      if (forward.includes(e.key)) {
        e.preventDefault();
        next();
      } else if (back.includes(e.key)) {
        e.preventDefault();
        prev();
      } else if (e.key === "Home") {
        e.preventDefault();
        jumpTo(0);
      } else if (e.key === "End") {
        e.preventDefault();
        jumpTo(total - 1);
      } else if (/^[1-9]$/.test(e.key)) {
        const n = parseInt(e.key, 10) - 1;
        if (n < total) {
          e.preventDefault();
          jumpTo(n);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, jumpTo, total]);

  return { index, next, prev, jumpTo };
}
