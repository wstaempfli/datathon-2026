import { useEffect, useRef, useState } from "react";
import { useDeckNav } from "./hooks/useDeckNav";
import { Slide1Title } from "./slides/Slide1Title";
import { Slide2Challenge } from "./slides/Slide2Challenge";
import { Slide3Tried } from "./slides/Slide3Tried";
import { Slide4Insight } from "./slides/Slide4Insight";
import { Slide5Solution } from "./slides/Slide5Solution";

const SLIDE_W = 1920;
const SLIDE_H = 1080;

function useFitScale() {
  const [scale, setScale] = useState(1);
  useEffect(() => {
    const compute = () => {
      const s = Math.min(window.innerWidth / SLIDE_W, window.innerHeight / SLIDE_H);
      setScale(s);
    };
    compute();
    window.addEventListener("resize", compute);
    return () => window.removeEventListener("resize", compute);
  }, []);
  return scale;
}

export default function App() {
  const total = 5;
  const { index } = useDeckNav(total);
  const scale = useFitScale();

  const Slides = [Slide1Title, Slide2Challenge, Slide3Tried, Slide4Insight, Slide5Solution];
  const Current = Slides[index];

  return (
    <div className="deck">
      <div className="stage">
        <div
          className="stage-inner"
          style={{
            transform: `scale(${scale})`,
            width: `${SLIDE_W}px`,
            height: `${SLIDE_H}px`,
          }}
        >
          <Current total={total} />
        </div>
      </div>

      <div className="print-stack">
        {Slides.map((S, i) => (
          <S key={i} total={total} />
        ))}
      </div>

      <div className="nav-indicator">
        {(index + 1).toString().padStart(2, "0")} / {total.toString().padStart(2, "0")}
      </div>
      <div className="hint">← → or 1-5 to navigate · ⌘P to export PDF</div>
    </div>
  );
}
