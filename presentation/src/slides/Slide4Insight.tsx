import { FhScatter } from "../components/FhScatter";
import { QuintileBar } from "../components/QuintileBar";
import { Slide } from "../components/Slide";

export function Slide4Insight({ total }: { total: number }) {
  return (
    <Slide
      pageNum={4}
      total={total}
      kicker="The Insight"
      title="Mean-Reversion + News"
      subtitle="Weak individual edge, but consistent across 1,000 sessions and survives vol scaling."
    >
      <div className="slide-body" style={{ flexDirection: "column", gap: 24 }}>
        <div className="stats">
          <div className="stat blue">
            <div className="label">corr(fh, target)</div>
            <div className="value">−0.069</div>
          </div>
          <div className="stat">
            <div className="label">Q1 (worst fh) target</div>
            <div className="value pos">+0.54%</div>
          </div>
          <div className="stat">
            <div className="label">Q5 (best fh) target</div>
            <div className="value">+0.13%</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 24, flex: 1, minHeight: 0 }}>
          <div style={{ flex: 1.2, display: "flex", minHeight: 0 }}>
            <FhScatter />
          </div>
          <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
            <QuintileBar />
          </div>
        </div>
      </div>
    </Slide>
  );
}
