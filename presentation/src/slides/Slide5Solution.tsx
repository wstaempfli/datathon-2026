import { CvBar } from "../components/CvBar";
import { Slide } from "../components/Slide";

export function Slide5Solution({ total }: { total: number }) {
  return (
    <Slide
      pageNum={5}
      total={total}
      kicker="Results"
      title="Drift-preserving wins."
      subtitle="5-fold contiguous CV on train. V1b dominates on both mean and worst-fold."
    >
      <div className="results-summary">
        <div className="metric">
          <span className="label">CV Mean</span>
          <span className="value">3.13</span>
        </div>
        <div className="metric">
          <span className="label">CV Min</span>
          <span className="value">2.18</span>
        </div>
        <div className="metric muted">
          <span className="label">Public LB</span>
          <span className="value">2.81</span>
        </div>
      </div>
      <div className="plot-frame">
        <CvBar />
      </div>
    </Slide>
  );
}
