import { SessionTrajectory } from "../components/SessionTrajectory";
import { Slide } from "../components/Slide";

export function Slide2Challenge({ total }: { total: number }) {
  return (
    <Slide
      pageNum={2}
      total={total}
      kicker="The Challenge"
      title="See 50 bars. Pick a position. Live with it."
      subtitle="We see the first half; the market decides the rest. Sharpe rewards consistency over heroics."
    >
      <div className="slide-body">
        <div className="slide-col" style={{ flex: "0 0 640px" }}>
          <div className="stats" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <div className="stat">
              <div className="label">Sessions (test)</div>
              <div className="value">20,000</div>
            </div>
            <div className="stat">
              <div className="label">Sessions (train)</div>
              <div className="value">1,000</div>
            </div>
            <div className="stat">
              <div className="label">Bars seen</div>
              <div className="value">0 → 49</div>
            </div>
            <div className="stat">
              <div className="label">Bars held</div>
              <div className="value">49 → 99</div>
            </div>
          </div>
          <ul className="bullets">
            <li>Decide <strong>target_position ∈ [−2, +2]</strong> at bar 49 close.</li>
            <li>Liquidate at bar 99 close. PnL = position · (c99/c49 − 1).</li>
            <li>
              Score = <span className="mono">mean(PnL) / std(PnL) · 16</span> — consistency beats accuracy.
            </li>
            <li>~10 headlines per session — most are noise.</li>
          </ul>
        </div>
        <div className="slide-col">
          <SessionTrajectory />
        </div>
      </div>
    </Slide>
  );
}
