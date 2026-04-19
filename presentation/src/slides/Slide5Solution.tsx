import { CvBar } from "../components/CvBar";
import { Formula } from "../components/Formula";
import { Slide } from "../components/Slide";

export function Slide5Solution({ total }: { total: number }) {
  return (
    <Slide
      pageNum={5}
      total={total}
      kicker="The Solution"
      title="V1b · Drift-preserving risk-parity"
      subtitle="Long the drift by default. Scale only the directional bet by realised volatility. News nudges the sign."
    >
      <div className="slide-body" style={{ flexDirection: "column", gap: 24 }}>
        <Formula />
        <div style={{ display: "flex", gap: 24, flex: 1, minHeight: 0 }}>
          <div className="slide-col" style={{ flex: "0 0 640px" }}>
            <ul className="bullets">
              <li>
                <strong>+1 intercept</strong> captures the 57% positive-drift prior — never scaled.
              </li>
              <li>
                <strong>−24 · fh</strong> fades the first-half move; <strong>+0.375 · bmb</strong>{" "}
                tilts by recent bull/bear headline polarity (τ = 20).
              </li>
              <li>
                <strong>Risk parity</strong>: <span className="mono">scaler = σ_ref / σ</span>, clipped to
                [0.5, 2.0]. Calm sessions bet bigger, spiky sessions bet smaller.
              </li>
              <li className="accent">
                Strictly dominates the unscaled baseline on both CV mean (3.13 vs 3.12) and min (2.18 vs 2.02).
              </li>
            </ul>
          </div>
          <div className="slide-col">
            <CvBar />
          </div>
        </div>
      </div>
    </Slide>
  );
}
