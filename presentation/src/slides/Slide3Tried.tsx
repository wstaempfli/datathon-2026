import { ModelBar } from "../components/ModelBar";
import { Slide } from "../components/Slide";

export function Slide3Tried({ total }: { total: number }) {
  return (
    <Slide
      pageNum={3}
      total={total}
      kicker="What we tried"
      title="We tried everything. The simple thing won."
      subtitle="10 models × 4 feature sets over 3 days. None of the ML tricks consistently beat a hand-tuned linear rule."
    >
      <div className="slide-body">
        <div className="slide-col" style={{ flex: "0 0 640px" }}>
          <ul className="bullets">
            <li>
              <strong>Gradient boosting</strong> — XGBoost, LightGBM, sklGBR.
            </li>
            <li>
              <strong>Linear family</strong> — Ridge, Lasso, ElasticNet, Huber, Quantile.
            </li>
            <li>
              <strong>Feature sets</strong> — <span className="mono">price5</span>,{" "}
              <span className="mono">numeric3</span>, <span className="mono">gate1top10</span>,{" "}
              <span className="mono">all12</span>.
            </li>
            <li>
              Only <strong>XGBoost · price5+gate1top10</strong> (LB 2.93) matched the rule on public LB —
              but it was unstable across CV folds.
            </li>
            <li className="accent">
              Lesson: with 1,000 train sessions, flexible models memorise noise. The rule generalises.
            </li>
          </ul>
        </div>
        <div className="slide-col">
          <ModelBar />
        </div>
      </div>
    </Slide>
  );
}
