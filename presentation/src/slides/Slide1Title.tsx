import { Slide } from "../components/Slide";

export function Slide1Title({ total }: { total: number }) {
  return (
    <Slide pageNum={1} total={total} className="title-slide">
      <div className="kicker">ETH Zurich · HRT Datathon 2026</div>
      <h1>
        Fade the Spike
        <br />
        <span className="dim">Mean-Reversion + News</span>
      </h1>
      <div className="team">Team Üetlibytes</div>
      <div className="members">
        Wanja Stämpfli · Julio Schneider · Kenji Nakano · David Frei
      </div>
      <div className="badges">
        <div className="badge">
          <span className="label">Kaggle LB</span>
          <span className="value">2.811</span>
        </div>
        <div className="badge">
          <span className="label">CV Mean</span>
          <span className="value">3.13</span>
        </div>
        <div className="badge">
          <span className="label">CV Min</span>
          <span className="value">2.18</span>
        </div>
      </div>
    </Slide>
  );
}
