import { Slide } from "../components/Slide";

export function Slide1Title({ total }: { total: number }) {
  return (
    <Slide pageNum={1} total={total} className="title-slide">
      <div className="kicker">ETH Zurich · HRT Datathon 2026</div>
      <h1>
        Fade the Spike
        <span className="dim">Mean-Reversion + News</span>
      </h1>
      <div className="team">Üetlibytes</div>
      <div className="members">
        Wanja Stämpfli · Julio Schneider · Kenji Nakano · David Frei
      </div>
      <div className="lb-mark">
        <span className="label">Public Leaderboard</span>
        <span className="value">2.811</span>
      </div>
    </Slide>
  );
}
