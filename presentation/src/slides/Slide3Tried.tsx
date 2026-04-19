import { FhScatter } from "../components/FhScatter";
import { QuintileBar } from "../components/QuintileBar";
import { Slide } from "../components/Slide";

export function Slide3Tried({ total }: { total: number }) {
  return (
    <Slide
      pageNum={3}
      total={total}
      kicker="The insight"
      title="Spikes revert."
      subtitle={
        <>
          First-half return vs. held return · <span className="accent">corr = −0.069</span> across 1,000 sessions.
        </>
      }
    >
      <div className="slide-body">
        <div className="slide-col">
          <FhScatter />
        </div>
        <div className="slide-col">
          <QuintileBar />
        </div>
      </div>
    </Slide>
  );
}
