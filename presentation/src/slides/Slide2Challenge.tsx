import { ModelBar } from "../components/ModelBar";
import { Slide } from "../components/Slide";

export function Slide2Challenge({ total }: { total: number }) {
  return (
    <Slide
      pageNum={2}
      total={total}
      kicker="What we tried"
      title="We tried everything."
      subtitle="A hand-tuned linear rule quietly beat every ML model we threw at it."
    >
      <div className="plot-frame">
        <ModelBar />
      </div>
    </Slide>
  );
}
