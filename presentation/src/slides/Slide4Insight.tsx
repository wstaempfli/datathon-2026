import { FormulaLatex } from "../components/FormulaLatex";
import { Slide } from "../components/Slide";

export function Slide4Insight({ total }: { total: number }) {
  return (
    <Slide
      pageNum={4}
      total={total}
      kicker="The formula"
      title="One position per session."
      subtitle="Long the drift. Fade the spike. Lean with the news. Scale by vol."
    >
      <FormulaLatex />
    </Slide>
  );
}
