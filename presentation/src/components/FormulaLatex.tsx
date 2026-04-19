import { BlockMath } from "react-katex";

const POSITION = String.raw`
\hat{p}_t \;=\; \operatorname{clip}\!\Bigl(\,1 \;+\; s_t\cdot\bigl(-24\,r^{\text{fh}}_{t} \;+\; 0.375\,b_t\bigr),\; -2,\; +2\Bigr)
`;

const SCALER = String.raw`
s_t \;=\; \operatorname{clip}\!\left(\dfrac{\sigma_{\text{ref}}}{\max\!\bigl(\sigma_t,\; 0.25\,\sigma_{\text{ref}}\bigr)},\; 0.5,\; 2.0\right)
`;

const SIGMA = String.raw`
\sigma_t \;=\; \operatorname{std}\!\bigl(\Delta \log c_{0:50}\bigr)
\qquad\quad
\sigma_{\text{ref}} \;=\; \operatorname{median}_{\,t\in\text{train}}\bigl(\sigma_t\bigr)
`;

const BMB = String.raw`
b_t \;=\; \sum_{h\,\in\,\mathcal{H}_t}\operatorname{sign}(h)\,\exp\!\left(-\dfrac{49 - i_h}{20}\right)
`;

type Note = { tex: string; text: string };

const NOTES: Note[] = [
  { tex: String.raw`1`,                  text: "always a little long — markets drift up" },
  { tex: String.raw`-24\,r^{\text{fh}}_t`, text: "fade the first-half move" },
  { tex: String.raw`+0.375\,b_t`,        text: "lean with recent headlines" },
  { tex: String.raw`s_t`,                text: "calm → bigger bet · spiky → smaller" },
];

export function FormulaLatex() {
  return (
    <div className="formula-grid">
      <div className="formula-main">
        <BlockMath math={POSITION} />
        <BlockMath math={SCALER} />
        <BlockMath math={SIGMA} />
        <BlockMath math={BMB} />
      </div>
      <ul className="annotations">
        {NOTES.map((n, i) => (
          <li key={i}>
            <span className="annotation-term">
              <BlockMath math={n.tex} />
            </span>
            <span className="annotation-text">{n.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
