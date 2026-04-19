import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import cv from "../data/cv.json";

export function CvBar() {
  const rows = cv.folds.map((f, i) => ({
    fold: `Fold ${f}`,
    baseline: cv.baseline[i],
    v1b: cv.v1b[i],
  }));

  return (
    <div className="chart-card">
      <div className="chart-body">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 24, right: 24, bottom: 24, left: 16 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="2 6" />
            <XAxis dataKey="fold" tick={{ fontSize: 18, fill: "var(--fg)" }} stroke="var(--rail)" />
            <YAxis
              tickFormatter={(v) => v.toFixed(1)}
              stroke="var(--rail)"
              domain={[0, 6]}
            />
            <Tooltip formatter={(v: number) => v.toFixed(3)} />
            <Legend
              wrapperStyle={{ paddingTop: 8, fontSize: 16 }}
              formatter={(v) => (v === "v1b" ? "V1b (drift-preserving)" : "Baseline (no vol scale)")}
            />
            <ReferenceLine y={cv.v1b_min} stroke="var(--accent)" strokeDasharray="4 6" />
            <Bar dataKey="baseline" fill="var(--muted)" fillOpacity={0.55} barSize={42} isAnimationActive={false} radius={[6, 6, 0, 0]}>
              <LabelList dataKey="baseline" position="top" formatter={(v: number) => v.toFixed(2)} style={{ fontSize: 14, fill: "var(--muted)" }} />
            </Bar>
            <Bar dataKey="v1b" fill="var(--accent)" barSize={42} isAnimationActive={false} radius={[6, 6, 0, 0]}>
              <LabelList dataKey="v1b" position="top" formatter={(v: number) => v.toFixed(2)} style={{ fontSize: 14, fontWeight: 700, fill: "var(--accent)" }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
