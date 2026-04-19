import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import quintile from "../data/quintile.json";

export function QuintileBar() {
  const bins = quintile.bins.map((b) => ({
    ...b,
    target_pct: b.target_mean * 100,
  }));
  const overall = quintile.overall_target_mean * 100;

  return (
    <div className="chart-card">
      <div className="chart-body">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={bins} margin={{ top: 24, right: 24, bottom: 24, left: 16 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="2 6" />
            <XAxis
              dataKey="quintile"
              tick={{ fontSize: 18, fill: "var(--fg)" }}
              stroke="var(--rail)"
            />
            <YAxis
              tickFormatter={(v) => `${v.toFixed(2)}%`}
              stroke="var(--rail)"
              domain={[0, 0.7]}
            />
            <Tooltip
              formatter={(v: number) => `${v.toFixed(3)}%`}
            />
            <ReferenceLine
              y={overall}
              stroke="var(--muted)"
              strokeDasharray="4 6"
              label={{
                value: `overall mean ${overall.toFixed(2)}%`,
                position: "right",
                fill: "var(--muted)",
                fontSize: 14,
              }}
            />
            <Bar dataKey="target_pct" barSize={88} isAnimationActive={false} radius={[8, 8, 0, 0]}>
              {bins.map((_, i) => (
                <Cell key={i} fill={i === 0 ? "var(--accent)" : "var(--muted)"} fillOpacity={i === 0 ? 1 : 0.45} />
              ))}
              <LabelList
                dataKey="target_pct"
                position="top"
                formatter={(v: number) => `${v.toFixed(2)}%`}
                style={{ fontSize: 18, fontWeight: 700, fill: "var(--fg)" }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
