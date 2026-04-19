import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import models from "../data/models.json";

type Row = { name: string; features: string; lb: number; isFinal: boolean };

export function ModelBar() {
  const sorted: Row[] = [...models.models].sort((a, b) => a.lb - b.lb);
  const rows = sorted.map((r) => ({ ...r, label: `${r.name} · ${r.features}` }));

  return (
    <div className="chart-card">
      <h3>Kaggle public-LB Sharpe per approach · V1b in accent blue</h3>
      <div className="chart-body">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={rows}
            layout="vertical"
            margin={{ top: 8, right: 80, bottom: 8, left: 24 }}
          >
            <XAxis
              type="number"
              domain={[0, 3.2]}
              tickFormatter={(v) => v.toFixed(1)}
              stroke="var(--rail)"
            />
            <YAxis
              dataKey="label"
              type="category"
              width={340}
              interval={0}
              tick={{ fontSize: 18, fill: "var(--fg)" }}
              stroke="var(--rail)"
            />
            <Tooltip formatter={(v: number) => v.toFixed(3)} />
            <Bar dataKey="lb" barSize={28} isAnimationActive={false} radius={[0, 6, 6, 0]}>
              {rows.map((r, i) => (
                <Cell
                  key={i}
                  fill={r.isFinal ? "var(--accent)" : "var(--muted)"}
                  fillOpacity={r.isFinal ? 1 : 0.55}
                />
              ))}
              <LabelList
                dataKey="lb"
                position="right"
                formatter={(v: number) => v.toFixed(3)}
                style={{ fontSize: 18, fontWeight: 600, fill: "var(--fg)" }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
