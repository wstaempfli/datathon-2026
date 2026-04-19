import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import data from "../data/session_example.json";

export function SessionTrajectory() {
  const series = data.close.map((c: number, i: number) => ({ bar: i, close: c }));
  const yMin = Math.min(...data.close);
  const yMax = Math.max(...data.close);
  const pad = (yMax - yMin) * 0.05 || 0.01;

  return (
    <div className="chart-card">
      <div className="chart-body">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={series} margin={{ top: 16, right: 24, bottom: 24, left: 8 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="2 6" />
            <XAxis
              dataKey="bar"
              type="number"
              domain={[0, 99]}
              ticks={[0, 10, 20, 30, 40, 49, 60, 70, 80, 99]}
              label={{ value: "bar_ix", position: "insideBottom", offset: -8, fill: "var(--muted)" }}
              stroke="var(--rail)"
            />
            <YAxis
              domain={[yMin - pad, yMax + pad]}
              tickFormatter={(v) => v.toFixed(2)}
              label={{ value: "close", angle: -90, position: "insideLeft", fill: "var(--muted)" }}
              stroke="var(--rail)"
            />
            <Tooltip
              formatter={(v: number) => v.toFixed(4)}
              labelFormatter={(l) => `bar ${l}`}
            />
            <ReferenceArea x1={49} x2={99} fill="var(--accent)" fillOpacity={0.08} />
            <ReferenceLine
              x={49}
              stroke="var(--accent)"
              strokeDasharray="6 4"
              label={{ value: "decide here →", position: "top", fill: "var(--accent)", fontWeight: 600 }}
            />
            <Line
              type="monotone"
              dataKey="close"
              stroke="var(--fg)"
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
