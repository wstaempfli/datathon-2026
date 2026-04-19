import {
  CartesianGrid,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ComposedChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import scatter from "../data/scatter.json";

type Pt = { fh: number; target: number };

export function FhScatter() {
  const pts: Pt[] = scatter.points;
  const { regression, corr } = scatter;
  const line = [
    { fh: regression.x_min, target: regression.y_at_xmin, isLine: 1 },
    { fh: regression.x_max, target: regression.y_at_xmax, isLine: 1 },
  ];

  return (
    <div className="chart-card">
      <h3>
        fh_return (bar 0→49) vs target_return (bar 49→99) · corr = <span className="neg">{corr.toFixed(3)}</span>
      </h3>
      <div className="chart-body">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 16, right: 24, bottom: 32, left: 16 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="2 6" />
            <XAxis
              type="number"
              dataKey="fh"
              domain={["auto", "auto"]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              label={{ value: "fh_return (first-half)", position: "insideBottom", offset: -16, fill: "var(--muted)" }}
              stroke="var(--rail)"
            />
            <YAxis
              type="number"
              dataKey="target"
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              label={{ value: "target_return (hold 49→99)", angle: -90, position: "insideLeft", fill: "var(--muted)" }}
              stroke="var(--rail)"
            />
            <ZAxis range={[32, 32]} />
            <Tooltip
              formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
              cursor={{ strokeDasharray: "3 3" }}
            />
            <ReferenceLine x={0} stroke="var(--rail)" />
            <ReferenceLine y={0} stroke="var(--rail)" />
            <Scatter data={pts} fill="var(--accent)" fillOpacity={0.35} isAnimationActive={false} />
            <Line
              data={line}
              dataKey="target"
              stroke="var(--neg)"
              strokeWidth={3}
              dot={false}
              isAnimationActive={false}
              legendType="none"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
