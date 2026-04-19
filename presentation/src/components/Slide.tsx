import { PropsWithChildren } from "react";

type Props = PropsWithChildren<{
  pageNum: number;
  total: number;
  kicker?: string;
  title?: string;
  subtitle?: string;
  className?: string;
}>;

export function Slide({ pageNum, total, kicker, title, subtitle, children, className }: Props) {
  return (
    <section className={`slide ${className ?? ""}`}>
      {(kicker || pageNum) && (
        <div className="slide-header">
          <span>{kicker ?? ""}</span>
          <span>
            Üetlibytes · Datathon 2026 · {pageNum.toString().padStart(2, "0")} / {total.toString().padStart(2, "0")}
          </span>
        </div>
      )}
      {title && <h1 className="slide-title">{title}</h1>}
      {subtitle && <p className="slide-subtitle">{subtitle}</p>}
      {children}
    </section>
  );
}
