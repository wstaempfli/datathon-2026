declare module "react-katex" {
  import * as React from "react";

  export interface KatexProps {
    math: string;
    renderError?: (error: Error) => React.ReactNode;
    errorColor?: string;
    settings?: object;
  }

  export const BlockMath: React.FC<KatexProps>;
  export const InlineMath: React.FC<KatexProps>;
}
