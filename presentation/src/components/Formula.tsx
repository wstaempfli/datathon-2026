export function Formula() {
  return (
    <div className="formula">
{`target_position = clip( 1  +  `}<span className="hl">scaler</span>{` · ( −24·fh_return + 0.375·bmb_recent ), −2, +2 )

`}<span className="hl">scaler</span>{`         = clip( σ_ref / max(σ, 0.25·σ_ref), 0.5, 2.0 )
σ             = std( diff( log(close[0:50]) ) )       # per session
σ_ref         = median(σ) on train
bmb_recent    = Σ_h  sign(h) · exp( −(49 − bar_ix_h) / 20 )`}
    </div>
  );
}
