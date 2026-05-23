import Link from "next/link";
import "./landing.css";

export function LandingShell() {
  return (
    <div className="medevo-home">
      <main className="home-page">
        <nav className="nav">
          <div className="brand">
            <div className="mark" />
            <div className="name">
              MedEvo<span className="ver">v0.4</span>
            </div>
          </div>
          <div className="nav-links">
            <a href="#how">How it works</a>
            <a href="#values">Values</a>
            <a href="#runs">Runs</a>
            <Link
              href="/run"
              className="cta-primary"
              style={{ padding: "10px 18px", fontSize: "13px" }}
            >
              Replay cached simulation <span className="arrow">→</span>
            </Link>
          </div>
        </nav>

        <section className="hero">
          <div className="eyebrow">
            <span className="dot" />A research instrument · not a forecast
          </div>
          <h1>
            Replay how AI-generated studies drift through an evidence
            pipeline.
          </h1>
          <p className="lede">
            MedEvo is a static public replay of a research instrument: 30
            clinical claims, six domains, historical evidence horizons, one FREE
            branch, and one CIVER/BRIM-gated branch.
          </p>
          <div className="hero-cta">
            <Link href="/run" className="cta-primary">
              Replay cached simulation <span className="arrow">→</span>
            </Link>
            <a href="#how" className="cta-secondary">
              How it works
            </a>
          </div>
        </section>

        <section className="section" id="how">
          <div className="section-eyebrow">01 · how it works</div>
          <h2 className="section-h2">
            A guideline goes in. Two futures come back.
          </h2>
          <p className="section-lede">
            Paste a guideline excerpt or a paper&apos;s conclusion. The engine
            extracts each clinical claim and lets it evolve along two parallel
            branches across three horizons.
          </p>

          <div className="flow">
            <div className="flow-step">
              <svg className="glyph" viewBox="0 0 48 48" fill="none">
                <rect x="6" y="6" width="36" height="36" rx="3" stroke="#11231e" strokeWidth="1.4" />
                <line x1="12" y1="16" x2="36" y2="16" stroke="#11231e" strokeWidth="1.4" />
                <line x1="12" y1="22" x2="32" y2="22" stroke="#11231e" strokeWidth="1.4" />
                <line x1="12" y1="28" x2="34" y2="28" stroke="#11231e" strokeWidth="1.4" />
                <line x1="12" y1="34" x2="26" y2="34" stroke="#11231e" strokeWidth="1.4" />
              </svg>
              <span className="step-num">step 01</span>
              <h3>Paste the source</h3>
              <p>
                A guideline excerpt, a paper&apos;s conclusion section, a draft
                recommendation. The engine extracts the underlying clinical
                claims and tags each with a direction and strength.
              </p>
            </div>
            <div className="flow-step">
              <svg className="glyph" viewBox="0 0 48 48" fill="none">
                <circle cx="14" cy="14" r="3" fill="#f28e2b" />
                <circle cx="14" cy="34" r="3" fill="#0f8d77" />
                <line x1="17" y1="14" x2="38" y2="14" stroke="#f28e2b" strokeWidth="1.4" strokeDasharray="2 2" />
                <line x1="17" y1="34" x2="38" y2="34" stroke="#0f8d77" strokeWidth="1.4" />
                <circle cx="38" cy="14" r="3" fill="#f28e2b" />
                <circle cx="38" cy="34" r="3" fill="#0f8d77" />
              </svg>
              <span className="step-num">step 02</span>
              <h3>Two branches run</h3>
              <p>
                The free branch admits every synthetic re-synthesis. The
                constrained branch only admits text that cites a surviving real
                anchor with intact provenance.
              </p>
            </div>
            <div className="flow-step">
              <svg className="glyph" viewBox="0 0 48 48" fill="none">
                <line x1="8" y1="38" x2="8" y2="14" stroke="#11231e" strokeWidth="1.4" />
                <line x1="20" y1="38" x2="20" y2="22" stroke="#11231e" strokeWidth="1.4" />
                <line x1="32" y1="38" x2="32" y2="10" stroke="#11231e" strokeWidth="1.4" />
                <line x1="6" y1="38" x2="42" y2="38" stroke="#11231e" strokeWidth="1.4" />
                <circle cx="8" cy="14" r="2.5" fill="#0f8d77" />
                <circle cx="20" cy="22" r="2.5" fill="#0f8d77" />
                <circle cx="32" cy="10" r="2.5" fill="#f28e2b" />
              </svg>
              <span className="step-num">step 03</span>
              <h3>Read the divergence</h3>
              <p>
                At year 10, 20, and 30, compare both branches. The honest
                reading is not which branch is &quot;right&quot; — it&apos;s
                whether a gate produced a delta worth auditing.
              </p>
            </div>
          </div>
        </section>

        <section className="section">
          <div className="section-eyebrow">02 · the two branches</div>
          <h2 className="section-h2">
            One future admits everything. The other holds a line.
          </h2>
          <p className="section-lede">
            The whole point of MedEvo is the gap between these two diagrams at
            year 30 — and whether that gap survives an audit.
          </p>

          <div className="branches">
            <article className="branch-card free">
              <span className="b-eyebrow">Free branch · no gate</span>
              <h3>Synthetic carriers admitted freely.</h3>
              <p>
                New AI-generated reviews are accepted at face value. They cite
                each other, harden into apparent consensus, and slowly bury the
                real anchors underneath.
              </p>
              <div className="branch-strata" aria-hidden="true">
                <div className="col-y">
                  <div className="layer real" /><div className="layer real" /><div className="layer real" /><div className="layer real" />
                </div>
                <div className="col-y">
                  <div className="layer real" /><div className="layer real" /><div className="layer real" /><div className="layer synth" style={{ flex: 0.6 }} />
                </div>
                <div className="col-y">
                  <div className="layer real" /><div className="layer real" /><div className="layer synth" style={{ flex: 1.6 }} />
                </div>
                <div className="col-y">
                  <div className="layer real" /><div className="layer synth" style={{ flex: 3.5 }} />
                </div>
              </div>
              <p
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "11px",
                  letterSpacing: "0.12em",
                  color: "var(--muted)",
                  textTransform: "uppercase",
                  marginTop: "4px",
                }}
              >
                y0 → y10 → y20 → y30
              </p>
            </article>

            <article className="branch-card con">
              <span className="b-eyebrow">Constrained branch · provenance gate</span>
              <h3>Synthetic re-syntheses refused at the gate.</h3>
              <p>
                New text must cite at least one surviving real anchor with
                intact provenance. The gate is a mechanism, not a guarantee —
                whether it keeps a claim honest is exactly what each run puts on
                trial.
              </p>
              <div className="branch-strata" aria-hidden="true">
                <div className="col-y">
                  <div className="layer real" /><div className="layer real" /><div className="layer real" /><div className="layer real" />
                </div>
                <div className="col-y">
                  <div className="layer real" /><div className="layer real" /><div className="layer real" /><div className="layer real" />
                </div>
                <div className="col-y">
                  <div className="layer real" /><div className="layer real" /><div className="layer real" /><div className="layer real" />
                </div>
                <div className="col-y">
                  <div className="layer real" /><div className="layer real" /><div className="layer real" /><div className="layer real" />
                </div>
              </div>
              <p
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "11px",
                  letterSpacing: "0.12em",
                  color: "var(--muted)",
                  textTransform: "uppercase",
                  marginTop: "4px",
                }}
              >
                y0 → y10 → y20 → y30
              </p>
            </article>
          </div>
        </section>

        <section className="section" id="values">
          <div className="section-eyebrow">03 · what it brings</div>
          <h2 className="section-h2">
            Three values the instrument refuses to break.
          </h2>

          <div className="values">
            <div className="value">
              <span className="v-num">01</span>
              <h4>Every horizon is one draw, never a prediction.</h4>
              <p>
                Year 10, 20, and 30 panels are samples from a distribution. The
                page never tells you what will happen — only what could.
              </p>
              <span className="v-tag">anti-forecast</span>
            </div>
            <div className="value">
              <span className="v-num">02</span>
              <h4>A run only counts when a real model produced it.</h4>
              <p>
                If the worker falls back to the local stub, the entire page
                reframes as <em>illustrative</em> — visibly, loudly, in the
                chrome.
              </p>
              <span className="v-tag">anti-evidence</span>
            </div>
            <div className="value">
              <span className="v-num">03</span>
              <h4>Nothing here should inform patient care.</h4>
              <p>
                This is research scaffolding for arguing about the future of
                EBM, not decision support. The footer says so every time.
              </p>
              <span className="v-tag">anti-clinical</span>
            </div>
          </div>
        </section>

        <section className="section" id="runs">
          <div className="section-eyebrow">04 · cached runs</div>
          <h2 className="section-h2">Three precomputed runs to start with.</h2>
          <p className="section-lede">
            Synthetic, demo-only clinical text. Click any card to open the
            showcase viewer and watch the agents work.
          </p>

          <div className="runs">
            <Link className="run-card" href="/run">
              <div className="r-header">
                <span className="tag tag-teal">
                  <span className="tag-dot" />branch split
                </span>
                <span className="r-id">run_a7f12c</span>
              </div>
              <h3>Adult sepsis — antibiotics &amp; early vasopressors</h3>
              <p className="r-desc">
                Free branch hardens two underpowered claims into apparent
                consensus; the branches separate by year 30.
              </p>
              <div className="verdict-stripe">
                <span style={{ flex: 5, background: "var(--ember)" }} />
                <span style={{ flex: 1, background: "var(--paper-2)" }} />
                <span style={{ flex: 3, background: "var(--teal)" }} />
              </div>
            </Link>

            <Link className="run-card" href="/run">
              <div className="r-header">
                <span className="tag tag-terra">
                  <span className="tag-dot" />null result
                </span>
                <span className="r-id">run_2c918e</span>
              </div>
              <h3>Bronchiolitis — high-flow nasal cannula</h3>
              <p className="r-desc">
                Live model run completed; branches never separated. Useful
                negative result.
              </p>
              <div className="verdict-stripe">
                <span style={{ flex: 1, background: "var(--ember)" }} />
                <span style={{ flex: 8, background: "var(--paper-2)" }} />
                <span style={{ flex: 1, background: "var(--teal)" }} />
              </div>
            </Link>

            <Link className="run-card" href="/run">
              <div className="r-header">
                <span className="tag tag-ember">
                  <span className="tag-dot" />illustrative
                </span>
                <span className="r-id">run_f4d201</span>
              </div>
              <h3>Antibiotic stewardship — duration of therapy</h3>
              <p className="r-desc">
                Worker degraded to local stub. Mechanism visible; this run
                cannot count as evidence.
              </p>
              <div className="verdict-stripe">
                <span style={{ flex: 4, background: "var(--ember)" }} />
                <span style={{ flex: 2, background: "rgba(176,126,15,0.4)" }} />
                <span style={{ flex: 4, background: "var(--teal)", opacity: 0.4 }} />
              </div>
            </Link>
          </div>
        </section>

        <footer className="home-footer">
          <p>
            MedEvo is a research instrument, not clinical decision support.
            Nothing it outputs should inform the care of an actual patient.
            Showcase runs use synthetic, demo-only clinical text.
          </p>
          <div className="author">
            Tuyen Tran, MD · pediatric surgeon
            <br />
            <span style={{ color: "var(--muted)" }}>
              ORCID 0009-0003-0535-6225
            </span>
          </div>
        </footer>
      </main>
    </div>
  );
}
