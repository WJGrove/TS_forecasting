# PRD Addendum: Evaluation App Visualizations (Dash on Databricks)

**Purpose:** This document is a paste-ready addendum for the PRD “Visualizations” section for an analyst-first evaluation app built with **Dash** and deployed as a **Databricks App**.

**Primary audience:** Analysts / DS (interactive exploration, slice-and-dice evaluation)  
**Secondary audience:** Managers (same views, but with curated defaults and fewer exposed knobs)

---

## Product goals

1. **Fast exploration** of forecast accuracy across multiple business dimensions (Market, Category, Size, etc.).
2. **Standardized evaluation story** that is easy to defend and repeat across projects.
3. **Portable architecture**: UI reads precomputed artifacts so the same app design works across Databricks or other cloud platforms later.

---

## Scope (v0)

### In scope
- Dash app with multi-select dimension filters and date range controls.
- Forecast vs actual visualization, baseline comparison, and standard error metrics.
- Error mix view (contribution to absolute error) across time.
- Volume vs error scatter (prioritization view).
- Error distribution diagnostics (box/hist) + “worst offenders” table.

### Out of scope (for v0)
- User authentication/roles beyond what Databricks Apps provides.
- Full “run the pipeline” UI (can be a separate app later).
- Advanced model explainability and feature-level attribution.

---

## Global definitions

### Volume
- **Volume = actual target `y_actual` aggregated over the selected window** (and time grain), for the selected slice.
- Use volume for:
  - weighting/prioritization (scatter)
  - optional “top-N” cutoffs when too many dimension values are selected

### Primary error metrics
- **WAPE** (preferred default for business discussions):
  - `WAPE = sum(|y - yhat|) / sum(|y|)`
  - Present as a percent. (Multiply by 100 in UI.)
- **MAPE** (secondary, optional default):
  - `MAPE = mean(|(y - yhat) / y|)`
  - **Zero-handling requirement:** define a consistent policy:
    - Option A: exclude points where `y == 0`
    - Option B: add small epsilon in denominator
  - For v0, default to **exclude y == 0** and display `n_excluded` in tooltips/metadata.

### Baseline forecast
- **Last year same period baseline**:
  - Weekly projects: `baseline(t) = y_actual(t - 52 weeks)` (or 53 as calendar requires)
  - Monthly projects: `baseline(t) = y_actual(t - 12 months)`
- Baseline is shown as a separate line in the Forecast Explorer, and used for delta metrics when desired.

### Confidence intervals
- Not guaranteed for every model.
- **UI rule:** only render confidence bands if `y_lower` and `y_upper` are present; otherwise hide the band and do not show “CI” legend entries.

---

## Global controls (persistent across views)

1. **Run selector**: `run_id` (single-select)
   - Optional later: “Compare two runs” toggle (overlay + delta metrics)
2. **Date range**: start/end (applies to all views)
3. **Time grain toggle**: {Weekly, Monthly}
   - Default = Weekly
   - Monthly is supported for projects where artifacts exist at monthly grain (or can be derived without heavy recompute)
4. **Dimension filters** (multi-select):
   - Market
   - Product Category
   - Pack Size (or other)
   - Additional dims as configured
5. **Dimension focus (for grouped visuals)**:
   - “Group by” dropdown for views that require picking a single dimension (Error Mix, Distribution)

**Multi-select behavior expectation**
- If user selects multiple values in one dimension, the app should show multiple lines/points where reasonable.
- If the selection produces too many traces (e.g., > 20), the app should:
  - default to top-N by volume
  - show a message indicating that results are limited for readability/performance

---

## Views / pages

### View 1 — Forecast Explorer
**Goal:** Validate forecast behavior for a slice/selection and quantify accuracy.

**Inputs**
- Global filters + optional “series selector” (search/select a single series id for deep dive)

**Outputs**
- Line chart:
  - Actuals vs Forecast
  - Baseline (last year same period)
  - Optional confidence band if available
- KPI cards:
  - WAPE
  - MAPE
  - Volume (sum of y_actual in window)
  - Observation count (`n_obs`)
- Optional table:
  - Metrics by sub-dimension (e.g., by Market within selected Category)

**Interactions**
- Hover tooltips show: y_actual, y_pred, baseline, error, timestamp
- Legend toggles traces
- Optional drill: click a trace/point → pins that slice and syncs other views

---

### View 2 — Error Mix (Contribution to Absolute Error)
**Goal:** Explain “where the error came from” and how the mix changes between windows.

**Controls (in addition to global)**
- “Group by dimension” dropdown (Market / Category / Size / …)
- Optional “Top-K categories” control (default 8–12)

**Visualization**
- 100% stacked bar over time (or stacked area) showing each group’s share of total absolute error.

**Standard definition**
- Absolute error per row: `abs_error = |y - yhat|`
- For each period `p` and group value `g`:
  - `abs_error_g,p = sum(abs_error for rows in group g during period p)`
  - `share_g,p = abs_error_g,p / sum(abs_error_all_groups,p)`

**Accompanying table**
- Ranking of groups by:
  - total abs_error (window)
  - WAPE for that group
  - volume

---

### View 3 — Volume vs Error Scatter (Prioritization)
**Goal:** Identify “high-volume, high-error” slices worth attention.

**Point definition**
- Each point = a slice (dimension value combination) summarized over the selected window.

**Axes**
- X = Volume (sum(y_actual))
- Y = Error metric (default WAPE; allow dropdown to switch to MAPE)

**Optional encodings**
- Point size = volume (or constant if size causes clutter)
- Color = chosen grouping dimension (Market/Category/Size)

**Interaction**
- Click a point → sets filters to that slice and navigates (or syncs) to Forecast Explorer.

**Optional guide overlays**
- Median volume vertical line and median error horizontal line to create 4 quadrants:
  - High volume / High error (priority)
  - High volume / Low error (healthy)
  - Low volume / High error (likely noisy)
  - Low volume / Low error (deprioritize)

---

### View 4 — Error Distribution (Diagnostics)
**Goal:** Understand error spread and outliers by a chosen dimension.

**Controls**
- Group by dimension dropdown
- Error metric selector (WAPE/MAPE)
- Optional: show only top-N groups by volume

**Visualizations**
- Box plot of the selected error metric by group
- Optional histogram for the overall distribution

**Table**
- “Worst offenders” table:
  - Top N slices by error metric
  - Include: volume, n_obs, baseline delta (optional)

---

## Data contract (artifacts expected from pipeline)

### `forecast_results`
- Keys: `run_id`, `ds` (timestamp), dims…
- Values:
  - `y_actual`
  - `y_pred`
  - optional: `y_lower`, `y_upper`
  - optional: `y_baseline` (last year same period)
- Notes:
  - Store at the evaluator’s supported grains (weekly and/or monthly) to keep the UI fast.

### `metrics_by_slice`
- Keys: `run_id`, `window_start`, `window_end`, dims…
- Values:
  - `wape`, `mape`
  - `volume` (sum(y_actual))
  - `n_obs`
  - optional: bias metrics, baseline deltas

### `error_contrib`
- Keys: `run_id`, `period`, `dimension_name`, `dimension_value`
- Values:
  - `abs_error`
  - `abs_error_share`

### `scatter_points`
- Keys: `run_id`, `window_start`, `window_end`, dims… (or a designated “slice_id”)
- Values:
  - `volume`
  - `wape`, `mape`
  - `n_obs`

---

## Performance and UX requirements

1. **No heavy recomputation in UI callbacks**
   - Prefer reading aggregated/summary tables (metrics_by_slice, scatter_points, error_contrib)
2. **Responsiveness targets**
   - Initial app load: ~3 seconds typical
   - Filter updates: ~1 second typical (depending on table sizes)
3. **Graceful degradation**
   - If a view would render too many traces/points, apply top-N by volume with a visible message.

---

## Dash implementation notes (callbacks, background work, caching)

Dash apps are driven by callbacks: changes to filters trigger recomputation of figures/tables.

### Callback patterns
- Use a **single source of truth** for filters (e.g., a `dcc.Store` holding current selections).
- Keep callbacks small and focused:
  - one callback prepares filtered datasets (or query params)
  - separate callbacks render charts/tables from prepared data

### Background callbacks (for slow operations)
If any callback may exceed a comfortable UI latency (e.g., larger queries, multi-run comparisons):
- Use **background callbacks** so the UI does not freeze during compute.
- Show a “running/loading” state and update outputs when complete.

### Caching patterns
To keep filter updates fast and reduce repeated reads:
- Cache query results keyed by:
  - `(run_id, date_range, grain, selected_dims...)`
- Practical caching approaches:
  - in-memory caching for small/medium aggregated tables
  - filesystem/disk caching for larger responses
- Cache invalidation strategy:
  - include `run_id` and artifact version in cache keys to avoid stale results

### Databricks-specific note
Prefer reading from **precomputed Delta tables** (or view materializations) rather than recomputing metrics inside the app. This keeps the app responsive and makes behavior consistent across environments.

---

## Default choices (v0)

- Default time grain: **Weekly**
- Supported grains: **Weekly and Monthly**
- Default primary metric: **WAPE**
- Secondary metric: **MAPE** (exclude y == 0 points by default; report exclusions)
- Default baseline: **Last year same period**
- Confidence intervals: **hide unless columns are present**
- Dimension filters: **multi-select enabled**

---

# UI / Style / Interaction Addendum (Enterprise Dashboard)

This section captures **look-and-feel** decisions and open questions for a Dash-based evaluator dashboard. It is intentionally explicit so it is easy to implement with agents and easy for you to review.

## High-level UI direction

### Visual style
- Target style: **clean enterprise dashboard**
- Development-first: expose “bells and whistles” (advanced controls, debug panels, extra overlays), with the expectation that these can be hidden later for manager-facing versions.

### Theme and colors
- Support **Light/Dark toggle**.
- Support a small set of configurable color codes (HEX) for key roles, ideally defined in one place:
  - Primary accent
  - Secondary accent
  - Success / Warning / Error
  - Neutral text and background
- Notes / open decisions:
  - Decide whether colors are set via:
    - a theme system in the chosen component library, or
    - CSS variables (recommended for portability and ease of editing).

---

## Layout and navigation

### Dashboard layout (hybrid filters + sticky behavior)
- Use a **hybrid filter layout**:
  - **Always-visible sticky controls** for the most frequently used filters:
    - Run selector (`run_id`)
    - Date range
    - Time grain (Weekly/Monthly)
    - Primary dimension multi-selects (core business slicing)
  - **Collapsible “Advanced Filters” drawer** for less common or high-cardinality controls:
    - Series selector/search
    - “Top-N by volume” controls
    - Group-by dimension selection (for Error Mix / Distributions)
    - Optional debug toggles (show query timings, row counts, cache hits)
- Sticky filters requirement:
  - Filters remain visible while scrolling the visualizations.

### Single long scroll vs tabs/pages
- Current preference: **single-page continuous scroll** (one screen that you scroll).
- Open decision to revisit (recommended checkpoint after v0 working):
  - **Single page scroll** works well for “narrative dashboard” use and quick scanning.
  - **Tabs/pages** can reduce cognitive load and make cross-filtering rules simpler.
- Action note: implement v0 as single scroll, but keep the code structured so views can be split into tabs/pages later with minimal refactor.

---

## Chart composition and formatting standards

### “Primary” timeseries chart composition
- A **large line chart** is the hero visual:
  - Plot **Actual**, **Forecast**, **Baseline** simultaneously.
  - If confidence intervals exist, show as a **shaded band** (lower/upper).
- Show a small KPI table/row adjacent to the chart (same viewport):
  - WAPE, MAPE, Volume, n_obs (and optionally Bias / Baseline delta later).
- Axis standards:
  - Always show **units label** on axes.
  - Consistent date formatting for weekly vs monthly.
  - Tooltips must show: timestamp, actual, forecast, baseline, error, and (if present) bounds.

### Global “always-on” aggregate chart (minimally filtered)
- Requirement: show an **aggregate Actual/Forecast/Baseline** line chart that is **not affected by certain filters**.
- Purpose: maintain a stable “macro” reference while exploring slices.
- Open decision:
  - Define which filters DO apply vs DO NOT apply to the global aggregate.
  - Suggested default:
    - Applies: run_id, date range, grain
    - Does NOT apply: dimension multi-selects (Market/Category/etc.), series selector
- Implementation note:
  - Treat this as a separate “filter scope” in the PRD (see Cross-filtering section).

### Overplotting limits and readability
- If a selection would generate too many traces/points:
  - Auto-apply **Top-N by volume** with a visible warning banner.
  - Include the selected N value and the ranking dimension used (volume).
- Open decision:
  - Default N (suggest 10–20 for lines; 200–1,000 for scatter depending on performance).

---

## Cross-filtering and “filter scopes”

You want cross-filtering, but not “everything affects everything.” The clean way to specify this is by defining **filter scopes**.

### Proposed filter scopes (v0)
1. **Global scope (applies everywhere)**
   - run_id
   - date range
   - time grain
2. **Slice scope (applies to slice-based visuals and tables)**
   - dimension multi-selects (Market/Category/Size/etc.)
   - series selector (optional)
3. **View-local controls**
   - group-by dimension (Error Mix / Distribution)
   - metric selector (WAPE vs MAPE)
   - top-N control

### Key open decision (must be explicitly documented)
- Exactly which visuals use Global only vs Global+Slice.
- Current intent:
  - The **macro aggregate chart** uses Global only.
  - The **Forecast Explorer** uses Global+Slice (and optionally series selector).
  - Error Mix / Scatter / Distribution typically use Global+Slice, with their own view-local controls.

---

## Additional advanced visualization request: 3D volume surface / bar

### Goal
A 3D visual to view:
- Axis 1: Products (or Product Category/SKU)
- Axis 2: Customers or Markets
- Axis 3: Volume (aggregated y_actual)

### Notes / open decisions (important)
- 3D visuals can be powerful but also harder to read and sometimes slower.
- Decision point:
  - Should this be **true 3D** (surface/3D bars) or a **standard heatmap** (often more readable and “enterprise standard”)?
- Suggested “standards-first” approach:
  1. Implement **heatmap** first (Product x Market, color = volume).
  2. Optionally add a **3D surface** toggle later if it remains valuable.

### Data and usability considerations
- High cardinality warning:
  - If Product or Customer has many values, require top-N selection or aggregation (e.g., Product Category instead of SKU).
- Default:
  - Top-N by volume for both axes, with “Other” bucket optional.

---

## Bookmarkable / shareable state

### Requirement
- App supports a **bookmarkable state** so a URL can represent:
  - run_id
  - date range
  - grain
  - selected dimensions
  - selected view position (optional)
- Open decisions:
  - Determine which parts of state should be encoded in the URL vs stored in-session.
  - Decide how to handle very large multi-selects (URL length concerns).

---

## Tables vs visuals

### Guidance
- Dashboard is **visual-first**, with a few small, purposeful tables:
  - “Top contributors” table under Error Mix
  - “Worst offenders” table under Distribution
  - Optional metrics-by-subdimension table near Forecast Explorer

### Table formatting standards
- Keep tables compact:
  - show key columns only
  - sortable
  - support copy/export later (optional)

---

## Definitions & equations panel (manager-friendly)

### Requirement
- Include an expandable “Definitions & Equations” panel that:
  - Defines WAPE and MAPE (including MAPE zero-handling policy)
  - Defines baseline (last year same period, with weekly/monthly offsets)
  - Defines contribution-to-error and volume
- UX note:
  - This panel should be discoverable but not distracting (collapsed by default).

---

## Component library decision and implications (for agent implementation)

### Requirement
- Use a UI component library (preferred for speed and consistent enterprise styling).

### Open decision to finalize
Pick one primary UI kit for v0:
- **Bootstrap-based** (common “enterprise dashboard” feel; predictable layout grid)
- **Mantine-based** (more modern components; still can be enterprise-clean)
- Custom CSS only (not recommended for v0)

### Recommendation (aligned to your preference)
- Start with a Bootstrap-style system for v0, then consider Mantine if you later want a more modern product-like feel.

---

## Implementation notes (non-functional, but affects UX)

### Loading states and long operations
- Any callback that could be slow should:
  - use background execution patterns (so the UI does not freeze)
  - show a loading state (spinner/skeleton) and a “last updated” timestamp

### Caching and performance visibility
- Add a small “Diagnostics” toggle (dev-only) to display:
  - query duration
  - rows returned
  - cache hit/miss

---

## Checklist (did we capture everything requested?)

- [x] Enterprise dashboard style, dev-friendly controls
- [x] Light/Dark toggle
- [x] Optional color-code inputs for main colors
- [x] Hybrid filters: some always visible, others in collapsible drawer
- [x] Sticky filters for scrolling dashboard
- [x] Large line chart + small KPI table visible at the same time
- [x] Shaded uncertainty band when intervals present; hidden otherwise
- [x] Aggregate Actual/Forecast/Baseline chart that is not affected by certain filters
- [x] Units labels on axes
- [x] 3D product x market/customer x volume visualization request, with standards-first note (heatmap first)
- [x] Cross-filtering “to a degree” via defined filter scopes
- [x] Bookmarkable state
- [x] Auto top-N by volume + warning when charts become busy
- [x] Visual-first with small tables as needed
- [x] Definitions/equations expandable panel
- [x] Use a UI component library to improve reviewability/consistency

