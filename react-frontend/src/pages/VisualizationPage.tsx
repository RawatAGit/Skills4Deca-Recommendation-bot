import { useState, useEffect, useMemo } from "react";
import Plot from "react-plotly.js";
import type { Data, Layout } from "plotly.js";
import { Link, useSearchParams } from "react-router-dom";
import "bootstrap-icons/font/bootstrap-icons.css";
import "../App.css";

// Define types for our UMAP-based visualization
interface PlotPoint {
  x: number;
  y: number;
  type: "course_summary" | "course_covered";
  university_short: string;
  university: string;
  course_title: string;
  text: string;
  full_text: string;
}

interface VisualizationResponse {
  plot_data: PlotPoint[];
  model: string;
  total_points: number;
  reduction_method: string;
  umap_params?: {
    n_neighbors: number;
    min_dist: number;
    metric: string;
  };
}

// University color mapping
const UNIVERSITY_COLORS: Record<string, string> = {
  TalTech: "#0072CE", // Blue
  VilniusTech: "#E63946", // Red
  RTU: "#06A77D", // Green
};

const UNIVERSITY_SYMBOLS: Record<string, string> = {
  TalTech: "circle", // Circle
  VilniusTech: "square", // Square
  RTU: "diamond", // Diamond
};

function VisualizationPage() {
  const [searchParams] = useSearchParams();
  const modelName = searchParams.get("model") || "Qwen";

  const [data, setData] = useState<VisualizationResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state for plot points - only universities (no type filter, only showing concepts)
  const [selectedUniversities, setSelectedUniversities] = useState<Set<string>>(
    new Set(["TalTech", "VilniusTech", "RTU"]),
  );

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        setData(null);

        const response = await fetch("/api/visualization_data");
        if (!response.ok) {
          throw new Error(`Failed to fetch visualization data: ${response.status}`);
        }

        const result: VisualizationResponse = await response.json();

        if (!result.plot_data || !Array.isArray(result.plot_data)) {
          throw new Error("Invalid data format received from server");
        }

        setData(result);
      } catch (err) {
        console.error("Error fetching visualization data:", err);
        setError(
          err instanceof Error ? err.message : "Failed to load visualization data",
        );
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [modelName]);

  const filteredPlotData = useMemo(() => {
    if (!data?.plot_data) return [];
    return data.plot_data.filter(
      (point) =>
        point.type === "course_covered" && selectedUniversities.has(point.university_short),
    );
  }, [data, selectedUniversities]);

  const traces = useMemo<Data[]>(() => {
    if (filteredPlotData.length === 0) return [];

    const universities = Array.from(
      new Set(filteredPlotData.map((p) => p.university_short)),
    );

    return universities.reduce<Data[]>((acc, uni) => {
      const points = filteredPlotData.filter((p) => p.university_short === uni);
      if (points.length === 0) {
        return acc;
      }

      const uniColor = UNIVERSITY_COLORS[uni] || "#666666";
      const uniSymbol = UNIVERSITY_SYMBOLS[uni] || "circle";

      acc.push({
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        mode: "markers",
        type: "scatter",
        name: `${uni} Concepts`,
        marker: {
          size: 8,
          color: uniColor,
          symbol: uniSymbol,
          opacity: 0.7,
          line: {
            color: "white",
            width: 1,
          },
        },
        customdata: points.map(
          (p) => [p.course_title, p.text, p.university, p.university_short],
        ) as any,
        hovertemplate:
          "<b>%{customdata[0]}</b><br>%{customdata[1]}<br><i>%{customdata[2]} (%{customdata[3]})</i><extra></extra>",
      });

      return acc;
    }, []);
  }, [filteredPlotData]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      title: {
        text: "",
      },
      autosize: true,
      height: 700,
      margin: { l: 60, r: 40, t: 40, b: 60 },
      xaxis: {
        title: "UMAP Dimension 1",
        zeroline: true,
        showgrid: true,
        gridcolor: "#d6deec",
        tickfont: { color: "#5a6b83" },
        titlefont: { color: "#24324a", size: 14 },
      },
      yaxis: {
        title: "UMAP Dimension 2",
        zeroline: true,
        showgrid: true,
        gridcolor: "#d6deec",
        tickfont: { color: "#5a6b83" },
        titlefont: { color: "#24324a", size: 14 },
      },
      hovermode: "closest",
      showlegend: true,
      legend: {
        x: 0,
        y: -0.12,
        xanchor: "left",
        orientation: "h",
        bgcolor: "rgba(255,255,255,0.92)",
        bordercolor: "rgba(36, 50, 74, 0.12)",
        borderwidth: 1,
        font: { color: "#24324a" },
      },
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
    }),
    [],
  );

  const contributionBreakdown = useMemo(() => {
    if (!data?.plot_data) return {} as Record<string, number>;

    return ["TalTech", "VilniusTech", "RTU"].reduce(
      (acc, uni) => {
        acc[uni] = data.plot_data.filter(
          (p) => p.university_short === uni && p.type === "course_covered",
        ).length;
        return acc;
      },
      {} as Record<string, number>,
    );
  }, [data]);

  const hasData = filteredPlotData.length > 0;

  return (
    <div className="app-shell visualization-page">
      <header className="app-header">
        <div className="app-header__text">
          <h1>Course Concepts Visualization</h1>
          <p>
            Explore the semantic landscape of the Skills4Deca curriculum with an
            interactive UMAP projection of course concepts.
          </p>
          <p className="app-header__meta">
            {data
              ? `${data.model} • ${data.reduction_method || "UMAP"} dimensionality reduction`
              : `Model: ${modelName} • UMAP dimensionality reduction`}
          </p>
        </div>
        <Link to="/" className="header-link">
          Back to recommendations
        </Link>
      </header>

      <div className="app-layout visualization-layout">
        <aside className="visualization-controls">
          <div className="panel-card visualization-panel">
            <h2 className="panel-title">Filter visualization</h2>
            <div className="filter-group">
              <p className="filter-group__label">Universities</p>
              <div className="filter-options">
                {["TalTech", "VilniusTech", "RTU"].map((uni) => (
                  <label className="filter-option" htmlFor={`uni-${uni}`} key={uni}>
                    <input
                      id={`uni-${uni}`}
                      className="filter-checkbox"
                      type="checkbox"
                      checked={selectedUniversities.has(uni)}
                      onChange={(e) => {
                        const newUnis = new Set(selectedUniversities);
                        if (e.target.checked) {
                          newUnis.add(uni);
                        } else {
                          newUnis.delete(uni);
                        }
                        setSelectedUniversities(newUnis);
                      }}
                    />
                    <span
                      className="filter-swatch"
                      style={{ backgroundColor: UNIVERSITY_COLORS[uni] }}
                    ></span>
                    <span className="filter-option__label">{uni}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="legend-grid">
              <div className="legend-item">
                <span className="legend-symbol" style={{ color: UNIVERSITY_COLORS.TalTech }}>
                  ●
                </span>
                <div className="legend-text">
                  <span className="legend-title">TalTech</span>
                  <span className="legend-meta">Circle markers</span>
                </div>
              </div>
              <div className="legend-item">
                <span className="legend-symbol" style={{ color: UNIVERSITY_COLORS.VilniusTech }}>
                  ■
                </span>
                <div className="legend-text">
                  <span className="legend-title">VilniusTech</span>
                  <span className="legend-meta">Square markers</span>
                </div>
              </div>
              <div className="legend-item">
                <span className="legend-symbol" style={{ color: UNIVERSITY_COLORS.RTU }}>
                  ◆
                </span>
                <div className="legend-text">
                  <span className="legend-title">RTU</span>
                  <span className="legend-meta">Diamond markers</span>
                </div>
              </div>
            </div>

            {data?.umap_params && (
              <div className="panel-note">
                <span className="panel-note__label">UMAP parameters</span>
                <ul>
                  <li>n_neighbors: {data.umap_params.n_neighbors}</li>
                  <li>min_dist: {data.umap_params.min_dist}</li>
                  <li>metric: {data.umap_params.metric}</li>
                </ul>
              </div>
            )}
          </div>

          <div className="panel-card panel-card--flat visualization-summary">
            <h3 className="panel-subtitle">Dataset summary</h3>
            <div className="metric-list">
              <div className="metric-item">
                <span className="metric-label">Total points</span>
                <span className="metric-value">{data?.total_points ?? "—"}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Currently displayed</span>
                <span className="metric-value">{filteredPlotData.length}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Active model</span>
                <span className="metric-value">{data?.model ?? modelName}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Reduction method</span>
                <span className="metric-value">{data?.reduction_method || "UMAP"}</span>
              </div>
            </div>
          </div>
        </aside>

        <main className="results-panel visualization-main">
          <div className="results-header">
            <div>
              <h2>Semantic landscape</h2>
              <p className="results-subtitle">
                {hasData
                  ? `Displaying ${filteredPlotData.length} course concepts across three partner universities.`
                  : "Monitor dataset readiness and refine filters to explore course coverage."}
              </p>
            </div>
          </div>

          <div className="visualization-main__content">
            {isLoading && (
              <div className="visualization-state visualization-state--loading">
                <span className="loading-spinner" aria-hidden="true"></span>
                <p>Loading visualization data...</p>
              </div>
            )}

            {!isLoading && error && (
              <div className="visualization-state visualization-state--error" role="alert">
                <i className="bi bi-exclamation-octagon"></i>
                <p>{error}</p>
                <p className="state-meta">Verify that the backend service is running and reachable.</p>
                <Link to="/" className="primary-btn state-action">
                  Back to recommendations
                </Link>
              </div>
            )}

            {!isLoading && !error && !hasData && (
              <div className="visualization-state visualization-state--empty">
                <i className="bi bi-info-circle"></i>
                <p>No data points are available with the current filters.</p>
                <p className="state-meta">Adjust university filters or refresh the dataset to continue.</p>
                <button
                  type="button"
                  className="primary-btn state-action"
                  onClick={() =>
                    setSelectedUniversities(new Set(["TalTech", "VilniusTech", "RTU"]))
                  }
                >
                  Reset filters
                </button>
              </div>
            )}

            {hasData && (
              <>
                <section className="visualization-insights">
                  <article className="insight-card">
                    <h3>Understanding the visualization</h3>
                    <p>
                      Think of the scatter plot as a map of courses: each dot represents one course and nearby dots cover
                      similar skills. We create this map by turning the course descriptions into long numerical
                      fingerprints ("embeddings") and then flattening them into two dimensions so they are easy to view.
                    </p>
                    <ul>
                      <li>
                        <strong>Proximity</strong> means the courses talk about related ideas—for example, energy efficiency
                        modules sit next to sustainable construction courses.
                      </li>
                      <li>
                        <strong>Color & shape</strong> show which university provides the course.
                      </li>
                      <li>
                        <strong>Clusters</strong> highlight topic neighborhoods such as BIM design, green building, or project
                        management.
                      </li>
                    </ul>
                    <p className="insight-example">
                      Example: "RTU — Sustainable Construction Technologies" appears close to TalTech and VilniusTech
                      courses on energy-efficient materials, signalling that learners will encounter overlapping skills
                      across universities.
                    </p>
                  </article>

                  <article className="insight-card">
                    <h3>University contributions</h3>
                    <div className="contribution-grid">
                      {["TalTech", "VilniusTech", "RTU"].map((uni) => (
                        <div
                          key={uni}
                          className="contribution-tile"
                          style={{ borderColor: UNIVERSITY_COLORS[uni], backgroundColor: `${UNIVERSITY_COLORS[uni]}12` }}
                        >
                          <span className="contribution-title" style={{ color: UNIVERSITY_COLORS[uni] }}>
                            {uni}
                          </span>
                          <span className="contribution-count">
                            {contributionBreakdown[uni] ?? 0} concepts
                          </span>
                          <span className="contribution-symbol">
                            {UNIVERSITY_SYMBOLS[uni] === "circle"
                              ? "●"
                              : UNIVERSITY_SYMBOLS[uni] === "square"
                                ? "■"
                                : "◆"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </article>
                </section>

                <section className="visualization-plot">
                  <div className="visualization-plot__header">
                    <h3>
                      <i className="bi bi-scatter-chart"></i> UMAP 2D projection
                    </h3>
                    <span className="plot-subtitle">
                      Hover to inspect concepts • toggle universities in the legend • use zoom tools to explore clusters
                    </span>
                  </div>
                  <div className="visualization-plot__canvas">
                    <Plot
                      data={traces}
                      layout={layout}
                      config={{
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ["lasso2d", "select2d"],
                        displaylogo: false,
                      }}
                      style={{ width: "100%", height: "700px" }}
                    />
                  </div>
                </section>

                <footer className="visualization-footer">
                  <small>
                    Powered by Qwen · UMAP dimensionality reduction · GLM 4.6-powered course discovery
                  </small>
                </footer>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default VisualizationPage;
