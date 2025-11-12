import { useState, useEffect, FormEvent } from "react";
import { Link } from "react-router-dom";
import "bootstrap-icons/font/bootstrap-icons.css";
import "./App.css";
import SearchProgress from "./components/SearchProgress";

// Define types for API responses
interface Recommendation {
  course_title: string;
  course_summary?: string;
  description?: string;
  explanation?: string;
  university?: string;
  enhanced_reranker_score?: number;
  llm_relevance_score?: number;
  llm_selection_reason?: string;
  reranker_score?: number;
  // Legacy fields for backwards compatibility
  theme_match?: number;
  subtopic_coverage?: number;
  matched_concepts?: string[];
}


function App() {
  const [prompt, setPrompt] = useState<string>("");
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [searchMethod, setSearchMethod] = useState<string>("");
  const [searchCompleted, setSearchCompleted] = useState<boolean>(false);
  const [hasSearched, setHasSearched] = useState<boolean>(false);
  const [isGuidanceOpen, setIsGuidanceOpen] = useState<boolean>(false);
  const [isProgressExpanded, setIsProgressExpanded] = useState<boolean>(true);

  // Helper function to determine badge color based on score
  const getBadgeColor = (score: number): string => {
    if (score >= 80) return "success";
    if (score >= 60) return "primary";
    if (score >= 40) return "info";
    if (score >= 20) return "warning";
    return "secondary";
  };

  // Handle real-time progress updates
  const handleProgressUpdate = (update: any) => {
    console.log('Progress update:', update);
  };

  // Handle search completion from streaming
  const handleSearchComplete = (results: any) => {
    console.log('Search complete:', results);
    setIsLoading(false);
    setSearchCompleted(true);
    if (!results.recommendations || results.recommendations.length === 0) {
      const errorMessage =
        results.message ||
        "No courses found matching your query. Please try a different topic.";
      setError(errorMessage);
    } else {
      setRecommendations(results.recommendations);
      setSearchMethod(
        results.search_method || "Multi-Query Analysis with LLM Validation",
      );
    }
  };

  // Handle search errors from streaming
  const handleSearchError = (error: string) => {
    console.error('Search error:', error);
    setError(error || "Search failed. Please try again.");
    setIsLoading(false);
    setSearchCompleted(true);
  };

  // Handle form submission (now starts streaming search)
  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault(); // Prevent default form submission
    if (!prompt) {
      setError("Please enter your learning interests.");
      return;
    }

    setIsLoading(true);
    setSearchCompleted(false);
    setError(null);
    setRecommendations([]); // Clear previous recommendations
    setSearchMethod("");
    setHasSearched(true);

    // The actual search will be triggered by SearchProgress component using SSE
    // No need to make the API call here anymore
  };

  useEffect(() => {
    if (isLoading) {
      setIsProgressExpanded(true);
      setHasSearched(true);
      return;
    }

    if (searchCompleted && recommendations.length > 0) {
      setIsProgressExpanded(false);

      const hideTimer = window.setTimeout(() => {
        setHasSearched(false);
      }, 700);

      return () => window.clearTimeout(hideTimer);
    }
  }, [isLoading, searchCompleted, recommendations.length]);

  // --- JSX Structure ---
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__text">
          <h1>Skills4Deca Course Recommendations</h1>
          <p>
            GLM 4.6-powered course discovery for construction professionals and students.
          </p>
          <p className="app-header__meta">
            Powered by GLM 4.6 · Enhanced with intelligent reranking
            and validation
          </p>
          <div className="app-header__notice">
            <i className="bi bi-info-circle"></i>
            <span>
              Due to external LLM providers, the chatbot may experience short-term downtime. 
              If a search takes more than 40 seconds, please try again shortly.
            </span>
          </div>
        </div>
        <Link to="/visualization" className="header-link">
          View course visualizations
        </Link>
      </header>

      <div className="app-layout">
        <aside className="search-panel">
          <div className="panel-card">
            <h2 className="panel-title">Search the catalog</h2>
            <form onSubmit={handleSubmit}>
              <label htmlFor="interestPrompt" className="panel-label">
                Describe what you want to learn
              </label>
              <textarea
                id="interestPrompt"
                className="panel-textarea"
                rows={5}
                placeholder="Example: I want to learn about BIM for construction projects, energy efficiency in buildings, or sustainable materials..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={isLoading}
              />

              <div className="panel-actions">
                <button
                  type="submit"
                  id="submitBtn"
                  className="primary-btn"
                  disabled={isLoading}
                >
                  {isLoading ? "Searching..." : "Get recommendations"}
                </button>
              </div>

              <div className="quick-queries">
                <p className="quick-queries__label">Quick guidance</p>
                <div className="quick-queries__chips">
                  <span className="query-chip">
                    BIM for construction projects
                  </span>
                  <span className="query-chip">
                    Sustainable materials and energy efficiency
                  </span>
                  <span className="query-chip">IoT and smart buildings</span>
                  <span className="query-chip">
                    Construction project management
                  </span>
                </div>
                <button
                  type="button"
                  className="secondary-link"
                  onClick={() => setIsGuidanceOpen(true)}
                >
                  Full guidance panel
                </button>
              </div>
            </form>
          </div>
        </aside>

        <main className="results-panel">
          <div className="results-header">
            <div>
              <h2>Recommendations</h2>
              <p className="results-subtitle">
                {searchCompleted || isLoading
                  ? "Pipeline updates appear as your search runs."
                  : "Run a search to view recommended courses."}
              </p>
            </div>
          </div>

          {hasSearched && (
            <div className="progress-dropdown">
              <button
                type="button"
                className="progress-toggle"
                onClick={() => setIsProgressExpanded((prev) => !prev)}
                aria-expanded={isProgressExpanded}
                disabled={isLoading}
              >
                <span>Pipeline progress</span>
                <span className="progress-toggle__state">
                  {isProgressExpanded ? "Hide details" : "Show details"}
                </span>
              </button>

              {isProgressExpanded && (
                <div className="results-progress">
                  <SearchProgress
                    isActive={isLoading}
                    isCompleted={searchCompleted && recommendations.length > 0}
                    query={prompt}
                    onProgressUpdate={handleProgressUpdate}
                    onComplete={handleSearchComplete}
                    onError={handleSearchError}
                  />
                </div>
              )}
            </div>
          )}

          {error && (
            <div id="errorMessage" className="alert alert-danger" role="alert">
              {error}
            </div>
          )}

          {!isLoading && recommendations.length > 0 && (
            <div id="resultsContainer" className="results-container">
              <div id="recommendationsList">
                {recommendations.map((rec, index) => (
                  <div key={index} className="card recommendation-card">
                    <div className="card-header">
                      <h5 className="course-title mb-0">{rec.course_title}</h5>
                      {rec.university && (
                        <p className="text-muted mb-2">
                          <i className="bi bi-building"></i> {rec.university}
                        </p>
                      )}
                      <div className="course-metrics">
                        {rec.llm_relevance_score !== undefined ? (
                          <span
                            className={`badge bg-${getBadgeColor(rec.llm_relevance_score * 100)}`}
                            title="GLM 4.6-validated relevance to your query"
                          >
                            {Math.round(rec.llm_relevance_score * 100)}%
                            Relevance
                          </span>
                        ) : rec.theme_match !== undefined ? (
                          <span
                            className={`badge bg-${getBadgeColor(rec.theme_match)}`}
                            title="Theme Match Score"
                          >
                            {Math.round(rec.theme_match)}% Match
                          </span>
                        ) : null}
                      </div>
                    </div>
                    <div className="card-body">
                      {rec.course_summary && (
                        <>
                          <h6 className="card-subtitle mb-2 text-muted">
                            Course Summary
                          </h6>
                          <p className="course-summary">{rec.course_summary}</p>
                        </>
                      )}
                      <h6 className="card-subtitle mb-2 text-muted">
                        Why this course is relevant
                      </h6>
                      <p className="course-explanation">
                        {rec.llm_selection_reason ||
                          rec.explanation ||
                          "This course matches your learning interests."}
                      </p>
                      {rec.matched_concepts &&
                        rec.matched_concepts.length > 0 && (
                          <>
                            <h6 className="card-subtitle mb-2 text-muted">
                              Matched Topics
                            </h6>
                            <div className="d-flex flex-wrap gap-1">
                              {rec.matched_concepts.map((concept, idx) => (
                                <span
                                  key={idx}
                                  className="badge bg-light text-dark border"
                                >
                                  {concept}
                                </span>
                              ))}
                            </div>
                          </>
                        )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {(searchMethod || recommendations.length > 0) && (
            <div className="info-drawer">
              {searchMethod && (
                <div className="info-drawer__item">
                  <span className="info-drawer__label">Search method</span>
                  <span className="info-drawer__value">{searchMethod}</span>
                </div>
              )}
              {recommendations.length > 0 && (
                <div className="info-drawer__item">
                  <span className="info-drawer__label">Courses returned</span>
                  <span className="info-drawer__value">
                    {recommendations.length}
                  </span>
                </div>
              )}
            </div>
          )}
        </main>

        <aside className={`guidance-panel ${isGuidanceOpen ? "open" : ""}`}>
          <div className="guidance-header">
            <h3>Query Examples & Tips</h3>
            <button
              type="button"
              className="guidance-close"
              onClick={() => setIsGuidanceOpen(false)}
            >
              Close
            </button>
          </div>
          <div className="guidance-body">
            <section>
              <h4>Multi-concept queries</h4>
              <p className="section-intro">
                Combine multiple technologies for comprehensive results.
              </p>
              <div className="guidance-actions">
                <button
                  type="button"
                  className="guidance-btn"
                  onClick={() => {
                    setPrompt(
                      "I want to learn about BIM implementation combined with IoT sensors for smart construction monitoring and facility management",
                    );
                    setIsGuidanceOpen(false);
                  }}
                >
                  BIM + IoT + Smart Monitoring
                  <span className="guidance-btn__meta">
                    Digital twins, real-time monitoring, facility management
                  </span>
                </button>
                <button
                  type="button"
                  className="guidance-btn"
                  onClick={() => {
                    setPrompt(
                      "I need courses covering sustainable materials, renewable energy systems, and green building certification processes like LEED or BREEAM",
                    );
                    setIsGuidanceOpen(false);
                  }}
                >
                  Sustainability + Energy + Certification
                  <span className="guidance-btn__meta">
                    Green materials, renewable energy, building certification
                  </span>
                </button>
              </div>
            </section>

            <section>
              <h4>Technology-specific queries</h4>
              <p className="section-intro">
                Focus on specific technologies and tools.
              </p>
              <div className="guidance-actions">
                <button
                  type="button"
                  className="guidance-btn"
                  onClick={() => {
                    setPrompt(
                      "I'm looking for courses specifically about 3D laser scanning, photogrammetry, and point cloud processing for construction surveying",
                    );
                    setIsGuidanceOpen(false);
                  }}
                >
                  3D Scanning & Surveying
                  <span className="guidance-btn__meta">
                    Laser scanning, photogrammetry, point cloud processing
                  </span>
                </button>
                <button
                  type="button"
                  className="guidance-btn"
                  onClick={() => {
                    setPrompt(
                      "I want to learn about computational design, parametric modeling, and digital fabrication in construction",
                    );
                    setIsGuidanceOpen(false);
                  }}
                >
                  Computational Design
                  <span className="guidance-btn__meta">
                    Parametric modeling, digital fabrication, algorithmic design
                  </span>
                </button>
                <button
                  type="button"
                  className="guidance-btn"
                  onClick={() => {
                    setPrompt(
                      "I need courses on artificial intelligence and machine learning applications in construction management and predictive maintenance",
                    );
                    setIsGuidanceOpen(false);
                  }}
                >
                  AI & Machine Learning
                  <span className="guidance-btn__meta">
                    AI applications, predictive maintenance, construction
                    analytics
                  </span>
                </button>
              </div>
            </section>

            <section>
              <h4>Application-focused queries</h4>
              <p className="section-intro">
                Describe specific use cases and applications.
              </p>
              <div className="guidance-actions">
                <button
                  type="button"
                  className="guidance-btn"
                  onClick={() => {
                    setPrompt(
                      "I'm a project manager who needs to learn about construction scheduling, delay analysis, and risk management techniques for large infrastructure projects",
                    );
                    setIsGuidanceOpen(false);
                  }}
                >
                  Project Management Applications
                  <span className="guidance-btn__meta">
                    Scheduling, delay analysis, risk management for
                    infrastructure
                  </span>
                </button>
                <button
                  type="button"
                  className="guidance-btn"
                  onClick={() => {
                    setPrompt(
                      "I want to learn about retrofitting existing buildings for energy efficiency, HVAC system optimization, and building performance simulation",
                    );
                    setIsGuidanceOpen(false);
                  }}
                >
                  Building Retrofit & Performance
                  <span className="guidance-btn__meta">
                    Energy retrofitting, HVAC optimization, performance
                    simulation
                  </span>
                </button>
              </div>
            </section>

            <section>
              <h4>Tips for better results</h4>
              <ul className="tips-list">
                <li>
                  <strong>Be specific:</strong> Use exact technologies ("BIM",
                  "IoT sensors", "energy modeling").
                </li>
                <li>
                  <strong>Use industry terms:</strong> Professional terminology
                  improves semantic matches.
                </li>
                <li>
                  <strong>Combine concepts:</strong> Mix related topics for more
                  comprehensive results.
                </li>
                <li>
                  <strong>Describe the context:</strong> Mention your role,
                  project type, or desired outcome.
                </li>
              </ul>
            </section>
          </div>
        </aside>
      </div>

      <footer className="app-footer">
        <p>Skills4Deca Project</p>
      </footer>
    </div>
  );
}

export default App;
