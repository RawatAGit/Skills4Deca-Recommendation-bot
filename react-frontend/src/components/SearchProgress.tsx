import { useState, useEffect, useRef } from "react";

interface ProgressStep {
  id: number;
  icon: string;
  title: string;
  description: string;
  estimatedTime: string;
  actualTime: string;
}

interface ProgressUpdate {
  type: "progress" | "complete" | "error";
  step_id?: number;
  step_name?: string;
  status?: "starting" | "completed" | "error";
  time_taken?: number;
  total_time?: number;
  timestamp?: number;
  error?: string;
  recommendations?: any[];
  search_method?: string;
  message?: string;
  processing_time?: any;
}

interface SearchProgressProps {
  isActive: boolean;
  isCompleted?: boolean;
  query?: string;
  onProgressUpdate?: (update: ProgressUpdate) => void;
  onComplete?: (results: any) => void;
  onError?: (error: string) => void;
}

const SearchProgress: React.FC<SearchProgressProps> = ({
  isActive,
  isCompleted = false,
  query,
  onProgressUpdate,
  onComplete,
  onError,
}) => {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [steps, setSteps] = useState<ProgressStep[]>([
    {
      id: 0,
      icon: "🧠",
      title: "Analyzing Your Interests",
description: "AI expanding your query into construction-focused searches",
      estimatedTime: "Starting...",
      actualTime: "",
    },
    {
      id: 1,
      icon: "🔍",
      title: "Smart Course Discovery",
      description: "Searching 49 courses from multiple angles",
      estimatedTime: "Waiting...",
      actualTime: "",
    },
    {
      id: 2,
      icon: "✨",
title: "AI Quality Validation",
      description: "Expert AI validating each course's relevance to your goals",
      estimatedTime: "Waiting...",
      actualTime: "",
    },
    {
      id: 3,
      icon: "🎯",
      title: "Ready!",
      description: "Your personalized course recommendations",
      estimatedTime: "Complete",
      actualTime: "",
    },
  ]);

  const eventSourceRef = useRef<EventSource | null>(null);
  const progressHistoryRef = useRef<ProgressUpdate[]>([]);

  // Reset state when starting new search
  useEffect(() => {
    if (!isActive && !isCompleted) {
      setCurrentStep(0);
      progressHistoryRef.current = [];

      // Reset steps
      setSteps([
        {
          id: 0,
          icon: "🧠",
          title: "Analyzing Your Interests",
description: "GLM 4.6 expanding your query into construction-focused searches",
          estimatedTime: "Starting...",
          actualTime: "",
        },
        {
          id: 1,
          icon: "🔍",
          title: "Smart Course Discovery",
          description: "Searching 49 courses from multiple angles",
          estimatedTime: "Waiting...",
          actualTime: "",
        },
        {
          id: 2,
          icon: "✨",
title: "GLM 4.6 Quality Validation",
      description: "GLM 4.6 validating each course's relevance to your goals",
          estimatedTime: "Waiting...",
          actualTime: "",
        },
        {
          id: 3,
          icon: "🎯",
          title: "Ready!",
          description: "Your personalized course recommendations",
          estimatedTime: "Complete",
          actualTime: "",
        },
      ]);

      // Close any existing EventSource
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    if (isCompleted) {
      setCurrentStep(steps.length - 1);
      return;
    }
  }, [isActive, isCompleted]);

  // Setup SSE connection when active and query is provided
  useEffect(() => {
    if (isActive && query && !eventSourceRef.current) {
      // Create EventSource for real-time progress updates
      const params = new URLSearchParams({
        query,
        use_explanation: "true",
        use_reranker: "true",
        use_llm_validation: "true",
        top_k: "5",
        n_candidates: "20",
      });
      const eventSource = new EventSource(`/api/recommend_multi_stream?${params.toString()}`);
      eventSourceRef.current = eventSource;

      // Listen for progress updates
      eventSource.onmessage = (event) => {
        try {
          const data: ProgressUpdate | { type: 'complete' | 'error', [key: string]: any } = JSON.parse(event.data);

          if (data.type === 'progress' && data.step_id !== undefined) {
            const progressUpdate = data as ProgressUpdate;
            progressHistoryRef.current.push(progressUpdate);
            const stepId = progressUpdate.step_id!; // Non-null assertion since we checked for undefined

            // Update current step based on backend progress
            if (progressUpdate.status === 'starting') {
              setCurrentStep(stepId);

              // Update step estimated time
              setSteps(prevSteps => {
                const newSteps = [...prevSteps];
                if (newSteps[stepId]) {
                  newSteps[stepId].estimatedTime = "In progress...";
                }
                return newSteps;
              });
            } else if (progressUpdate.status === 'completed') {
              // Update step with actual time
              setSteps(prevSteps => {
                const newSteps = [...prevSteps];
                if (newSteps[stepId]) {
                  newSteps[stepId].actualTime =
                    progressUpdate.time_taken ? `${progressUpdate.time_taken.toFixed(1)}s` : 'Completed';
                  newSteps[stepId].estimatedTime = "Complete";
                }
                return newSteps;
              });

              // Move to next step
              if (stepId < steps.length - 1) {
                setCurrentStep(stepId + 1);

                // Update next step to show it's starting
                setSteps(prevSteps => {
                  const newSteps = [...prevSteps];
                  if (newSteps[stepId + 1]) {
                    newSteps[stepId + 1].estimatedTime = "Starting...";
                  }
                  return newSteps;
                });
              }
            } else if (progressUpdate.status === 'error') {
              // Handle step error
              setSteps(prevSteps => {
                const newSteps = [...prevSteps];
                if (newSteps[stepId]) {
                  newSteps[stepId].estimatedTime = "Error";
                  newSteps[stepId].actualTime = "Failed";
                }
                return newSteps;
              });
            }

            if (onProgressUpdate) {
              onProgressUpdate(progressUpdate);
            }
          } else if (data.type === 'complete') {
            // Handle final results
            setCurrentStep(steps.length - 1);

            // Update all remaining steps as complete
            setSteps(prevSteps => {
              const newSteps = [...prevSteps];
              newSteps.forEach((step, index) => {
                if (index < steps.length - 1 && !step.actualTime) {
                  step.actualTime = 'Completed';
                  step.estimatedTime = 'Complete';
                }
              });
              return newSteps;
            });

            if (onComplete) {
              onComplete(data);
            }

            // Close EventSource
            eventSource.close();
            eventSourceRef.current = null;
          } else if (data.type === 'error') {
            console.error('Search error:', data.error);
            if (onError) {
              onError(data.error || 'Unknown error');
            }

            // Close EventSource
            eventSource.close();
            eventSourceRef.current = null;
          }
        } catch (error) {
          console.error('Error parsing SSE data:', error);
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        if (onError) onError('Connection error');

        // Close EventSource on error
        eventSource.close();
        eventSourceRef.current = null;
      };

      // Cleanup on unmount
      return () => {
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
          eventSourceRef.current = null;
        }
      };
    }
  }, [isActive, query, onComplete, onError, onProgressUpdate]);

  if (!isActive && !isCompleted) return null;

  return (
    <div className="search-progress-timeline">
      <div className="timeline-header">
        <h6 className="text-muted mb-3">
          {isCompleted
            ? "Multi-Query AI Search Complete"
            : "Multi-Query AI Search in Progress"}
        </h6>
      </div>

      <div className="timeline-steps">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={`timeline-step ${index === currentStep ? "active" : ""} ${index < currentStep ? "completed" : ""}`}
          >
            <div className="timeline-step-content">
              <div className="timeline-step-icon">
                <span className="step-icon">{step.icon}</span>
                {index < currentStep && (
                  <span className="step-checkmark">✓</span>
                )}
              </div>

              <div className="timeline-step-text">
                <div className="timeline-step-header">
                  <span className="step-title">{step.title}</span>
                  <span className="step-time">
                    {index < currentStep
                      ? step.actualTime
                      : index === currentStep
                        ? step.estimatedTime
                        : step.estimatedTime}
                  </span>
                </div>
                <div className="step-description">{step.description}</div>
              </div>
            </div>

            {index < steps.length - 1 && (
              <div
                className={`timeline-connector ${index < currentStep ? "completed" : ""}`}
              ></div>
            )}
          </div>
        ))}
      </div>

      <div className="timeline-footer">
        <small className="text-muted">
          {(() => {
            // Calculate actual elapsed time from progress history
            if (progressHistoryRef.current.length > 0) {
              const firstUpdate = progressHistoryRef.current[0];
              const elapsed = firstUpdate.timestamp ?
                ((Date.now() - firstUpdate.timestamp) / 1000).toFixed(1) : '0.0';
              const completedSteps = steps.filter(step => step.actualTime).length;
              const totalSteps = steps.length - 1; // Exclude "Ready!" step

              if (completedSteps === totalSteps) {
                return `Completed in ${elapsed}s • AI-powered multi-query analysis`;
              } else {
                return `Elapsed: ${elapsed}s • ${completedSteps}/${totalSteps} steps completed • AI-powered analysis`;
              }
            }
            return "AI-powered multi-query analysis in progress...";
          })()}
        </small>
      </div>
    </div>
  );
};

export default SearchProgress;
