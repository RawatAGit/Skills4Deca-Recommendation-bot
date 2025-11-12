import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom"; // Import router components
import "bootstrap/dist/css/bootstrap.min.css";
import "./index.css";
import App from "./App.tsx"; // Main recommendation page component
import VisualizationPage from "./pages/VisualizationPage"; // Visualization page component
// SimilarityPage component removed - using new hybrid RAG system

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Failed to find the root element");

createRoot(rootElement).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/visualization" element={<VisualizationPage />} />
        {/* Similarity page removed - using new hybrid RAG system */}
        {/* Add other routes here if needed */}
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
