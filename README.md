# Skills4Deca

AI-powered course recommendation platform for construction professionals and students across the Baltic region.

## 🎯 Project Overview

**Skills4Deca** is an intelligent course discovery platform that helps construction professionals and students find relevant courses from three leading Baltic technical universities:

- **Tallinn University of Technology (TalTech)** - Estonia
- **Vilnius Technical University (VilniusTech)** - Lithuania
- **Riga Technical University (RTU)** - Latvia

The platform uses advanced AI technology with multi-query expansion and direct reranking to match professionals' learning interests with university courses through personalized recommendations and explanations.

### Key Features

- **🤖 Multi-Query AI Search** - Intelligent query expansion for comprehensive course discovery
- **⚡ Direct Reranking** - Advanced AI models for precise course matching
- **🔄 Automatic Fallback** - Fireworks AI as primary with DeepInfra backup for reliability
- **📊 Interactive Visualizations** - 2D UMAP course relationship mapping
- **🎯 Personalized Explanations** - AI-generated explanations for why each course matches
- **📱 Responsive Design** - Modern React frontend with glass morphism UI

### Course Coverage

The platform specializes in construction industry topics including:

- **Energy Systems** - Solar, HVAC, thermal management
- **Smart Buildings** - IoT, automation, digital twins
- **Advanced Materials** - Composites, sustainable concrete
- **Digital Technologies** - BIM, 3D scanning, simulation
- **Sustainability** - Circular economy, decarbonization

---

## 🏗️ Architecture

### Backend (Flask + Python)

**Technology Stack:**
- **Framework:** Flask with CORS support
- **AI Models:**
  - Primary: Fireworks AI (`fireworks/glm-4p6`)
  - Fallback: DeepInfra (`zai-org/GLM-4.6`)
  - Reranker: DeepInfra (`Qwen/Qwen3-Reranker-4B`)
- **Deployment:** Docker containerized on Render.com

**Core Pipeline:**
1. **Query Expansion** (4s) - AI generates 4 focused query variations
2. **Course Loading** (<1s) - Loads 49 courses with enhanced metadata
3. **Parallel Multi-Query Reranking** (36s) - Processes all queries simultaneously
4. **LLM Validation + Explanations** (4s) - Scores relevance and generates explanations
5. **Response Formatting** (<1s) - Final output preparation

**Performance:** ~44 seconds total (84% faster than previous implementation)

### Frontend (React + TypeScript)

**Technology Stack:**
- **Framework:** React 18 with TypeScript
- **Build Tool:** Vite
- **UI Library:** Bootstrap 5 with custom styling
- **Visualization:** Plotly.js for interactive course mapping
- **Routing:** React Router for SPA navigation

**Key Components:**
- **Search Interface** - Glass morphism design with guidance system
- **Progress Tracking** - Real-time pipeline status with streaming updates
- **Results Display** - Course cards with relevance scores and explanations
- **Visualization Page** - Interactive 2D UMAP projection of courses
- **Query Guidance** - Quick templates and best practices

---

## 🚀 Deployment

### Environment Setup

**Required Environment Variables:**
```bash
# AI Provider API Keys
DEEPINFRA_API_KEY=your_deepinfra_key_here
FIREWORKS_API_KEY=your_fireworks_key_here

# Provider Selection (optional, defaults to Fireworks)
USE_FIREWORKS_FOR_LLM=true

# Server Configuration
PORT=5000
FLASK_ENV=production
```

### Quick Start

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Skills4Deca
   ```

2. **Set Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Install Dependencies**
   ```bash
   # Backend
   pip install -r requirements-production.txt

   # Frontend
   cd react-frontend
   npm install
   ```

4. **Build Frontend**
   ```bash
   cd react-frontend
   npm run build
   cd ..
   ```

5. **Run Application**
   ```bash
   python3 app.py
   ```

### Docker Deployment

**Build Docker Image:**
```bash
docker build -t skills4deca .
```

**Run Container:**
```bash
docker run -p 5000:5000 --env-file .env skills4deca
```

### Render.com Deployment

The application is configured for Render.com deployment:

- **Dockerfile** - Multi-stage build with production optimizations
- **render.yaml** - Render deployment configuration
- **Health Checks** - `/api/health` endpoint for monitoring

**Deployment Configuration:**
- **Web Service:** Docker runtime
- **Build Command:** `docker build -t skills4deca .`
- **Start Command:** `docker run -p $PORT:$PORT --env-file .env skills4deca`

---

## 📡 API Documentation

### Core Endpoints

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "api_keys_configured": {
    "deepinfra": true,
    "fireworks": true
  },
  "clients_initialized": {
    "deepinfra": true,
    "fireworks": true
  },
  "llm_provider": {
    "current": "fireworks",
    "fallback": "deepinfra",
    "prefer_fireworks": true
  },
  "models": {
    "llm_primary": "fireworks/glm-4p6",
    "llm_fallback": "zai-org/GLM-4.6",
    "reranker": "Qwen/Qwen3-Reranker-4B"
  },
  "course_metadata_loaded": 49,
  "visualization_cached": true
}
```

#### Course Recommendation
```http
POST /api/recommend
Content-Type: application/json

{
  "query": "solar energy systems for buildings",
  "top_k": 5,
  "use_explanation": true,
  "use_llm_validation": true
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "course_id": "taltech_solar_energy_basics",
      "course_title": "Solar Energy Systems for Buildings",
      "university": "Tallinn University of Technology",
      "university_short": "TalTech",
      "description": "Comprehensive course on solar energy integration...",
      "course_summary": "Comprehensive course on solar energy integration...",
      "reranker_score": 0.89,
      "llm_relevance_score": 0.95,
      "llm_selection_reason": "Direct match for solar energy systems",
      "theme_match": 89.0,
      "subtopic_coverage": 95.0,
      "explanation": "This course provides essential knowledge...",
      "source_query": "solar energy systems",
      "topics_covered": ["solar panels", "energy storage", "building integration"]
    }
  ],
  "query": "solar energy systems for buildings",
  "query_variations": [
    "solar energy systems",
    "photovoltaic building integration",
    "renewable energy construction",
    "solar panel installation"
  ],
  "total_results": 5,
  "search_method": "Direct Reranking with Multi-Query Analysis",
  "processing_time": {
    "total": 44.2,
    "query_expansion": 4.1,
    "course_loading": 0.8,
    "multi_query_search": 36.3,
    "llm_validation": 4.4,
    "explanations": 0.0
  }
}
```

#### Streaming Recommendation (Real-time Progress)
```http
POST /api/recommend_multi_stream
Content-Type: application/json

{
  "query": "BIM modeling techniques",
  "top_k": 3,
  "use_explanation": true,
  "use_llm_validation": true
}
```

**Response Format:** Server-Sent Events (SSE) with real-time progress updates

#### Visualization Data
```http
GET /api/visualization_data
```

**Response:**
```json
{
  "plot_data": [...],
  "total_points": 49,
  "universities": ["TalTech", "VilniusTech", "RTU"],
  "performance": {
    "cached": true,
    "response_time_ms": "< 50ms"
  }
}
```

### Error Handling

**Standard Error Response:**
```json
{
  "error": "Query parameter is required."
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad Request (missing/invalid parameters)
- `500` - Internal Server Error (AI service failures, etc.)

---

## 📊 Data Structure

### Course Metadata

**Course Data Source:** `NEW_JSON/` directory
- **Total Courses:** 49 courses
- **TalTech:** 17 courses
- **VilniusTech:** 18 courses
- **RTU:** 14 courses

**Metadata Format:**
```json
{
  "course_id": "unique_identifier",
  "course_title": "Course Name",
  "university": "University Name",
  "university_short": "University Abbreviation",
  "description": "Course description",
  "enhanced_description": "SEO-optimized description",
  "topics_covered": ["topic1", "topic2", ...],
  "course_coordinator": "Coordinator Name",
  "credits": 6,
  "duration": "One semester"
}
```

### Performance Metrics

**Current Performance:**
- **Total Response Time:** ~44 seconds
- **Query Expansion:** 4.2 seconds
- **Multi-Query Search:** 36.3 seconds
- **LLM Validation:** 4.4 seconds
- **Performance Improvement:** 84% faster than previous implementation

**Optimizations Applied:**
- **Parallel Processing:** 66% faster than sequential queries
- **Combined Validation:** 73% faster than separate API calls
- **Precomputed Visualization:** <50ms response times
- **Provider Fallback:** Zero downtime with automatic switching

---

## 🔧 Configuration

### AI Provider Configuration

**Primary Provider:** Fireworks AI
- **Model:** `fireworks/glm-4p6`
- **Advantages:** 84% faster response times
- **Use Case:** Default for all LLM operations

**Fallback Provider:** DeepInfra
- **Model:** `zai-org/GLM-4.6`
- **Purpose:** Automatic fallback when Fireworks unavailable
- **Switching:** First failure attempt triggers fallback

### Performance Settings

```python
# Pipeline Configuration
MAX_PARALLEL_RERANKING_WORKERS = 3
DEFAULT_TOP_K = 5
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0  # seconds

# Model Configuration
RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"
TOP_K_PER_QUERY = 5
MAX_EXPLANATION_CANDIDATES = 20
```

### Environment Variables

**Required:**
- `DEEPINFRA_API_KEY` - DeepInfra API key
- `FIREWORKS_API_KEY` - Fireworks AI API key

**Optional:**
- `USE_FIREWORKS_FOR_LLM` - Prefer Fireworks over DeepInfra (default: true)
- `PORT` - Server port (default: 5000)
- `FLASK_ENV` - Environment mode (development/production)

---

## 🛠️ Development

### Project Structure

```
Skills4Deca/
├── app.py                          # Main Flask application
├── requirements.txt               # Development dependencies
├── requirements-production.txt    # Production dependencies
├── Dockerfile                     # Docker configuration
├── render.yaml                   # Render deployment config
├── .env.example                  # Environment variables template
├── course_metadata_cache.json   # Course data cache
├── visualization_data_precomputed.json  # Precomputed viz data
├── NEW_JSON/                     # Raw course data
│   ├── taltech_courses/
│   ├── vtech_courses/
│   └── rtu_courses/
├── static/                       # Built React frontend
└── react-frontend/               # Frontend source code
    ├── src/
    │   ├── components/
    │   ├── pages/
    │   ├── types/
    │   └── utils/
    ├── public/
    ├── package.json
    └── package-lock.json
```

### Build Process

**Frontend Build:**
```bash
cd react-frontend
npm install
npm run build
```

**Production Build Features:**
- TypeScript compilation
- Code minification and optimization
- Bundle analysis with Vite
- Static asset optimization

---

## 📈 Monitoring & Logging

### Application Logs

**Logging Configuration:**
- **Level:** INFO for production, DEBUG for development
- **Format:** Timestamp, level, message
- **Rotation:** File-based logging with console output

**Key Log Events:**
- AI provider switching
- Performance metrics
- Error conditions and recovery
- Query processing status

### Health Monitoring

**Health Check Endpoint:** `/api/health`
- API key validation
- Client initialization status
- Provider configuration
- Data cache status
- Model information

**Performance Metrics:**
- Response time tracking
- Provider success rates
- Error rate monitoring
- Cache hit ratios

---

## 🤝 Contributing

### Code Standards

**Python (Backend):**
- Follow PEP 8 style guidelines
- Type hints for all functions
- Comprehensive error handling
- Logging for all significant operations

**TypeScript (Frontend):**
- Strict TypeScript configuration
- Functional components with hooks
- Proper error boundaries
- Responsive design principles

### Testing

**Backend Testing:**
```bash
python3 -m pytest tests/
```

**Frontend Testing:**
```bash
cd react-frontend
npm test
```

---

## 📄 License

This project is proprietary software for construction industry education in the Baltic region.

---

## 📞 Support

For technical support or questions:

- **Technical Issues:** Check application logs and health endpoint
- **API Issues:** Verify API key configuration and rate limits
- **Performance:** Monitor processing times and provider status

**Last Updated:** November 2025
**Version:** 2.0 (Fireworks AI Integration)