#!/usr/bin/env python3
"""
Skills4Deca Course Recommendation Backend

AI-powered course discovery using direct reranking pipeline:
- Multi-query expansion (GLM-4.6)
- Parallel direct reranking (Qwen3-Reranker-4B)
- LLM validation and explanation generation (GLM-4.6)

No vector embeddings or semantic search - uses reranker API directly.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
    stream_with_context,
)
from flask_cors import CORS
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()

# Set up logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "course_recommendation.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

# ============================================================================
# Configuration
# ============================================================================

# DeepInfra API Configuration
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
if not DEEPINFRA_API_KEY:
    logging.error("DEEPINFRA_API_KEY environment variable is not set!")
    raise ValueError("DEEPINFRA_API_KEY is required")

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"

# Fireworks AI API Configuration (for faster LLM inference)
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_LLM_MODEL = "fireworks/glm-4p6"  # Faster GLM model

# AI Models
LLM_MODEL = (
    "zai-org/GLM-4.6"  # For query expansion, validation, explanations (DeepInfra)
)
RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"  # For direct reranking
RERANKER_URL = "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B"


# LLM Provider Selection
def env_flag(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


PREFER_FIREWORKS_FOR_LLM = env_flag(os.getenv("USE_FIREWORKS_FOR_LLM"), default=True)

USE_FIREWORKS_FOR_LLM = bool(
    FIREWORKS_API_KEY
    and FIREWORKS_API_KEY != "your_fireworks_api_key_here"
    and PREFER_FIREWORKS_FOR_LLM
)

# Performance settings
MAX_PARALLEL_RERANKING_WORKERS = 3  # Parallel query processing
DEFAULT_TOP_K = 5  # Default number of recommendations
DEFAULT_EXPLANATION = "This course matches your search criteria and may be relevant to your professional development goals."
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 1.0  # seconds

# ============================================================================
# Flask App Initialization
# ============================================================================

folder_path = os.path.dirname(os.path.abspath(__file__))

# Try multiple possible static folder locations
static_folder_candidates = [
    os.path.join(folder_path, "static"),  # Production build location
    os.path.join(folder_path, "react-frontend", "dist"),  # Development location
]

static_folder = None
for candidate in static_folder_candidates:
    if os.path.exists(candidate):
        static_folder = candidate
        break

# If no static folder found, use default
if static_folder is None:
    static_folder = os.path.join(folder_path, "static")
    logging.warning(f"Static folder not found, using default: {static_folder}")

app = Flask(
    __name__,
    static_folder=static_folder,
    static_url_path="",
)
CORS(app)

logging.info(f"Static folder set to: {static_folder}")

# ============================================================================
# Global Variables
# ============================================================================

deepinfra_client = None
fireworks_client = None
course_metadata_cache = {}  # Cache for full course metadata
visualization_cache = None  # Cache for precomputed visualization data

# ============================================================================
# Pydantic Models for Structured LLM Responses
# ============================================================================


class QueryExpansion(BaseModel):
    """Structured model for query expansion response."""

    original_query: str = Field(..., description="The original user query")
    variations: List[str] = Field(..., description="List of 4 query variations")


class SelectedCourse(BaseModel):
    """Structured representation of a validated course."""

    course_index: int = Field(..., ge=1, description="1-based index of the course")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score between 0 and 1"
    )
    selection_reason: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=1)


class CourseValidation(BaseModel):
    """Structured model for LLM validation response."""

    selected_courses: List[SelectedCourse] = Field(
        ..., description="List of validated courses with scores and reasons"
    )


# ============================================================================
# Utility Functions
# ============================================================================


def truncate_at_sentence(text: str, max_chars: int = 500) -> str:
    """Intelligently truncate text at sentence boundary."""
    text = text or ""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    last_exclaim = truncated.rfind("!")
    last_question = truncated.rfind("?")
    boundary = max(last_period, last_exclaim, last_question)

    if boundary > max_chars * 0.7:  # At least 70% of max_chars
        return text[: boundary + 1]
    else:
        return truncated + "..."


def normalize_json_content(content: str) -> str:
    """Remove common formatting wrappers (e.g., markdown fences) from JSON text."""
    if not content:
        return ""

    cleaned = content.strip()

    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
        else:
            cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()

    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    return cleaned


def validate_model(model_cls: type[BaseModel], data: Any) -> BaseModel:
    """Validate data against a Pydantic model with v1/v2 compatibility."""
    if hasattr(model_cls, "model_validate"):
        # Pydantic v2
        return model_cls.model_validate(data)
    # Pydantic v1 fallback
    return model_cls.parse_obj(data)


def sse_message(payload: Dict[str, Any]) -> str:
    """Format a payload as an SSE-compatible data message with timestamp."""
    payload = dict(payload)  # shallow copy to avoid mutating callers
    payload.setdefault("timestamp", int(time.time() * 1000))
    return f"data: {json.dumps(payload)}\n\n"


# ============================================================================
# Initialization Functions
# ============================================================================


def initialize_clients() -> None:
    """Initialize the DeepInfra and Fireworks AI clients."""
    global deepinfra_client, fireworks_client

    if not DEEPINFRA_API_KEY:
        deepinfra_client = None
        logging.warning("DeepInfra client not initialized because API key is missing.")
    else:
        deepinfra_client = OpenAI(
            api_key=DEEPINFRA_API_KEY, base_url=DEEPINFRA_BASE_URL
        )
        logging.info("DeepInfra client initialized successfully.")

    # Initialize Fireworks AI client if key is available
    if USE_FIREWORKS_FOR_LLM:
        fireworks_client = OpenAI(
            api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL
        )
        logging.info(
            f"Fireworks AI client initialized successfully. Using model: {FIREWORKS_LLM_MODEL}"
        )
    else:
        fireworks_client = None
        if FIREWORKS_API_KEY and FIREWORKS_API_KEY != "your_fireworks_api_key_here":
            logging.info(
                "Fireworks API key detected but USE_FIREWORKS_FOR_LLM flag is disabled. Using DeepInfra."
            )
        else:
            logging.info(
                "Fireworks AI not configured. Using DeepInfra for all LLM calls."
            )


def get_llm_client() -> Tuple[Optional[OpenAI], str]:
    """
    Get the appropriate LLM client and model name.
    Returns: (client, model_name)
    - Fireworks AI (GLM) as default primary provider
    - DeepInfra as automatic fallback if Fireworks fails
    """
    if USE_FIREWORKS_FOR_LLM and fireworks_client:
        return (fireworks_client, FIREWORKS_LLM_MODEL)
    elif deepinfra_client:
        return (deepinfra_client, LLM_MODEL)
    else:
        return (None, "")


def get_llm_client_with_fallback() -> Tuple[Optional[OpenAI], str, str]:
    """
    Get LLM client with provider identification for fallback handling.
    Returns: (client, model_name, provider_name)
    provider_name: "fireworks" or "deepinfra"
    """
    if USE_FIREWORKS_FOR_LLM and fireworks_client:
        return (fireworks_client, FIREWORKS_LLM_MODEL, "fireworks")
    elif deepinfra_client:
        return (deepinfra_client, LLM_MODEL, "deepinfra")
    else:
        return (None, "", "none")


def get_fallback_client() -> Tuple[Optional[OpenAI], str, str]:
    """
    Get the fallback LLM client when primary provider fails.
    Returns: (client, model_name, provider_name)
    """
    # Try DeepInfra if Fireworks was primary
    if USE_FIREWORKS_FOR_LLM and fireworks_client and deepinfra_client:
        return (deepinfra_client, LLM_MODEL, "deepinfra")
    # Try Fireworks if DeepInfra was primary
    elif not USE_FIREWORKS_FOR_LLM and deepinfra_client and fireworks_client:
        return (fireworks_client, FIREWORKS_LLM_MODEL, "fireworks")
    else:
        return (None, "", "none")


def load_course_metadata_cache() -> None:
    """Load course metadata from disk."""
    global course_metadata_cache

    try:
        metadata_path = os.path.join(folder_path, "course_metadata_cache.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        courses = data.get("courses") or data
        if not isinstance(courses, dict):
            raise ValueError("Invalid course metadata format.")

        course_metadata_cache = courses
        logging.info(
            f"Loaded {len(course_metadata_cache)} courses from metadata cache."
        )

    except FileNotFoundError:
        logging.error(f"Course metadata cache not found at {metadata_path}")
        course_metadata_cache = {}
    except Exception as e:
        logging.error(f"Error loading course metadata cache: {e}")
        course_metadata_cache = {}


def load_all_courses_for_reranking() -> List[Dict[str, Any]]:
    """
    Prepare a list of all courses for direct reranking.
    Each course contains text representation for the reranker.
    """
    if not course_metadata_cache:
        logging.warning("Course metadata cache is empty.")
        return []

    courses = []
    for course_id, metadata in course_metadata_cache.items():
        # Build text representation for reranking
        enhanced_desc = metadata.get("enhanced_description", "")
        original_desc = metadata.get("description", "")
        topics = metadata.get("topics_covered", [])

        course_title = (
            metadata.get("course_title") or metadata.get("title") or "Unknown Course"
        )
        university_name = (
            metadata.get("university")
            or metadata.get("university_name")
            or "Unknown University"
        )
        university_short = (
            metadata.get("university_short") or metadata.get("university_abbr") or ""
        )

        # Combine for rich text representation
        topics_text = " | ".join(topics) if topics else ""
        text = f"{enhanced_desc or original_desc}\n\nTopics: {topics_text}"

        course = {
            "course_id": course_id,
            "course_title": course_title,
            "university": university_name,
            "university_short": university_short,
            "text": text,
            "enhanced_description": enhanced_desc,
            "description": original_desc,
            "topics_covered": topics,
        }
        courses.append(course)

    logging.info(f"Prepared {len(courses)} courses for reranking.")
    return courses


def load_precomputed_visualization() -> None:
    """Load precomputed visualization data if available."""
    global visualization_cache

    visualization_path = os.path.join(
        folder_path, "visualization_data_precomputed.json"
    )

    if not os.path.exists(visualization_path):
        logging.warning(f"Visualization data file not found at {visualization_path}")
        visualization_cache = None
        return

    try:
        with open(visualization_path, "r", encoding="utf-8") as f:
            visualization_cache = json.load(f)
        logging.info("Loaded precomputed visualization data successfully.")
    except Exception as e:
        logging.error(f"Error loading visualization data: {e}")
        visualization_cache = None


# ============================================================================
# Query Expansion (Multi-Query Generation)
# ============================================================================


def expand_query_with_llm(user_query: str) -> Tuple[List[str], float]:
    """
    Expand a user query into 4 construction-focused variations using GLM model.
    Uses Fireworks AI as default, with automatic fallback to DeepInfra if Fireworks fails.
    """
    # Get primary client
    client, model, provider = get_llm_client_with_fallback()
    if not client:
        raise RuntimeError("No LLM client available for query expansion.")

    prompt = f"""You are a construction education expert. Expand this query into 4 focused variations for finding relevant construction courses.

Original query: "{user_query}"

Return ONLY valid JSON in this exact format:
{{{{
  "original_query": "{user_query}",
  "variations": [
    "variation 1",
    "variation 2",
    "variation 3",
    "variation 4"
  ]
}}}}

Make variations specific to construction, buildings, infrastructure, and professional development."""

    start_time = time.time()
    last_error: Optional[Exception] = None
    current_provider = provider

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            content = response.choices[0].message.content or ""
            logging.info(
                f"Raw query expansion response from {current_provider} (attempt {attempt}): {content[:500]}"
            )
            cleaned = normalize_json_content(content)
            data = json.loads(cleaned)
            expansion = validate_model(QueryExpansion, data)

            variations = [
                variation.strip()
                for variation in expansion.variations
                if variation.strip()
            ]
            if len(variations) < 4:
                raise ValueError(
                    f"Expected at least 4 variations, received {len(variations)}."
                )

            elapsed = time.time() - start_time
            logging.info(
                f"Query expansion successful in {elapsed:.2f}s using {current_provider} on attempt {attempt}: {variations[:4]}"
            )
            return variations[:4], elapsed

        except (json.JSONDecodeError, ValidationError, ValueError) as parse_error:
            last_error = parse_error
            logging.warning(
                f"Query expansion attempt {attempt} failed to parse JSON from {current_provider}: {parse_error}"
            )
        except Exception as unexpected_error:
            last_error = unexpected_error
            logging.error(
                f"Unexpected error during query expansion attempt {attempt} with {current_provider}: {unexpected_error}",
                exc_info=True,
            )

        # If this is the first failure and we haven't tried fallback yet, try fallback provider
        if attempt == 1 and current_provider != "none":
            fallback_client, fallback_model, fallback_provider = get_fallback_client()
            if fallback_client and fallback_provider != current_provider:
                logging.info(f"Primary provider {current_provider} failed, switching to fallback provider {fallback_provider}")
                client, model, current_provider = fallback_client, fallback_model, fallback_provider
                # Reset attempt counter for new provider
                attempt = 0

        if attempt < LLM_MAX_RETRIES:
            delay = LLM_RETRY_BASE_DELAY * attempt
            logging.info(
                f"Retrying query expansion in {delay:.1f}s with {current_provider} (attempt {attempt + 1})"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Query expansion failed after {LLM_MAX_RETRIES} attempts across providers: {last_error}"
    )


# ============================================================================
# Direct Reranking (Core Search Function)
# ============================================================================


def perform_direct_reranking(
    query: str,
    courses: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Call the DeepInfra reranker to score courses for a given query.
    No embeddings - direct reranking via API.
    """
    if not courses:
        return []

    documents = [course.get("text", "") for course in courses]
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "queries": [query],
        "documents": documents,
        "top_n": min(top_k, len(documents)),
    }

    try:
        response = requests.post(
            RERANKER_URL,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        # API returns scores array (one per document)
        scores = result.get("scores", [])

        reranked_courses = []
        for idx, score in enumerate(scores):
            if idx < len(courses):
                course = courses[idx].copy()
                course["reranker_score"] = float(score)
                reranked_courses.append(course)

        # Sort by reranker score descending and take top_k
        reranked_courses.sort(key=lambda x: x.get("reranker_score", 0.0), reverse=True)
        reranked_courses = reranked_courses[:top_k]

        logging.info(
            f"Reranked {len(reranked_courses)} courses for query: '{query[:50]}...'"
        )
        return reranked_courses

    except Exception as e:
        logging.error(f"Error in reranking: {e}")
        return []


# ============================================================================
# Parallel Multi-Query Search
# ============================================================================


def perform_multi_query_search(
    query_variations: List[str],
    all_courses: List[Dict[str, Any]],
    top_k_per_query: int = 5,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Perform parallel multi-query search with multiple query variations.
    Returns combined results from all queries with deduplication.
    """
    if not query_variations:
        return [], 0.0

    logging.info(
        f"Starting parallel multi-query search with {len(query_variations)} variations"
    )

    # Use ThreadPoolExecutor for parallel reranking
    max_workers = min(MAX_PARALLEL_RERANKING_WORKERS, len(query_variations))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all reranking tasks in parallel
        future_to_variation = {
            executor.submit(
                perform_direct_reranking, variation, all_courses, top_k_per_query
            ): (index, variation)
            for index, variation in enumerate(query_variations, start=1)
        }

        # Process results as they complete
        all_results = []
        query_times = []

        for future in as_completed(future_to_variation):
            index, variation = future_to_variation[future]
            start_time = time.time()

            try:
                reranked_courses = future.result()
                elapsed = time.time() - start_time
                query_times.append(elapsed)

                logging.info(
                    f"Variation {index}/{len(query_variations)} '{variation}' returned {len(reranked_courses)} courses in {elapsed:.2f}s"
                )

                # Add query metadata to each result
                for course in reranked_courses:
                    course["source_query"] = variation
                    course["source_query_index"] = index

                all_results.extend(reranked_courses)

            except Exception as exc:
                logging.error(
                    f"Parallel reranking failed for variation '{variation}': {exc}"
                )
                query_times.append(time.time() - start_time)

    # Deduplicate results by course_id
    seen_course_ids = set()
    deduplicated_results = []

    for course in all_results:
        course_id = course.get("course_id")
        if course_id and course_id not in seen_course_ids:
            seen_course_ids.add(course_id)
            deduplicated_results.append(course)

    # Sort by reranker score
    deduplicated_results.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)

    total_time = sum(query_times) if query_times else 0.0
    logging.info(f"Parallel multi-query search complete:")
    logging.info(f"  Total raw results: {len(all_results)}")
    logging.info(f"  After deduplication: {len(deduplicated_results)}")

    return deduplicated_results, total_time


# ============================================================================
# LLM Validation
# ============================================================================


def validate_with_llm(
    query: str,
    reranked_courses: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Use GLM-4.6 to validate, select, and generate explanations for courses.
    Combined validation + explanation in one API call for better performance.
    Returns top_k validated courses with relevance scores and explanations.
    Uses Fireworks AI as default, with automatic fallback to DeepInfra if Fireworks fails.
    """
    if not reranked_courses:
        return []

    # Get primary client
    client, model, provider = get_llm_client_with_fallback()
    if not client:
        raise RuntimeError("No LLM client available for validation.")

    course_summaries = []
    for idx, course in enumerate(reranked_courses[:20], 1):
        summary = {
            "course_index": idx,
            "course_id": course.get("course_id"),
            "title": course.get("course_title"),
            "university": course.get("university"),
            "description": truncate_at_sentence(
                course.get("enhanced_description") or course.get("description", ""),
                max_chars=300,
            ),
            "topics_covered": course.get("topics_covered", [])[:15],
        }
        course_summaries.append(summary)

    sample_output = {
        "selected_courses": [
            {
                "course_index": 1,
                "relevance_score": 0.95,
                "selection_reason": "This course covers quantum computing applications in construction.",
                "explanation": "This course will help you understand how quantum computing can be applied to construction project management, risk analysis, and optimization of complex scenarios.",
            }
        ]
    }

    prompt_parts = [
        "You are an expert course advisor. Analyze these courses for DIRECT relevance to the user's query.",
        "",
        f'User Query: "{query}"',
        "",
        "Available Courses:",
        json.dumps(course_summaries, indent=2),
        "",
        "Each course includes:",
        "- title: Course name",
        "- description: Course overview",
        "- topics_covered: List of specific topics taught in the course",
        "",
        "STRICT SELECTION CRITERIA:",
        "- ONLY select courses that DIRECTLY match the query's specific topics or technologies",
        "- If the query asks for specific technologies (e.g., 'LLM', 'NLP', 'blockchain'), ONLY return courses that explicitly mention those technologies in their description or topics_covered",
        "- DO NOT include loosely related or tangentially relevant courses",
        "- DO NOT include prerequisite or foundational courses unless they directly address the query",
        "- If NO courses directly match, return an empty list - this is acceptable and preferred over returning irrelevant courses",
        f"- You may return 0 to {top_k} courses - quality over quantity",
        "",
        "For each selected course, provide:",
        "- course_index: The index number from the list above",
        "- relevance_score: A score from 0.0 to 1.0 (only courses scoring 0.7+ should be selected)",
        "- selection_reason: Brief explanation citing SPECIFIC matching topics/technologies from the course",
        "- explanation: Personalized 2-3 sentence recommendation explaining why the user should take this course",
        "",
        "Return ONLY valid JSON in this format:",
        json.dumps(sample_output, indent=2),
        "",
        "Remember: It is better to return 0-2 highly relevant courses than to include loosely related ones.",
    ]

    prompt = "\n".join(prompt_parts)

    last_error: Optional[Exception] = None
    current_provider = provider

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )

            content = response.choices[0].message.content or ""
            logging.info(
                f"Raw LLM validation response from {current_provider} (first 500 chars): {content[:500]}"
            )

            cleaned = normalize_json_content(content)
            data = json.loads(cleaned)
            validation = validate_model(CourseValidation, data)

            # Empty list is now acceptable if no courses are truly relevant
            if not validation.selected_courses:
                logging.info(
                    f"LLM validation returned 0 courses - no directly relevant matches found."
                )
                return []

            validated_courses: List[Dict[str, Any]] = []
            for item in validation.selected_courses:
                idx = item.course_index - 1
                if not (0 <= idx < len(reranked_courses)):
                    raise ValueError(
                        f"Invalid course_index {item.course_index} in response."
                    )

                course = reranked_courses[idx].copy()
                course["llm_relevance_score"] = item.relevance_score
                course["llm_selection_reason"] = item.selection_reason
                course["explanation"] = item.explanation
                validated_courses.append(course)

            logging.info(
                f"LLM validation successful using {current_provider} on attempt {attempt}: {len(validated_courses)} courses"
            )
            return validated_courses[:top_k]

        except (json.JSONDecodeError, ValidationError, ValueError) as parse_error:
            last_error = parse_error
            logging.warning(f"Validation attempt {attempt} failed to parse JSON from {current_provider}: {parse_error}")
        except Exception as unexpected_error:
            last_error = unexpected_error
            logging.error(
                f"Unexpected error during validation attempt {attempt} with {current_provider}: {unexpected_error}",
                exc_info=True,
            )

        # If this is the first failure and we haven't tried fallback yet, try fallback provider
        if attempt == 1 and current_provider != "none":
            fallback_client, fallback_model, fallback_provider = get_fallback_client()
            if fallback_client and fallback_provider != current_provider:
                logging.info(f"Primary provider {current_provider} failed, switching to fallback provider {fallback_provider}")
                client, model, current_provider = fallback_client, fallback_model, fallback_provider
                # Reset attempt counter for new provider
                attempt = 0

        if attempt < LLM_MAX_RETRIES:
            delay = LLM_RETRY_BASE_DELAY * attempt
            logging.info(f"Retrying validation in {delay:.1f}s with {current_provider} (attempt {attempt + 1})")
            time.sleep(delay)

    raise RuntimeError(
        f"LLM validation failed after {LLM_MAX_RETRIES} attempts across providers: {last_error}"
    )


# ============================================================================
# Explanation Generation
# ============================================================================


def generate_explanation(query: str, course: Dict[str, Any]) -> str:
    """Generate a personalized explanation for a course with fallback support."""
    # Get primary client
    client, model, provider = get_llm_client_with_fallback()
    if not client:
        return DEFAULT_EXPLANATION

    course_title = course.get("course_title", "Unknown course")
    course_desc = course.get("enhanced_description") or course.get("description") or ""

    prompt = f"""You are an expert advisor. Explain in 2-3 sentences why this course is relevant to the user.

User interest: {query}

Course: {course_title}
Description: {truncate_at_sentence(course_desc, 300)}

Write a personalized explanation focused on how this course addresses the user's needs."""

    current_provider = provider

    # Try primary provider first, then fallback if needed
    for attempt in range(1, 3):  # Max 2 attempts: primary + fallback
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful course advisor."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=500,
            )

            explanation = response.choices[0].message.content.strip()
            logging.info(f"Explanation generated using {current_provider}")
            return explanation

        except Exception as e:
            logging.warning(f"Error generating explanation with {current_provider}: {e}")

            # Try fallback provider on first attempt
            if attempt == 1 and current_provider != "none":
                fallback_client, fallback_model, fallback_provider = get_fallback_client()
                if fallback_client and fallback_provider != current_provider:
                    logging.info(f"Switching to fallback provider {fallback_provider}")
                    client, model, current_provider = fallback_client, fallback_model, fallback_provider
                    continue

            # If fallback fails or no fallback available, return default
            break

    logging.error("Failed to generate explanation with all providers")
    return DEFAULT_EXPLANATION


def generate_explanations_parallel(
    query: str,
    courses: List[Dict[str, Any]],
    max_workers: Optional[int] = None,
) -> None:
    """Generate explanations concurrently for courses (modifies in place)."""
    if not courses:
        return

    workers = max_workers or min(3, len(courses))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(generate_explanation, query, course): idx
            for idx, course in enumerate(courses)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                explanation = future.result()
                courses[idx]["explanation"] = explanation
            except Exception as exc:
                logging.error(f"Explanation generation failed for course {idx}: {exc}")
                courses[idx]["explanation"] = DEFAULT_EXPLANATION


# ============================================================================
# Response Formatting
# ============================================================================


def format_course_response(course: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a course dictionary for API responses."""
    course_id = course.get("course_id", "")
    description = course.get("enhanced_description") or course.get("description") or ""
    summary = truncate_at_sentence(description, max_chars=500)

    return {
        "course_id": course_id,
        "course_title": course.get("course_title", "Unknown Course"),
        "university": course.get("university", "Unknown University"),
        "university_short": course.get("university_short", ""),
        "description": description,
        "course_summary": summary,
        "reranker_score": float(course.get("reranker_score", 0.0)),
        "llm_relevance_score": (
            float(course.get("llm_relevance_score"))
            if course.get("llm_relevance_score") is not None
            else None
        ),
        "llm_selection_reason": course.get("llm_selection_reason"),
        "theme_match": course.get("reranker_score", 0.0) * 100,
        "subtopic_coverage": course.get(
            "llm_relevance_score", course.get("reranker_score", 0.0)
        )
        * 100,
        "explanation": course.get("explanation", DEFAULT_EXPLANATION),
        "source_query": course.get("source_query", ""),
        "topics_covered": course.get("topics_covered", []),
    }


# ============================================================================
# Main Recommendation Pipeline
# ============================================================================


def run_recommendation_pipeline(
    query: str,
    top_k: int = 5,
    use_explanation: bool = True,
    use_llm_validation: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full recommendation pipeline.

    Steps:
    1. Query expansion (4 variations)
    2. Load all courses
    3. Parallel multi-query reranking
    4. LLM validation (optional)
    5. Explanation generation (optional)
    """
    query = (query or "").strip()
    if not query:
        return {"error": "Query is required", "recommendations": []}

    start_time = time.time()

    # Step 1: Query expansion
    logging.info(f"Processing recommendation request: '{query}'")
    query_variations, expansion_time = expand_query_with_llm(query)

    # Step 2: Load courses
    load_start = time.time()
    all_courses = load_all_courses_for_reranking()
    load_time = time.time() - load_start

    if not all_courses:
        return {
            "recommendations": [],
            "message": "No courses available",
            "total_results": 0,
        }

    # Step 3: Multi-query search
    multi_query_start = time.time()
    multi_query_results, multi_query_time = perform_multi_query_search(
        query_variations, all_courses, top_k_per_query=5
    )

    # Step 4: LLM Validation (includes explanations if use_explanation=True)
    if use_llm_validation:
        llm_validation_start = time.time()
        final_courses = validate_with_llm(query, multi_query_results, top_k=top_k)
        llm_validation_time = time.time() - llm_validation_start
        # Explanations already generated by validate_with_llm
        explanation_time = 0
    else:
        final_courses = multi_query_results[:top_k]
        llm_validation_time = 0

        # Step 5: Generate explanations (only if LLM validation not used)
        if use_explanation:
            explanation_start = time.time()
            generate_explanations_parallel(query, final_courses)
            explanation_time = time.time() - explanation_start
        else:
            explanation_time = 0

    # Sort final courses by strongest AI relevance (fallback to reranker score)
    final_courses.sort(
        key=lambda course: course.get(
            "llm_relevance_score", course.get("reranker_score", 0.0)
        ),
        reverse=True,
    )

    # Format response
    recommendations = [format_course_response(course) for course in final_courses]
    total_time = time.time() - start_time

    # Log performance
    logging.info(f"Recommendation complete in {total_time:.3f}s:")
    logging.info(f"  Query expansion: {expansion_time:.3f}s")
    logging.info(f"  Course loading: {load_time:.3f}s")
    logging.info(f"  Multi-query search: {multi_query_time:.3f}s")
    logging.info(f"  LLM Validation: {llm_validation_time:.3f}s")
    logging.info(f"  Explanations: {explanation_time:.3f}s")
    logging.info(f"  Returned {len(recommendations)} recommendations")

    return {
        "recommendations": recommendations,
        "query": query,
        "query_variations": query_variations,
        "total_results": len(recommendations),
        "search_method": "Direct Reranking with Multi-Query Analysis",
        "processing_time": {
            "total": float(total_time),
            "query_expansion": float(expansion_time),
            "course_loading": float(load_time),
            "multi_query_search": float(multi_query_time),
            "llm_validation": float(llm_validation_time),
            "explanations": float(explanation_time),
        },
    }


# ============================================================================
# Flask Routes
# ============================================================================


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    """Serve the React frontend or static assets."""
    if path.startswith("api/"):
        return jsonify({"error": "API endpoint not found"}), 404

    if not os.path.exists(app.static_folder):
        return (
            jsonify(
                {
                    "error": "Frontend build not found.",
                    "static_folder": app.static_folder,
                    "message": "Run 'cd react-frontend && npm run build' to build the frontend.",
                }
            ),
            500,
        )

    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health_check():
    """Health check endpoint."""
    # Get current provider info
    _, _, current_provider = get_llm_client_with_fallback()
    _, _, fallback_provider = get_fallback_client()

    return jsonify(
        {
            "status": "healthy",
            "api_keys_configured": {
                "deepinfra": bool(DEEPINFRA_API_KEY),
                "fireworks": bool(FIREWORKS_API_KEY and FIREWORKS_API_KEY != "your_fireworks_api_key_here"),
            },
            "clients_initialized": {
                "deepinfra": deepinfra_client is not None,
                "fireworks": fireworks_client is not None,
            },
            "llm_provider": {
                "current": current_provider,
                "fallback": fallback_provider if fallback_provider != "none" else None,
                "prefer_fireworks": PREFER_FIREWORKS_FOR_LLM,
            },
            "course_metadata_loaded": len(course_metadata_cache),
            "visualization_cached": visualization_cache is not None,
            "models": {
                "llm_primary": FIREWORKS_LLM_MODEL if current_provider == "fireworks" else LLM_MODEL,
                "llm_fallback": LLM_MODEL if current_provider == "fireworks" else FIREWORKS_LLM_MODEL,
                "reranker": RERANKER_MODEL,
            },
        }
    )


@app.route("/api/recommend", methods=["POST"])
@app.route("/api/recommend_multi", methods=["POST"])  # Legacy alias
def recommend_courses():
    """
    Main recommendation endpoint using direct reranking pipeline.

    Request body:
    {
        "query": "user query string",
        "top_k": 5,  # optional
        "use_explanation": true,  # optional
        "use_llm_validation": true  # optional
    }
    """
    try:
        data = request.get_json(force=True) or {}
        query = (data.get("query") or data.get("prompt") or "").strip()

        if not query:
            return jsonify({"error": "Query parameter is required."}), 400

        top_k = int(data.get("top_k", DEFAULT_TOP_K))
        use_explanation = bool(data.get("use_explanation", True))
        use_llm_validation = bool(data.get("use_llm_validation", True))

        result = run_recommendation_pipeline(
            query=query,
            top_k=top_k,
            use_explanation=use_explanation,
            use_llm_validation=use_llm_validation,
        )

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except ValueError as e:
        logging.error(f"Value error in recommendation: {e}", exc_info=True)
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        logging.error(f"Error in recommendation: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/recommend_multi_stream", methods=["GET", "POST"])
def recommend_courses_stream():
    """
    Streaming recommendation endpoint with real-time progress updates.
    Returns Server-Sent Events (SSE) for progress tracking.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
    else:
        data = request.args.to_dict() if request.args else {}

    query = (data.get("query") or data.get("prompt") or "").strip()
    if not query:
        return jsonify({"error": "Query parameter is required."}), 400

    top_k = int(data.get("top_k", DEFAULT_TOP_K))
    use_explanation = data.get("use_explanation", "true").lower() == "true"
    use_llm_validation = data.get("use_llm_validation", "true").lower() == "true"

    def generate_progress():
        """Generator for SSE progress updates."""
        start_time = time.time()
        try:
            # Step 0: Query expansion
            yield sse_message(
                {
                    "type": "progress",
                    "step_id": 0,
                    "step_name": "query_expansion",
                    "status": "starting",
                    "message": "Expanding your query with construction-focused variations...",
                }
            )
            # Send keep-alive to prevent proxy timeout during LLM call
            yield ": keep-alive\n\n"
            query_variations, expansion_time = expand_query_with_llm(query)
            yield sse_message(
                {
                    "type": "progress",
                    "step_id": 0,
                    "step_name": "query_expansion",
                    "status": "completed",
                    "time_taken": expansion_time,
                }
            )

            # Step 1: Smart discovery (loading + reranking)
            discovery_start = time.time()
            yield sse_message(
                {
                    "type": "progress",
                    "step_id": 1,
                    "step_name": "discovery",
                    "status": "starting",
                    "message": "Scanning all courses and running multi-query reranking...",
                }
            )
            # Send keep-alive to prevent proxy timeout during reranking
            yield ": keep-alive\n\n"

            load_start = time.time()
            all_courses = load_all_courses_for_reranking()
            load_time = time.time() - load_start

            multi_query_results, multi_query_time = perform_multi_query_search(
                query_variations, all_courses, top_k_per_query=5
            )
            discovery_time = time.time() - discovery_start
            yield sse_message(
                {
                    "type": "progress",
                    "step_id": 1,
                    "step_name": "discovery",
                    "status": "completed",
                    "time_taken": discovery_time,
                    "details": {
                        "load_time": load_time,
                        "rerank_time": multi_query_time,
                        "courses_loaded": len(all_courses),
                        "candidates": len(multi_query_results),
                    },
                }
            )

            # Step 4: LLM validation (includes explanations)
            llm_time = 0.0
            explanation_time = 0.0
            if use_llm_validation:
                yield sse_message(
                    {
                        "type": "progress",
                        "step_id": 2,
                        "step_name": "validation",
                        "status": "starting",
                        "message": "AI validating top courses and crafting explanations...",
                    }
                )
                # Send keep-alive to prevent proxy timeout during LLM validation
                yield ": keep-alive\n\n"
                llm_start = time.time()
                final_courses = validate_with_llm(
                    query, multi_query_results, top_k=top_k
                )
                llm_time = time.time() - llm_start
                yield sse_message(
                    {
                        "type": "progress",
                        "step_id": 2,
                        "step_name": "validation",
                        "status": "completed",
                        "time_taken": llm_time,
                    }
                )
            else:
                final_courses = multi_query_results[:top_k]
                if use_explanation:
                    exp_start = time.time()
                    generate_explanations_parallel(query, final_courses)
                    explanation_time = time.time() - exp_start
                yield sse_message(
                    {
                        "type": "progress",
                        "step_id": 2,
                        "step_name": "validation",
                        "status": "completed",
                        "message": (
                            "LLM validation disabled. Generated standard explanations instead."
                            if use_explanation
                            else "LLM validation disabled (no explanations generated)."
                        ),
                        "time_taken": explanation_time,
                    }
                )

            # Final results
            recommendations = [
                format_course_response(course) for course in final_courses
            ]
            total_time = time.time() - start_time
            result = {
                "type": "complete",
                "status": "success",
                "recommendations": recommendations,
                "total_results": len(recommendations),
                "search_method": "Direct Reranking with Multi-Query Analysis",
                "processing_time": {
                    "total": float(total_time),
                    "query_expansion": float(expansion_time),
                    "course_loading": float(load_time),
                    "multi_query_search": float(multi_query_time),
                    "llm_validation": float(llm_time),
                    "explanations": float(explanation_time),
                },
            }
            yield sse_message(result)

        except Exception as e:
            logging.error(f"Error in streaming recommendation: {e}")
            error_result = {"type": "error", "error": str(e)}
            yield sse_message(error_result)

    return Response(
        stream_with_context(generate_progress()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/visualization_data", methods=["GET"])
def get_visualization_data():
    """Serve precomputed visualization data."""
    if not visualization_cache:
        return (
            jsonify(
                {
                    "error": "Visualization data not available.",
                    "message": "Run 'python generate_visualization_data.py' to create it.",
                }
            ),
            404,
        )

    try:
        response_data = {
            "plot_data": visualization_cache.get("plot_data", []),
            "model": visualization_cache.get("model", "unknown"),
            "total_points": visualization_cache.get("total_points", 0),
            "reduction_method": visualization_cache.get("reduction_method", "unknown"),
            "umap_params": visualization_cache.get("umap_params", {}),
            "universities": visualization_cache.get("universities", []),
            "generated_at": visualization_cache.get("generated_at"),
            "performance": {
                "cached": True,
                "response_time_ms": "< 50ms",
            },
        }

        logging.info(f"Served {len(response_data['plot_data'])} visualization points")
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error serving visualization data: {e}")
        return jsonify({"error": "Failed to serve visualization data"}), 500


# ============================================================================
# Application Startup
# ============================================================================


def initialize_app():
    """Initialize all components at startup."""
    logging.info("=" * 80)
    logging.info("Skills4Deca Course Recommendation System - Starting Up")
    logging.info("=" * 80)

    initialize_clients()
    load_course_metadata_cache()
    load_precomputed_visualization()

    logging.info("=" * 80)
    logging.info("Initialization Complete")
    logging.info(f"Courses loaded: {len(course_metadata_cache)}")
    logging.info(f"Visualization cached: {visualization_cache is not None}")
    if USE_FIREWORKS_FOR_LLM and fireworks_client:
        logging.info(f"LLM Provider: Fireworks AI ({FIREWORKS_LLM_MODEL})")
    elif deepinfra_client:
        logging.info(f"LLM Provider: DeepInfra ({LLM_MODEL})")
    else:
        logging.info("LLM Provider: None (disabled)")
    logging.info("=" * 80)


# Initialize on import
initialize_app()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
