"""
PoR Middleware for GitHub Copilot SDK
======================================

Interception layer implementing Silence-as-Control for Copilot SDK.
Provides drift/coherence gates before LLM responses are emitted.

Usage:
    from por_middleware import PorCopilotClient, SilenceResponse

    client = PorCopilotClient(
        drift_tolerance=0.1,
        coherence_threshold=0.7
    )

    response = client.complete("Write a function...")
    if isinstance(response, SilenceResponse):
        print("System chose to abstain due to instability")
    else:
        print(response.text)

Author: Anton Semenenko
License: MIT
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
from enum import Enum
import hashlib
import math
import time


# ============================================================================
# CORE TOKENS
# ============================================================================

class ControlToken(Enum):
    """Control tokens for PoR decision outcomes."""

    SILENCE = "SILENCE"
    PROCEED = "PROCEED"


@dataclass
class SilenceResponse:
    """
    Returned when the system chooses to abstain from responding.

    Silence is not an error - it's a valid, intentional control signal
    indicating that coherence could not be guaranteed.
    """

    reason: str
    drift: float
    coherence: float
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"<SILENCE reason='{self.reason}' drift={self.drift:.3f} coherence={self.coherence:.3f}>"

    def __bool__(self) -> bool:
        return False  # Silence is falsy


@dataclass
class ProceedResponse:
    """
    Wrapper for successful Copilot responses that passed PoR gates.
    """

    text: str
    drift: float
    coherence: float
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"<PROCEED coherence={self.coherence:.3f} len={len(self.text)}>"

    def __bool__(self) -> bool:
        return True  # Proceed is truthy


# ============================================================================
# METRICS
# ============================================================================

class DriftEstimator:
    """
    Estimates semantic drift between conversation turns.

    Uses rolling context window to detect when the conversation
    has drifted from the original topic/intent.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.context_hashes: List[str] = []
        self.embeddings_cache: Dict[str, List[float]] = {}

    def _simple_hash(self, text: str) -> str:
        """Create a simple hash for text comparison."""
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def _text_to_features(self, text: str) -> List[float]:
        """
        Extract simple features from text for drift detection.
        In production, replace with actual embeddings.
        """
        # Simple bag-of-words style features
        words = text.lower().split()
        word_set = set(words)

        features = [
            len(words),  # Length
            len(word_set),  # Vocabulary
            len(word_set) / max(len(words), 1),  # Lexical diversity
            sum(1 for w in words if len(w) > 6) / max(len(words), 1),  # Complex words
            text.count("?") + text.count("!"),  # Questions/exclamations
        ]
        return features

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two feature vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def update(self, text: str) -> None:
        """Add new text to the context window."""
        text_hash = self._simple_hash(text)
        self.context_hashes.append(text_hash)
        self.embeddings_cache[text_hash] = self._text_to_features(text)

        # Maintain window size
        if len(self.context_hashes) > self.window_size:
            old_hash = self.context_hashes.pop(0)
            if old_hash in self.embeddings_cache:
                del self.embeddings_cache[old_hash]

    def measure(self, new_text: str) -> float:
        """
        Measure drift of new text from conversation context.

        Returns:
            float: Drift score (0.0 = no drift, 1.0 = complete drift)
        """
        if not self.context_hashes:
            return 0.0  # No context = no drift

        new_features = self._text_to_features(new_text)

        # Compare against all context items
        similarities = []
        for ctx_hash in self.context_hashes:
            if ctx_hash in self.embeddings_cache:
                ctx_features = self.embeddings_cache[ctx_hash]
                sim = self._cosine_similarity(new_features, ctx_features)
                similarities.append(sim)

        if not similarities:
            return 0.0

        # Average similarity → drift
        avg_similarity = sum(similarities) / len(similarities)
        drift = 1.0 - avg_similarity

        return max(0.0, min(1.0, drift))


class CoherenceEstimator:
    """
    Estimates response coherence based on various heuristics.

    In production, this should use actual coherence models or
    embedding-based semantic similarity.
    """

    def __init__(self):
        self.incoherence_patterns = [
            "I don't know",
            "I'm not sure",
            "I cannot",
            "As an AI",
            "I apologize",
            "Error:",
            "undefined",
            "null",
            "NaN",
        ]

    def measure(self, prompt: str, response: str) -> float:
        """
        Measure coherence between prompt and response.

        Returns:
            float: Coherence score (0.0 = incoherent, 1.0 = fully coherent)
        """
        if not response or not response.strip():
            return 0.0

        coherence = 1.0
        response_lower = response.lower()

        # Check for incoherence patterns
        for pattern in self.incoherence_patterns:
            if pattern.lower() in response_lower:
                coherence -= 0.15

        # Length ratio check (response shouldn't be too short for complex prompts)
        prompt_words = len(prompt.split())
        response_words = len(response.split())

        if prompt_words > 20 and response_words < 10:
            coherence -= 0.2

        # Repetition check
        sentences = response.split(".")
        if len(sentences) > 2:
            unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
            repetition_ratio = len(unique_sentences) / len(sentences)
            if repetition_ratio < 0.5:
                coherence -= 0.3

        # Keyword overlap (simple relevance check)
        prompt_keywords = set(w.lower() for w in prompt.split() if len(w) > 4)
        response_keywords = set(w.lower() for w in response.split() if len(w) > 4)

        if prompt_keywords:
            overlap = len(prompt_keywords & response_keywords) / len(prompt_keywords)
            if overlap < 0.1:
                coherence -= 0.2

        return max(0.0, min(1.0, coherence))


# ============================================================================
# POR KERNEL
# ============================================================================

def por_kernel(
    drift: float,
    coherence: float,
    tol: float = 0.1,
    thresh: float = 0.7,
) -> ControlToken:
    """
    Proof-of-Resonance Kernel Decision Function.

    Determines whether the AI system should respond or remain silent
    based on drift and coherence metrics.

    Args:
        drift: Measure of contextual drift (0.0 = no drift, 1.0 = complete drift)
        coherence: Measure of response coherence (0.0 = incoherent, 1.0 = coherent)
        tol: Drift tolerance threshold (default: 0.1)
        thresh: Coherence threshold (default: 0.7)

    Returns:
        ControlToken.SILENCE: If system should remain silent
        ControlToken.PROCEED: If system is safe to respond
    """
    if drift > tol or coherence < thresh:
        return ControlToken.SILENCE
    return ControlToken.PROCEED


# ============================================================================
# MIDDLEWARE CLIENT
# ============================================================================

class PorCopilotClient:
    """
    Copilot SDK client with PoR (Proof-of-Resonance) interception layer.

    Wraps Copilot SDK calls with drift/coherence gates that can
    trigger intentional silence when stability conditions are violated.

    Example:
        client = PorCopilotClient()

        # Simple completion
        response = client.complete("Write a Python function to sort a list")

        if response:  # ProceedResponse is truthy, SilenceResponse is falsy
            print(response.text)
        else:
            print(f"Abstained: {response.reason}")

        # With conversation history
        client.add_context("User asked about sorting algorithms")
        response = client.complete("Now implement quicksort")
    """

    def __init__(
        self,
        drift_tolerance: float = 0.1,
        coherence_threshold: float = 0.7,
        copilot_client: Optional[Any] = None,
        on_silence: Optional[Callable[[SilenceResponse], None]] = None,
    ):
        """
        Initialize PoR-enabled Copilot client.

        Args:
            drift_tolerance: Maximum acceptable drift (default: 0.1)
            coherence_threshold: Minimum required coherence (default: 0.7)
            copilot_client: Optional existing Copilot SDK client
            on_silence: Optional callback when silence is triggered
        """
        self.drift_tolerance = drift_tolerance
        self.coherence_threshold = coherence_threshold
        self.copilot_client = copilot_client
        self.on_silence = on_silence

        # Initialize estimators
        self.drift_estimator = DriftEstimator()
        self.coherence_estimator = CoherenceEstimator()

        # Metrics tracking
        self.stats = {
            "total_requests": 0,
            "silence_count": 0,
            "proceed_count": 0,
            "avg_drift": 0.0,
            "avg_coherence": 0.0,
        }

    def add_context(self, text: str) -> None:
        """Add text to the conversation context for drift tracking."""
        self.drift_estimator.update(text)

    def _call_copilot(self, prompt: str, **kwargs) -> str:
        """
        Call the underlying Copilot SDK.

        Override this method to integrate with actual Copilot SDK.
        """
        if self.copilot_client is not None:
            # Real Copilot SDK integration
            # return self.copilot_client.complete(prompt, **kwargs)
            pass

        # Placeholder for demo - returns simulated response
        return f"[Simulated Copilot response for: {prompt[:50]}...]"

    def complete(
        self,
        prompt: str,
        pre_check: bool = True,
        post_check: bool = True,
        **kwargs,
    ) -> Union[ProceedResponse, SilenceResponse]:
        """
        Complete a prompt with PoR safety gates.

        Args:
            prompt: The prompt to send to Copilot
            pre_check: Check drift before calling Copilot
            post_check: Check coherence after receiving response
            **kwargs: Additional arguments for Copilot SDK

        Returns:
            ProceedResponse: If all checks pass
            SilenceResponse: If any check fails (abstention)
        """
        self.stats["total_requests"] += 1

        # ===== PRE-CHECK: Drift Gate =====
        if pre_check:
            drift = self.drift_estimator.measure(prompt)

            if drift > self.drift_tolerance:
                silence = SilenceResponse(
                    reason=f"Pre-check failed: drift ({drift:.3f}) > tolerance ({self.drift_tolerance})",
                    drift=drift,
                    coherence=0.0,
                )
                self._handle_silence(silence)
                return silence
        else:
            drift = 0.0

        # ===== CALL COPILOT =====
        try:
            raw_response = self._call_copilot(prompt, **kwargs)
        except Exception as e:
            silence = SilenceResponse(
                reason=f"Copilot call failed: {str(e)}",
                drift=drift,
                coherence=0.0,
            )
            self._handle_silence(silence)
            return silence

        # ===== POST-CHECK: Coherence Gate =====
        if post_check:
            coherence = self.coherence_estimator.measure(prompt, raw_response)

            # Final PoR Kernel decision
            decision = por_kernel(
                drift=drift,
                coherence=coherence,
                tol=self.drift_tolerance,
                thresh=self.coherence_threshold,
            )

            if decision == ControlToken.SILENCE:
                silence = SilenceResponse(
                    reason=(
                        "Post-check failed: "
                        f"coherence ({coherence:.3f}) < threshold ({self.coherence_threshold})"
                    ),
                    drift=drift,
                    coherence=coherence,
                )
                self._handle_silence(silence)
                return silence
        else:
            coherence = 1.0

        # ===== SUCCESS: Update context and return =====
        self.drift_estimator.update(prompt)
        self.drift_estimator.update(raw_response)

        self.stats["proceed_count"] += 1
        self._update_avg_metrics(drift, coherence)

        return ProceedResponse(
            text=raw_response,
            drift=drift,
            coherence=coherence,
        )

    def _handle_silence(self, silence: SilenceResponse) -> None:
        """Handle silence events (logging, callbacks, metrics)."""
        self.stats["silence_count"] += 1
        self._update_avg_metrics(silence.drift, silence.coherence)

        if self.on_silence:
            self.on_silence(silence)

    def _update_avg_metrics(self, drift: float, coherence: float) -> None:
        """Update rolling average metrics."""
        n = self.stats["total_requests"]
        self.stats["avg_drift"] = (self.stats["avg_drift"] * (n - 1) + drift) / n
        self.stats["avg_coherence"] = (self.stats["avg_coherence"] * (n - 1) + coherence) / n

    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics."""
        total = self.stats["total_requests"]
        return {
            **self.stats,
            "silence_rate": self.stats["silence_count"] / total if total > 0 else 0.0,
        }

    def reset_context(self) -> None:
        """Reset conversation context (start fresh)."""
        self.drift_estimator = DriftEstimator()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_por_client(
    drift_tolerance: float = 0.1,
    coherence_threshold: float = 0.7,
    **kwargs,
) -> PorCopilotClient:
    """
    Factory function to create a PoR-enabled Copilot client.

    Args:
        drift_tolerance: Maximum acceptable drift
        coherence_threshold: Minimum required coherence
        **kwargs: Additional client configuration

    Returns:
        Configured PorCopilotClient instance
    """
    return PorCopilotClient(
        drift_tolerance=drift_tolerance,
        coherence_threshold=coherence_threshold,
        **kwargs,
    )


def with_por_gate(
    func: Callable,
    drift_tolerance: float = 0.1,
    coherence_threshold: float = 0.7,
) -> Callable:
    """
    Decorator to add PoR gates to any function that returns text.

    Example:
        @with_por_gate(drift_tolerance=0.15)
        def my_llm_call(prompt):
            return llm.complete(prompt)
    """
    client = PorCopilotClient(
        drift_tolerance=drift_tolerance,
        coherence_threshold=coherence_threshold,
    )

    def wrapper(prompt: str, *args, **kwargs) -> Union[ProceedResponse, SilenceResponse]:
        # Measure drift
        drift = client.drift_estimator.measure(prompt)

        if drift > drift_tolerance:
            return SilenceResponse(
                reason="Drift exceeded tolerance",
                drift=drift,
                coherence=0.0,
            )

        # Call original function
        result = func(prompt, *args, **kwargs)

        # Measure coherence
        coherence = client.coherence_estimator.measure(prompt, result)

        decision = por_kernel(drift, coherence, drift_tolerance, coherence_threshold)

        if decision == ControlToken.SILENCE:
            return SilenceResponse(
                reason="Coherence below threshold",
                drift=drift,
                coherence=coherence,
            )

        client.drift_estimator.update(prompt)
        client.drift_estimator.update(result)

        return ProceedResponse(text=result, drift=drift, coherence=coherence)

    return wrapper


# ============================================================================
# DEMO / TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PoR Middleware Demo")
    print("=" * 60)

    # Create client with default thresholds
    client = create_por_client(
        drift_tolerance=0.3,  # More lenient for demo
        coherence_threshold=0.5,
    )

    # Simulate conversation
    prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Now add memoization to optimize it",
        "What's the weather like today?",  # Should trigger drift!
        "Convert the function to use iterative approach",
    ]

    print("\nSimulating conversation with PoR gates:\n")

    for i, prompt in enumerate(prompts, start=1):
        print(f"[{i}] Prompt: {prompt}")
        response = client.complete(prompt)
        if response:
            print(f"    Proceed: {response.text}")
        else:
            print(f"    Silence: {response.reason}")
        print(f"    drift={response.drift:.3f} coherence={response.coherence:.3f}")
        print("-" * 60)

    print("\nStats:")
    print(client.get_stats())
