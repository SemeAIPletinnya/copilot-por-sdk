"""
Minimal PoR interception demo for Copilot SDK
Demonstrates abstention as a control primitive
"""

from dataclasses import dataclass

# --- PoR kernel (minimal stub for demo) ---

@dataclass
class SilenceToken:
    reason: str

def por_decide(drift: float, coherence: float,
               drift_threshold=0.6,
               coherence_threshold=0.4):
    if drift > drift_threshold or coherence < coherence_threshold:
        return SilenceToken(
            reason=f"drift={drift:.2f}, coherence={coherence:.2f}"
        )
    return "PROCEED"


# --- Copilot interception layer (mock) ---

def copilot_complete(prompt: str):
    # stand-in for real Copilot SDK call
    return f"[COPILOT OUTPUT] {prompt}"


def copilot_with_por(prompt: str, drift: float, coherence: float):
    decision = por_decide(drift, coherence)

    if isinstance(decision, SilenceToken):
        return {
            "status": "ABSTAIN",
            "reason": decision.reason
        }

    return {
        "status": "OK",
        "output": copilot_complete(prompt)
    }


# --- Demo runs ---

if __name__ == "__main__":
    tests = [
        ("write SQL migration", 0.2, 0.9),
        ("optimize crypto trading bot", 0.8, 0.3),
    ]

    for prompt, drift, coherence in tests:
        print("\nPROMPT:", prompt)
        result = copilot_with_por(prompt, drift, coherence)
        print("RESULT:", result)
