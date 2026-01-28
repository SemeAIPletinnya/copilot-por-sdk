# PoR Interception Layer

## What this is

This repository introduces an **interception layer** for the GitHub Copilot SDK that allows *abstention* to be represented as an explicit, first‑class outcome rather than being collapsed into errors or timeouts.

The goal is deliberately narrow:

> **distinguish intentional silence from failure** in agentic systems.

This change is currently scoped to **test harnesses and SDK‑level semantics**, not runtime behavior.

---

## The problem

In agentic or long‑running assistant loops, silence can mean different things:

* the model failed
* the backend is unavailable
* the agent is stuck in a loop
* the agent intentionally chose not to respond

Today, most SDKs collapse all of these into the same surface signal:

* timeouts
* exceptions
* retries

This makes it impossible for tests, orchestration layers, or downstream agents to reason about *why* nothing happened.

---

## Why interception (not generation)

PoR is not introduced at the generation layer.

It lives **between steps** — at the point where an agent decides whether to:

* commit to the next action
* wait
* stop
* or abstain

Placing the logic here avoids:

* changing model behavior
* modifying backend protocols
* introducing opinionated policies

Instead, it exposes **a missing control signal**.

---

## What changes in this fork

This fork introduces:

* an explicit *abstention outcome* in SDK test harnesses
* a way to surface silence as a non‑error signal
* separation between:

  * failure
  * timeout
  * intentional no‑response

No runtime behavior is modified.

No model decisions are overridden.

---

## Why tests first

Tests are where semantics become visible.

If abstention cannot be represented in tests, it cannot be reasoned about safely at runtime.

By formalizing abstention as a test outcome:

* CI can distinguish silence from failure
* SDK consumers can experiment safely
* future runtime changes have a clear contract

This mirrors how cancellation, retries, and backoff were historically introduced in other systems.

---

## Relation to Proof‑of‑Resonance (PoR)

PoR metrics (coherence, drift, envelopes) naturally belong at this interception point.

In future iterations, they may inform *when* abstention occurs.

In this repository, PoR is intentionally kept minimal:

* **no scoring**
* **no thresholds**
* **no policy enforcement**

Only the *semantic placeholder* is introduced.

---

## Current status

* ✅ Explicit abstention outcome in test harnesses
* ✅ No runtime changes
* 🚧 Examples and documentation evolving
* 🚫 Not a production policy

---

## Why this matters

Without an abstention signal:

> silence is indistinguishable from failure

With it:

> silence becomes a controllable, inspectable state

This is a small change with systemic implications for agent reliability.

---

## Scope disclaimer

This repository is **not**:

* a full PoR implementation
* a behavioral modifier for Copilot
* a proposal to change model outputs

It is a controlled experiment in **making absence observable**.
