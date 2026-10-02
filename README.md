# Temporal Agent Memory System

A production-oriented temporal memory system for an LLM agent that stores, retrieves, updates, and reasons over user memories while respecting **time, provenance, confidence, and lifecycle state**.

The system is designed around a core temporal-memory problem:

```text
Node.js  [Jan, Jun)
Go      [Jun, Sep)
Python  [Sep, ∞)