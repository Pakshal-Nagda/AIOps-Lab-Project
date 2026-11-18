# Project Medhā - Sanskrit & Prakrit Live Recitation Verifier

**Project Medhā** is an edge-deployable, real-time speech verification platform designed for traditional Sanskrit and Prakrit oral recitation. Combining a quantized Conformer-CTC acoustic model, space-agnostic phonetic character alignment, and streaming Voice Activity Detection (VAD), Medhā provides live syllable-level pronunciation feedback, cadence tracking, and blind memorization testing.

---

## Documentation

For architectural design, operational specifications, and verification details, refer to the documentation in `docs/`:

* **[High-Level Design (HLD)](docs/high_level_design.md):** System architecture, design decisions (Ray, NeMo, MLflow, DVC), and end-to-end data flow.
* **[Low-Level Design (LLD)](docs/low_level_design.md):** Component classes, REST & WebSocket API schemas (`/ws/recite`, `/api/v1/analyze`), and Prometheus metrics.
* **[Architecture Schematic](docs/architecture.svg):** Standalone visual architecture and MLOps pipeline diagram.
* **[Test Plan](docs/test_plan.md):** 27-point master test matrix across unit, integration, and performance tiers.
* **[Test Report](docs/test_report.md):** Empirical model validation benchmarks (CER, WER, RTF) and automated test results.
* **[User Manual](docs/user_manual.md):** Step-by-step practitioner guide for live continuous recitation, audio file verification, and voice dataset contribution.

Find the video demonstration of the entire project [here](https://drive.google.com/file/d/1ziLDyUMikVS0uwilkjQjcG-ARF5n5gbZ/view?usp=sharing).

---

## LLM Usage Disclosure

I used **ChatGPT** to solve minor doubts and **Gemini 3.8 Flash (Medium)** to significantly aid in coding and documentation. I take full responsibility of the code and all files that I turn in.