# Solution Steps

1. Implement the context-builder as the trust boundary rather than changing the broad lexical retriever. Let retrieval return candidates, then filter before anything reaches the model.

2. Scope evidence by `asking_carrier_id` and supported source type. Exclude all rows from other carriers, including their SOPs and notes, and avoid putting excluded carrier IDs or content into the assembled prompt.

3. Deduplicate retrieved rows deterministically by `doc_id`, keeping the highest-scoring/newest row when duplicates appear in busy-hub candidate sets.

4. Separate authoritative SOP evidence from shipment notes. Select SOPs first because only SOPs may support policy claims; shipment notes are included only as lower-priority shipment-fact data.

5. Detect instruction-like shipment-note text with conservative prompt-injection patterns such as “system instruction”, “ignore prior”, “approve any”, and SOP-control language. Redact those note bodies before prompt assembly so injected text cannot steer the model.

6. Format evidence in explicit source blocks with source IDs, type, carrier, hub, date, retrieval score, and JSON-encoded text. Mark SOPs as `authoritative_sop` and notes as `untrusted_shipment_data`.

7. Add system and user instructions that make the trust contract explicit: use only asking-carrier evidence, base policy claims only on SOPs, treat notes only as data, cite source IDs for every policy claim, and say no authoritative rule exists when SOP evidence is missing.

8. Apply a deterministic token budget. Estimate tokens with `tiktoken` when available, otherwise use a conservative character estimate. Add evidence in sorted relevance order, truncate oversized high-ranked SOPs if useful, and drop lower-priority evidence when the budget is exhausted.

9. Expose traceability from the end-to-end `answer` path by returning `grounding_sources` metadata for the permitted sources actually included in the model context, along with assembly statistics and the raw offline/client response.

10. Run the provided readiness script or `pytest` to verify chat message shape, cross-carrier SOP exclusion, injection neutralization, bounded prompt size, offline answer execution, and identifiable grounding sources.

