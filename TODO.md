# TODO

## Test Jev (TypeSafe AI) as a classification provider

Added 2026-10-14. Not started.

Jev is a "System One" decision model, not a text-generating LLM: a block of
text (the "state") plus typed questions, all answered in one pass, each with a
calibrated probability and no rationale. Early access since 2026-09-15
(waitlist). Vendor claims: 70–500 ms per call, $0.042 per 1M input tokens,
output free. The official API docs were not found; the details below come from
third-party guides, so confirm the request and response schema (TypeSafe's
OpenAPI spec) before building.

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`, Bearer key,
  `"model": "jev-latest"` (pin a version, e.g. `jev-1.13.0`, for research runs).
  Also listed on DigitalOcean (Serverless Inference), ZenMux, LLM Gateway.
- Request: `state` (text) + `questions` (named map). Question types: `choice`
  (one of up to ~255 options, probability per option), `score` (ordered scale),
  and a yes/no type the guides call "Noul" (probability the statement is true;
  exact `type` string unconfirmed). Text only; context 32k or 64k (sources
  disagree).

Plan:

1. New provider `typesafe` in `_providers.py` with its own request builder
   (not the chat-message path). Add it to `UNSUPPORTED_BATCH_PROVIDERS`;
   ignore creativity and thinking_budget.
2. Multi-label: one yes/no question per category (name + description), the
   response as `state`, all categories in one call; threshold each probability
   at 0.5 and emit the usual `{"1": "0", "2": "1", ...}` so aggregation and
   output are unchanged. Optionally keep the probabilities as extra columns.
3. Single-label: one `choice` question with the categories as options (ties in
   with the compact single-label format, branch `feat/single-label-compact`).
4. Open design question: where the task instructions go. Jev reads questions
   literally and is reported to lose accuracy when the state carries material a
   question does not need, so compare instructions in `state` against
   instructions repeated in each question.
5. Pilot on one question of the closed_vs_open_llm wave 2 data (frozen inputs,
   coders' categories and instructions), scored against the coders like the
   other models; consider it as an appendix arm, not part of the 18-model
   lineup.

Reading: Zhang et al., "Same Scores, Different Decisions: Evaluating JEV and
Language Models for Legal Document Understanding", arXiv:2609.27678 (Jev
cheapest and fastest; every hosted LLM compared was more accurate on
ContractNLI). Related but different: AutoTrust's open-weight JEV-27B-VL (vision)
and the open "Jev-like" Kev models on Qwen (github.com/jaredpalmer/kev), which
could be run locally.
