# Scenario Runs re-evaluate Demand Class for affected series

When a Scenario Run applies an override that materially changes a series' demand pattern (e.g., injecting a promo event or a stockout), Prism re-runs `classify_demand_profiles` for those affected series before re-running feature config and model selection. The Demand Class is not frozen at the baseline value.

The alternative — freezing Demand Class — is simpler but wrong: if a stockout turns a SMOOTH series intermittent, running SMOOTH models against it produces invalid forecasts. The re-classification cost is small (Syntetos-Boylan is deterministic and cheap), and the model gate correctness is worth it.

**Consequence for plan_v1.md §9:** `run_forge_for_scenario` must re-run `classify_demand_profiles` for affected series in addition to `specify_feature_config`. The current spec description ("re-runs feature_config only") is incorrect and must be updated.