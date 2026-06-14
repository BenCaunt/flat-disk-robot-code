# 2026-06-13 Open-Vocab Nav Research Loop

Seed experiment for Qwen planner prompt variants using the real robot-compatible
tool surface in THOR.

Dry-run:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_prompt_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Topomap-memory sweep dry-run:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Runpod/Linux sweep dry-run:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Execute locally with a Qwen OpenAI-compatible endpoint already running:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_prompt_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --preflight-endpoints \
  --parallelism 1
```

The configs intentionally use only model-based Qwen variants. Research configs
default to `strict_model_based=true`, so hard-coded scripted runners and
THOR-derived semantic topomap terms are rejected unless a debug config explicitly
opts out. Scripted runners are reserved for smoke tests and should not be
counted as general open-vocabulary results.

`qwen_topomap_memory_sweep.json` is local/Apple oriented and uses the MLX Qwen
and Florence paths. `qwen_topomap_memory_sweep_runpod_linux.json` is the
minimal Runpod/Linux sibling, and `qwen_strategy_sweep_runpod_linux.json` is the
current generated-strategy sweep. The Linux configs expect
`Qwen/Qwen3-VL-8B-Instruct` behind an OpenAI-compatible endpoint at
`127.0.0.1:8000` and use `florence-transformers` for `visual_servo_object`.
A Runpod worker can either connect to a pre-existing endpoint or set
`START_QWEN_SERVER=1` / `--start-qwen-server` to start a local vLLM endpoint
before preflight.

Variants can opt into non-motion topomap memory by adding
`topomap_memory_map_dir` and `topomap_memory_use_clip`. Keep
`topomap_memory_allow_semantic_terms` off for strict policy runs unless the saved
semantic terms came from a real-robot-compatible labeling process.

`topomap_memory_map_dir` supports `{episode}` templating. The topomap sweep
expects maps such as `sim/scratch/semantic_topomaps/living_room_sofa_clip`.
Run with `--preflight-endpoints` before launching THOR; this checks both the
Qwen endpoint and configured topomap maps/CLIP embeddings. The
`qwen-strategy-runpod-linux-v4` Warmhub queue includes conditional build guards
for missing CLIP topomap artifacts in fixture, preflight, and topomap-memory
slices, without rebuilding existing maps.

Use `--preflight-only` when a worker should validate endpoint plus topomap
fixtures and stop before launching any THOR episodes.

Runpod launch command dry-run:

```bash
uv run --project sim flatdisk-sim-runpod-launch-task \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --start-qwen-server \
  --terminate-after 4h
```

Add `--launch` only after the generated `runpodctl pod create` command is
reviewed, the current git ref is pushed, and Runpod credentials plus `wh` CLI
availability are confirmed.

Local Codex shells do not automatically load the root `.env`. This repo's
`.env` may contain `RUNPOD_API_KEY`; before any `runpodctl` command, load only
that variable and do not print it:

```bash
export RUNPOD_API_KEY="$(grep -m1 '^RUNPOD_API_KEY=' .env | cut -d= -f2- | sed 's/^"//;s/"$//')"
runpodctl user
```

If `runpodctl user` reports account JSON, Runpod auth is available for launch.
If it says `api key not configured`, the variable was not loaded into that
shell.

Runpod dispatch dry-run for multiple planned trial slices:

```bash
uv run --project sim flatdisk-sim-runpod-dispatch \
  --name-prefix qwen-strategy-runpod-linux-v4-run- \
  --tag trial-slice \
  --max-workers 2 \
  --start-qwen-server \
  --dispatch-manifest sim/scratch/open_vocab_nav_research_loop/runpod_dispatch_manifest.json \
  --terminate-after 4h
```

Use the manifest as a redacted dispatch review artifact and attach it to
Warmhub task results so later agents can see which planned tasks were selected
or skipped before any pods were launched.

Generated strategy sweep:

```bash
uv run --project sim flatdisk-sim-generate-nav-strategy-sweep \
  --base-config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep_runpod_linux.json \
  --output experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --experiment-id open_vocab_nav_qwen_strategy_runpod_linux_v1
```

The generated sweep keeps all variants model-based (`qwen`), avoids static
object/color examples in actor rules, and keeps semantic topomap term routing
off. It relies on the default strict research-loop gate. Use
`flatdisk-sim-research-warmhub task-plan-config` to seed planned Runpod
trial-slice tasks from it. The generated training-review task materializes
`qwen_dpo_messages.jsonl` from `training_export/policy_dataset_v1` before
running training readiness, so future workers can see whether preference-tuning
handoff data exists.

The `qwen_grounding_recovery` variant is derived from policy-review trace
failure analysis. It specifically tests whether Qwen can bind its own
grounding-audit result to the next action: after mismatched or unstable visual
servo grounding, it should avoid repeating the same servo prompt and instead
change viewpoint, query image memory, or use a distinct visible waypoint.

The `qwen_grounding_audit_critic` variant is the next trace-derived pressure
test. It enables `action_history_summary` and uses `critic_mode=same-model` so
a second Qwen pass can reject a repeated `visual_servo_object` prompt when the
actor's own prior grounding audit said the prompt should change. This keeps the
policy model-based and scene-general while directly testing the dominant
bathroom cross-run failure pattern.

To queue only that pressure test without duplicating the full sweep, use the
task planner's variant and episode filters:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-plan-config \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --plan-id qwen-grounding-audit-critic-v1 \
  --owner unassigned \
  --tag qwen --tag runpod --tag targeted --tag grounding-audit-critic \
  --variant qwen_grounding_audit_critic \
  --episode living_room_sofa --episode bedroom_bed --episode bathroom_toilet \
  --include-slice-tasks
```

The `qwen_grounding_dino_recovery` variant is derived from detector-doctor
evidence on the bathroom failure case: Florence returned no boxes on a saved
frame where the target fixture was plainly visible, while GroundingDINO selected
the visible fixture. This variant keeps the Qwen/tool loop general and changes
only the open-vocabulary grounding backend.

Executed sweeps write `training_export/policy_steps.jsonl`,
`training_export/policy_review_traces.jsonl`,
`training_export/episode_rollouts.jsonl`, `training_export/rollout_groups.jsonl`,
`training_export/trajectory_preferences.jsonl`, and
`training_export/policy_dataset_v1/`. These are intended as the bridge to
offline SFT/GRPO/PPO work: policy sample records contain only model-facing
inputs and outputs, while hidden THOR distance/success is stored in separate
evaluator reward/label fields. Use `policy_review_traces.jsonl` for fast
failure triage and sub-agent handoff: it records Qwen actor output, executed
tool calls, critic decisions, grounding-audit fields, and contact-sheet paths
without hidden target distance or THOR object metadata.

Before training, run the Qwen tool-training materializer, which joins
`policy_dataset_v1/policy_samples.jsonl` with
`policy_dataset_v1/evaluator_labels.jsonl`, validates referenced image paths,
filters unsafe or rejected actions, emits Qwen-compatible multimodal SFT JSONL,
emits `qwen_action_preferences.jsonl` for guard-replaced actor actions, and
emits `qwen_dpo_messages.jsonl` with explicit `prompt`, `chosen`, `rejected`,
and `images` columns for VLM preference tuning. It also audits that no
privileged evaluator fields leak into model-facing messages.
Preference records use only the model-facing Qwen prompt as input, choose the
executed safe action, and reject the original actor action without contaminating
accepted SFT.
Treat PPO/GRPO as a later step once grouped rollouts, reward normalization, and
on-policy collection are explicit.

```bash
uv run --project sim flatdisk-sim-prepare-qwen-tool-training \
  --input sim/scratch/open_vocab_nav_research_loop/<run>/training_export/policy_dataset_v1 \
  --output-dir sim/scratch/open_vocab_nav_research_loop/<run>/qwen_tool_training
```

Then run training readiness over the run directory. It will discover both
`training_export/training_manifest.json` and
`qwen_tool_training/qwen_tool_training_manifest.json`, reporting raw policy
samples, accepted Qwen SFT records, Qwen guard-replacement preference pairs, and
standard DPO handoff records. Any missing Qwen image or privileged-token blocker
is reported separately.

```bash
uv run --project sim flatdisk-sim-nav-training-readiness \
  --input sim/scratch/open_vocab_nav_research_loop/<run> \
  --output-dir sim/scratch/open_vocab_nav_research_loop/<run>/training_readiness
```

After readiness, generate the DPO training handoff. This validates
`qwen_dpo_messages.jsonl`, confirms the referenced Qwen images still exist,
and writes a job manifest plus a generated TRL script without starting GPU
fine-tuning in the local sim environment.

```bash
uv run --project sim flatdisk-sim-plan-qwen-dpo-training \
  --input sim/scratch/open_vocab_nav_research_loop/<run>/qwen_tool_training \
  --output-dir sim/scratch/open_vocab_nav_research_loop/<run>/qwen_dpo_training
```

On a training-capable worker with `accelerate`, `datasets`, `peft`, `pillow`,
`torch`, `transformers`, and `trl` installed, run the generated job:

```bash
uv run --project sim flatdisk-sim-run-qwen-dpo-training \
  --job sim/scratch/open_vocab_nav_research_loop/<run>/qwen_dpo_training/qwen_dpo_training_job.json
```

Use `--dry-run --skip-dependency-check` for queue or path validation without
starting the generated `accelerate launch ...` command.

For PPO/GRPO work, materialize the grouped rollout handoff. This reads
`rollout_groups.jsonl`, can merge repeated `--input` exports for the same
episode prompt, keeps evaluator rewards out of Qwen prompt/completion messages,
and marks only actor-equals-executed trajectories as trainable.

```bash
uv run --project sim flatdisk-sim-prepare-qwen-grpo-training \
  --input sim/scratch/open_vocab_nav_research_loop/<run> \
  --output-dir sim/scratch/open_vocab_nav_research_loop/<run>/qwen_grpo_training
```

For split Runpod artifacts, repeat `--input` once per artifact directory and
write a merged output directory such as `qwen_grpo_training/bathroom_cross_run`.

Then plan the offline replay GRPO training job. This validates the ready
`qwen_grpo_training_manifest.json`, writes `qwen_grpo_trl_dataset.jsonl`, and
generates a TRL `GRPOTrainer` script without starting GPU training locally.

```bash
uv run --project sim flatdisk-sim-plan-qwen-grpo-training \
  --input sim/scratch/open_vocab_nav_research_loop/<run>/qwen_grpo_training \
  --output-dir sim/scratch/open_vocab_nav_research_loop/<run>/qwen_grpo_jobs
```

On a training-capable worker, run the generated job:

```bash
uv run --project sim flatdisk-sim-run-qwen-grpo-training \
  --job sim/scratch/open_vocab_nav_research_loop/<run>/qwen_grpo_jobs/qwen_grpo_training_job.json
```

Use `--dry-run --skip-dependency-check` to validate the job handoff without
starting `accelerate launch`. This GRPO path uses recorded rollout rewards as
an offline proxy reward; online simulator reward training is still a separate
future worker design.
