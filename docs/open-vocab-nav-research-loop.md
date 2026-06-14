# Open-Vocabulary Navigation Research Loop

This is the research-loop contract for general indoor navigation experiments.
The policy input path must stay true to the real robot: RGB camera frames,
previous motion strips, IMU yaw, bounded tool results, and model memory. Hidden
THOR metadata, target distance, scene objects, and target labels are allowed only
for scoring, debugging, and artifact review outside the policy directory.

## Current Loop

`flatdisk-sim-research-loop` reads a JSON experiment config, expands a trial
matrix across prompt/model/tool variants and THOR episodes, and can execute the
matrix in parallel through the existing `run_thor_harness_episode` path.

Dry-run mode writes:

- `research_loop_manifest.json`: exact trial matrix and config.
- `research_loop_summary.json`: aggregate status.
- `research_loop_report.md`: human-readable report.
- `warmhub_shapes.json`: proposed Warmhub shapes.
- `warmhub_ops.json`: Warmhub commit operations for experiments, variants,
  episode specs, runs, artifacts, assessments, and failure observations.
- `AGENTS.warmhub.md`: startup instructions for future agents.

Executed runs also write `training_export/` with policy-step JSONL,
policy-review traces, episode-rollout JSONL, rollout groups, trajectory
preference pairs, and `policy_dataset_v1/`. The policy channel contains actor
prompts, image/contact sheet paths, sanitized observations, actions, and tool
feedback. `policy_review_traces.jsonl` is the compact sub-agent review surface
for Qwen actor outputs, critic decisions, executed tools, grounding audits, and
general flags such as unstable visual-servo grounding. Hidden THOR
distance/success is kept in evaluator-only reward/label channels for offline
ranking, SFT filtering, PPO, or GRPO; it is never inserted into model inputs or
policy-review traces. `flatdisk-sim-prepare-qwen-tool-training` can also
materialize guard-replaced actor actions as Qwen action-preference pairs, where
the executed safe action is the chosen target and the rejected actor action is
kept out of accepted SFT.

Seed config:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_prompt_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Topomap-memory comparison config:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Runpod/Linux generated strategy config:

```bash
uv run --project sim flatdisk-sim-research-loop \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --output-dir sim/scratch/open_vocab_nav_research_loop \
  --dry-run
```

Generated strategy sweeps can be created from an existing config:

```bash
uv run --project sim flatdisk-sim-generate-nav-strategy-sweep \
  --base-config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep_runpod_linux.json \
  --output experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --experiment-id open_vocab_nav_qwen_strategy_runpod_linux_v1
```

This emits generic Qwen strategies for baseline behavior, viewpoint coverage,
evidence exploitation, failure recovery, and optional CLIP-backed topomap image
memory. Generated strategies are covered by tests that audit static prompt
context for object/color terms.

Remove `--dry-run` to launch THOR. Add `--parallelism N` to override config
concurrency. Add `--preflight-endpoints` to check OpenAI-compatible Qwen
endpoints before launching THOR and record unavailable endpoints as structured
failed trials. Add `--preflight-only` to perform endpoint/topomap checks and
stop before launching THOR.

## Current Tool Surface

Qwen can choose bounded motion, Florence/GroundingDINO phrase-grounded visual
servo, `wait`, `stop`, and `query_topomap_memory`. The topomap memory tool is
non-motion and opt-in: variants may set `topomap_memory_map_dir` and
`topomap_memory_use_clip` to let Qwen query a saved semantic topomap with the
latest RGB frame and text goal. The model-facing result contains image-match
scores, route node ids, and a contact sheet; it omits poses, scene metadata,
object metadata, target distance, and hidden THOR state.

Qwen variants use `critic_mode: none` by default in the active configs. The
actor therefore owns bounded tool selection and can call `stop` to assert
completion without a local rule-based critic replacing the action. The old
`SafetyCriticRunner` remains available only for explicit `critic_mode: safety`
baseline/debug runs, and `critic_mode: same-model` can be used for a separate
model-based critic ablation.

Strict policy runs should prefer CLIP-backed topomaps. The
`topomap_memory_allow_semantic_terms` switch is reserved for debugging or maps
whose semantic terms are known to come from a real-robot-compatible process.
`topomap_memory_map_dir` may use `{episode}` so one sweep can compare per-scene
maps such as `sim/scratch/semantic_topomaps/living_room_sofa_clip`.

## Warmhub

Warmhub is the cross-agent memory layer. Its docs describe shapes, things,
assertions, and wrefs as the core data model, with CLI writes through `wh commit
submit`. New agents should run `wh prime` at startup, then query recent
`AgentTask`, `NavEvalRun`, `PromptVariant`, and `FailureObservation` records
before choosing work.

Default repo target:

```bash
export WARMHUB_REPO=bencaunt-2/open-vocab-nav-research-loop
wh prime
wh repo describe --json
uv run --project sim flatdisk-sim-research-warmhub status --limit 20
uv run --project sim flatdisk-sim-research-warmhub task-list --status planned --ready-only --limit 20
wh thing query --shape AgentTask --where status=running --limit 20 --json
wh thing query --shape AgentTask --where status=planned --limit 20 --json
wh thing query --shape NavEvalRun --limit 20 --json
wh thing query --shape PromptVariant --limit 20 --json
wh assertion list --shape FailureObservation --limit 20 --json
```

For a compact queue and eval triage view, use:

```bash
uv run --project sim flatdisk-sim-research-warmhub status --limit 20
uv run --project sim flatdisk-sim-research-warmhub status \
  --related-experiment open_vocab_nav_qwen_strategy_runpod_linux_v1 \
  --json
```

The status command summarizes task counts, recent `NavEvalRun` records,
failure observations, sub-agent results, and recommended next actions.
It flags running tasks whose `updatedAt` is older than four hours by default;
use `--stale-running-after-s 0` to disable that check, or lower the threshold
when auditing short pod jobs.
Planned tasks may declare `notes.prerequisites`; workers should use
`task-list --ready-only` or the Runpod dispatcher default so trial slices are
not claimed before fixture/preflight tasks are complete. `task-claim` also
rejects incomplete prerequisites unless `--force` is supplied.

The research loop does not write to Warmhub unless `--commit-warmhub` is passed.
Use `--init-warmhub-repo` to create the repo and missing shapes before submit.
Agents should claim planned queue work before running it, then finish the task
with evidence artifacts:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-list \
  --status planned \
  --ready-only \
  --limit 20

uv run --project sim flatdisk-sim-research-warmhub task-claim \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --owner agent-name \
  --note "Starting topomap fixture build."

uv run --project sim flatdisk-sim-research-warmhub note \
  --author agent-name \
  --about NavExperiment/open_vocab_nav_qwen_prompt_sweep_v1 \
  --note "Observed repeated final-goal servo without visible target evidence." \
  --tag qwen --tag failure

uv run --project sim flatdisk-sim-research-warmhub task-finish \
  --task inspect-failures-001 \
  --agent agent-name \
  --status complete \
  --summary "Failure mode logged; next action is an exploration-biased prompt."
```

For unattended local or pod-side workers, execute queued commands with the full
claim/run/finish lifecycle. Multi-command fixture tasks should use
`--all-commands` so every map build and validation command finishes before the
task is marked complete:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-run-command \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --complete-exit-code 0 \
  --complete-exit-code 2 \
  --log-file sim/scratch/agent_logs/qwen-strategy-v4-topomap-fixtures.log \
  --dry-run

uv run --project sim flatdisk-sim-research-warmhub task-run-command \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --complete-exit-code 0 \
  --complete-exit-code 2 \
  --log-file sim/scratch/agent_logs/qwen-strategy-v4-topomap-fixtures.log \
  --evidence-artifact sim/scratch/semantic_topomaps/living_room_sofa_clip
```

Use `task-start` only for ad hoc work that is not already represented by a
planned `AgentTask`.
Research-loop exit code `2` is accepted for queued worker tasks because it means
the eval produced unsuccessful navigation or structured failure data; worker
failure should be reserved for crashes, missing artifacts, or command errors.

When a failure trace shows `visual_servo_object` returned `no_detection` on
visually plausible frames, run the detector doctor on saved policy frames before
rerunning a full slice:

```bash
uv run --project sim --with torch --with torchvision --with transformers --with timm --with einops \
  flatdisk-sim-detector-doctor \
  --image path/to/frame.jpg \
  --prompt "visible object phrase" \
  --detector grounding-dino \
  --output-dir sim/scratch/detector_doctor/run-id
```

The doctor writes JSON, Markdown, and detector overlays so agents can compare
Florence/GroundingDINO-style backends without using hidden simulator metadata.

To create a Warmhub work queue from a research-loop config:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-plan-config \
  --config experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
  --plan-id qwen-strategy-runpod-linux-v4 \
  --owner unassigned \
  --tag qwen --tag runpod --tag generated-strategies \
  --include-slice-tasks
```

For Runpod/Linux workers, use the v4 generated-strategy queue above. The
topomap-fixture task notes include concrete `flatdisk-sim-build-semantic-topomap`
commands for each episode and `flatdisk-sim-research-loop --preflight-only`
validation commands. The v4 fixture, preflight, and topomap-memory slice
commands include conditional map-build guards, so fresh pods can create missing
CLIP map artifacts before running the eval without rebuilding existing maps.

## Runpod Worker

The pod-side worker wrapper is:

```bash
TASK_ID=qwen-strategy-runpod-linux-v4-run-qwen_topomap_memory-living_room_sofa \
CONFIG=experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json \
VARIANT=qwen_topomap_memory \
EPISODE=living_room_sofa \
scripts/runpod_open_vocab_nav_research_loop.sh
```

It writes `/workspace/open_vocab_nav_research_loop.log`,
`/workspace/open_vocab_nav_research_loop.exit`, and timestamped output
directories under `/workspace/outputs/open_vocab_nav_research_loop` by default.
If `TASK_ID` is set, it claims the Warmhub task at startup and writes a
`SubAgentResult` on exit.

The wrapper defaults to the Runpod/Linux config, `UV_EXTRAS=thor`, and
`UV_WITH=torch,transformers`, avoiding the local MLX harness dependency. Serve
Qwen through a Linux-compatible OpenAI-style endpoint such as vLLM on
`127.0.0.1:8000`, set `START_QWEN_SERVER=1` to have the wrapper start that
endpoint with `scripts/runpod_start_qwen_vllm.sh`, or override `qwen_endpoint`
in a copied config. For v4 queued commands, topomap fixtures under
`sim/scratch/semantic_topomaps/{episode}_clip` are built automatically when
missing before preflight or topomap-memory slices run. Set `PREFLIGHT_ONLY=1`
for a validation-only pod job. The Qwen startup path is opt-in because it can
install vLLM and download model weights.

To create a Runpod launch command from a planned Warmhub task, use the
dry-run-first launcher:

```bash
uv run --project sim flatdisk-sim-runpod-launch-task \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --start-qwen-server \
  --qwen-vllm-extra-args "--max-model-len 8192" \
  --terminate-after 4h \
  --evidence-artifact /workspace/outputs/open_vocab_nav_research_loop
```

The launcher emits a `runpodctl pod create` command using Runpod's documented
`--image`, `--gpu-id`, `--env`, and `--docker-args` flags. Add `--launch` only
after the generated command looks right, `RUNPOD_API_KEY` is configured for
pod management, the git ref is pushed, and the pod image can run the `wh` CLI
or `WH_INSTALL_CMD` installs it. By default, real launch is refused from a dirty
worktree because the pod clones `origin` and would otherwise miss local worker
changes.

With `--start-qwen-server`, the generated worker claims the Warmhub task, starts
`Qwen/Qwen3-VL-8B-Instruct` through vLLM, waits for `/v1/models`, and then runs
the planned command with `--no-claim`. The Qwen server log is attached as task
evidence.

To fan out several planned trial-slice tasks, use the dispatcher. It queries
Warmhub `AgentTask` records, filters locally, and emits one Runpod pod command
per selected task:

```bash
uv run --project sim flatdisk-sim-runpod-dispatch \
  --name-prefix qwen-strategy-runpod-linux-v4-run- \
  --tag trial-slice \
  --max-workers 2 \
  --start-qwen-server \
  --qwen-vllm-extra-args "--max-model-len 8192" \
  --dispatch-manifest sim/scratch/open_vocab_nav_research_loop/runpod_dispatch_manifest.json \
  --terminate-after 4h \
  --evidence-artifact /workspace/outputs/open_vocab_nav_research_loop
```

Add `--launch` only after reviewing the dry-run output. The dispatcher refuses
real launches from a dirty worktree unless `--allow-dirty` is supplied.
The dispatch manifest is a redacted, durable review artifact; attach it to
`SubAgentResult.evidenceArtifacts` or cite it from an `AgentNote` so future
agents can reconstruct which Warmhub tasks were selected, skipped, and launched
from which git ref.
It also includes bounded running and blocked `AgentTask` samples so reviewers
can see active queue pressure before adding more Runpod workers, and flags
running tasks whose `updatedAt` looks stale.
The dispatcher reads a larger Warmhub task page before local filtering so older
matching planned tasks are not missed in a busy queue.
By default it skips tasks whose `notes.prerequisites` are not complete; pass
`--ignore-prerequisites` only for deliberate recovery/debug work.

## Generality Rules

- No color-threshold object detector in the model-facing path.
- No hard-coded object-family policy should count as a research-loop baseline.
- THOR target labels and object metadata are evaluator-only.
- Florence/GroundingDINO-style phrase grounding is a tool result, not proof of
  semantic goal completion.
- Research configs default to `strict_model_based=true`, which rejects
  non-model runners and THOR-derived semantic topomap term routing before a run
  starts. Set it false only for explicit smoke/debug experiments.
- `noHardcodedLabelsOrColors` is derived, not asserted: strict results require a
  model-based runner (`qwen` or `codex`) and no static object/color examples in
  the model-facing prompt context.
- Topomap memory hints are navigation memory, not ground-truth completion.
- Scripted or fast hard-coded runners may be used as smoke tests, but not as
  evidence of a general model-based solution.

## Next Integrations

- Run `qwen-strategy-runpod-linux-v4-preflight` with `START_QWEN_SERVER=1` on
  Runpod, then fan out v4 trial slices after preflight completes.
- Use `flatdisk-sim-prepare-qwen-tool-training` before PPO/GRPO to join
  `training_export/policy_dataset_v1/policy_samples.jsonl` with
  `policy_dataset_v1/evaluator_labels.jsonl`, resolve image paths, emit
  Qwen-compatible multimodal SFT JSONL plus `qwen_action_preferences.jsonl` for
  guard-replaced actor actions, and write an audit report covering reward
  filters, missing images, actor-vs-executed action targets, and privileged-token
  leakage.
- Run `flatdisk-sim-nav-training-readiness` over both the `training_export/`
  tree and any `qwen_tool_training/` output before choosing a training method.
  The readiness assertion now reports raw policy samples, accepted Qwen SFT
  records, Qwen guard-replacement preferences, missing Qwen images, and
  privileged-token scans separately.
- Use `training_export/policy_review_traces.jsonl` first for failure triage and
  parallel-agent handoff; it records tool calls and contact-sheet paths without
  hidden target distances or THOR object metadata.
- Use `flatdisk-sim-analyze-nav-failures` to turn policy-review traces into
  WarmHub AgentNotes with trace-grounded prompt/tool recommendations.
- Use `rollout_groups.jsonl` / `trajectory_preferences.jsonl` for trajectory
  ranking only after the materializer can prove policy inputs remain clean and
  there are enough successful and failed runs for a meaningful comparison.
- Add model variants beyond Qwen3-VL 8B: small open VLMs for planning, separate
  critics, and alternate open-vocabulary grounding tools.
- Convert successful and failed `NavEvalRun` records into PPO/GRPO training data
  only after grouped rollouts, reward normalization/advantages, and on-policy
  rollout collection are explicit rather than inferred from offline labels.
