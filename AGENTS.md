# Agent Startup: Open-Vocab Navigation Research

For work on the open-vocabulary indoor navigation research loop, use Warmhub as
the durable cross-agent memory layer:

```bash
export WARMHUB_REPO=bencaunt-2/open-vocab-nav-research-loop
wh prime
wh repo describe --json
uv run --project sim flatdisk-sim-research-warmhub status --limit 20
uv run --project sim flatdisk-sim-research-warmhub task-list --status planned --ready-only --limit 20
wh thing query --shape AgentTask --where status=running --limit 20 --json
wh thing query --shape NavEvalRun --limit 20 --json
wh thing query --shape PromptVariant --limit 20 --json
wh assertion list --shape FailureObservation --limit 20 --json
wh thing search "current open vocabulary navigation failures qwen florence exploration" --mode hybrid --json
```

Record each simulator or robot run as a `NavEvalRun`, attach generated artifacts
as `NavArtifact` records, and write `FailureObservation` assertions for failures
or suspicious successes. Keep policy inputs true to the real robot: RGB frames,
previous motion strips, IMU yaw, bounded tool results, and model memory. Hidden
THOR metadata is evaluator/debug data only.
Research configs default to `strict_model_based=true`: non-model smoke runners
and THOR-derived semantic topomap term routing are rejected unless a config
explicitly opts out for debugging. Strict `noHardcodedLabelsOrColors` results
also require a clean static prompt audit.

For planned queue work, use:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-claim \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --owner agent-name \
  --note "Starting topomap fixture build."

uv run --project sim flatdisk-sim-research-warmhub task-finish \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --status complete \
  --summary "Topomap fixture build finished; see attached output artifacts."
```

Planned tasks may include `notes.prerequisites`. Prefer `task-list
--ready-only` or the Runpod dispatcher default so trial slices wait for fixture
and preflight tasks. `task-claim` rejects incomplete prerequisites unless
`--force` is supplied.

For simple queued command execution, a worker can claim, run
`notes.commands[N]`, and finish the task in one command. Use `--all-commands`
for multi-command tasks such as topomap fixture generation:

```bash
uv run --project sim flatdisk-sim-research-warmhub task-run-command \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --complete-exit-code 0 \
  --complete-exit-code 2 \
  --log-file sim/scratch/agent_logs/qwen-strategy-v4-topomap-fixtures.log
```

Use `task-start` only for new ad hoc work that is not already represented by a
planned `AgentTask`. Use `note` for durable scratchpad observations.
For research-loop commands, exit code `2` means the eval produced unsuccessful
navigation or structured failure data; treat it as completed evidence unless the
artifact/log output shows an infrastructure crash.

Before using exported runs for SFT/GRPO/PPO, materialize and audit Qwen tool-use
records:

```bash
uv run --project sim flatdisk-sim-prepare-qwen-tool-training \
  --input path/to/training_export/policy_dataset_v1 \
  --output-dir path/to/qwen_tool_training
```

The materializer writes Qwen-compatible multimodal SFT messages only for samples
with clean policy inputs, positive SFT weight, existing image artifacts, and
matching actor/executed actions. Evaluator rewards stay outside `messages`.

For Runpod/Linux slices, prefer the `qwen-strategy-runpod-linux-v4` queue from
`experiments/2026-06-13-open-vocab-nav-research-loop/qwen_strategy_sweep_runpod_linux.json`
or `scripts/runpod_open_vocab_nav_research_loop.sh`. Use
`flatdisk-sim-research-loop --preflight-only` or `PREFLIGHT_ONLY=1` to validate
Qwen endpoint and topomap fixtures before launching THOR episodes.
Set `START_QWEN_SERVER=1` on the wrapper when the pod should start its own
local vLLM server for `Qwen/Qwen3-VL-8B-Instruct` on `127.0.0.1:8000`.
The v4 fixture, preflight, and topomap-memory slice commands include conditional
`flatdisk-sim-build-semantic-topomap` guards, so fresh pods can create missing
CLIP map artifacts before running the eval without rebuilding existing maps.

To turn a planned Warmhub task into a Runpod pod launch command, use dry-run
first:

```bash
uv run --project sim flatdisk-sim-runpod-launch-task \
  --task qwen-strategy-runpod-linux-v4-topomap-fixtures \
  --agent agent-name \
  --all-commands \
  --start-qwen-server \
  --terminate-after 4h
```

Real launches require `runpodctl`, `RUNPOD_API_KEY`, a pushed git ref containing
the worker code, and a pod image or `WH_INSTALL_CMD` that provides the `wh` CLI.
Codex shells do not automatically load the repo root `.env`; if `.env` contains
`RUNPOD_API_KEY`, load only that variable before Runpod commands and do not print
it:

```bash
export RUNPOD_API_KEY="$(grep -m1 '^RUNPOD_API_KEY=' .env | cut -d= -f2- | sed 's/^"//;s/"$//')"
runpodctl user
```

`runpodctl user` returning account JSON means Runpod auth is available. An
`api key not configured` error means the variable was not loaded into that
shell, not necessarily that the key is invalid.

Current handoff for GRPO training, 2026-06-14:

- The repo root `.env` `RUNPOD_API_KEY` was verified with `runpodctl user` when
  loaded by the command above. Do not print the key.
- Reuse the existing Runpod pod before launching another one:
  `pbjlh2zytzulte` / `flatdisk-openloris-scene-full`, A40 46 GB, SSH
  `root@69.30.85.150 -p 22031` with
  `/Users/bencaunt/.runpod/ssh/runpodctl-ssh-key`.
- `/workspace/flat-disk-robot-code` on that pod is not a git checkout. For a
  smoke run, create a fresh `/workspace/flat-disk-robot-code-smoke` checkout of
  `codex/open-vocab-nav-research-loop` rather than modifying the existing
  directory.
- A one-step Qwen3-VL 8B GRPO smoke completed on that pod in
  `/workspace/flat-disk-robot-code-smoke-20260614-grpo`. The result manifest is
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/qwen_grpo_training_result.json`
  with `status=complete`, `returncode=0`, and `duration_s=626.335`.
- The saved adapter is under
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run/adapter`;
  `adapter_model.safetensors` is about 87 MB.
- A longer 4-step capped LoRA GRPO run also completed on the same pod in
  `/workspace/flat-disk-robot-code-train-20260614-grpo4`, from pushed commit
  `70df4be`. The job directory is
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48`.
  The result manifest has `status=complete`, `returncode=0`, and
  `duration_s=1783.814`; it trained 4 steps with `--max-completion-length 48`.
  The final adapter is
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48/adapter/adapter_model.safetensors`
  and is about 87 MB.
- The 4-step run reported train runtime about 1551 s, train loss `0.01178`,
  reward mean `-0.4789`, reward std `0.1339`, mean completion length `47.25`,
  clipped ratio `0.9688`, and step time about 383 s. The high clipped ratio is
  the main signal to address before scaling much further.
- A logged one-step run from commit `0961739` in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-log1` confirmed the
  failure mode: `completion_samples.jsonl` had 8 samples, `0/8` parsed actions,
  and every completion was cut off after starting a larger JSON object with
  `thought` or `grounding_audit`.
- Commit `6b4eaac` adds a generic GRPO action-only response contract to the
  training prompt. The comparison run in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-contract1` completed with
  `duration_s=675.898`, `completion_log_sample_count=8`, `8/8` parsed actions,
  `5/8` exact reference actions, `0` markdown fences, and TRL
  `completions/clipped_ratio=0`.
- A larger 4-step post-fix run from commit `98b962a` completed in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-contract4`.
  `qwen_grpo_training_result.json` reports `status=complete`, `returncode=0`,
  `duration_s=1907.898`, `completion_log_sample_count=32`, and
  `completion_log_metrics` with `32/32` parsed actions, `9/32` exact reference
  actions, `0` markdown fences, and `0` truncated texts. TRL reported
  `completions/clipped_ratio=0`, reward mean `-0.132`, reward std `0.1579`,
  and train loss `-0.01928`.
- After that run, the reward function was tightened so parsed but non-reference
  actions cannot receive positive reward from a positive source trajectory
  reward. The next run should use this reward cap before judging action-choice
  learning.
- A 4-step reward-cap run from commit `6f93b6c` completed in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-rewardcap4`.
  `qwen_grpo_training_result.json` reports `status=complete`, `returncode=0`,
  `duration_s=1919.723`, `completion_log_sample_count=32`, and
  `completion_log_metrics` with `32/32` parsed actions, `12/32` exact reference
  actions, `0` markdown fences, and `0` truncated texts. TRL reported
  `completions/clipped_ratio=0`, reward mean `-0.1365`, reward std `0.1809`,
  and train loss `-0.01638`. Manual comparison against the pre-cap 4-step run
  showed positive rewards for non-reference actions dropped from 5 to 0.
- Commit `1ce438b` adds generic GRPO action reward shaping: non-exact parsed
  actions still cannot receive positive reward, but same-tool and
  argument-key matches get a less-negative shaped signal. The 4-step comparison
  run completed in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-toolreward4`.
  Local result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_tool_reward/qwen_grpo_training_result.json`.
  It reports `status=complete`, `returncode=0`, `duration_s=1888.442`,
  `completion_log_sample_count=32`, `32/32` parsed actions, `12/32` exact
  reference actions, `17/32` same-tool matches, mean argument-match fraction
  `0.4375`, `0` positive non-reference rewards, `0` markdown fences, and `0`
  truncated texts. TRL reported train runtime about `1647` seconds, train loss
  `-0.01663`, reward mean `-0.1392`, reward std `0.1748`, and
  `completions/clipped_ratio=0`. Exact action rate did not improve over the
  reward-cap run on this tiny four-step comparison; the useful change is the
  now-measurable generic partial-credit reward surface.
- Commit `deef420` adds an opt-in GRPO planner audit and
  `--balance-reference-tools` mode. It duplicates underrepresented reference
  tool-family samples with provenance fields and records before/after tool and
  reward distributions in `dataset_action_audit`. The bathroom handoff changes
  from `98` samples (`visual_servo_object=56`, `turn_by_angle=28`,
  `check_object_grounding=13`, `stop=1`) to `168` samples
  (`visual_servo_object=56`, `turn_by_angle=56`, `check_object_grounding=52`,
  `stop=4`) with `--max-balance-multiplier 4`.
- The balanced 4-step comparison completed in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-toolbalanced4`.
  Local result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_tool_balanced/qwen_grpo_training_result.json`.
  It reports `status=complete`, `returncode=0`, `duration_s=1695.164`,
  `completion_log_sample_count=32`, `32/32` parsed actions, `12/32` exact
  reference actions, `20/32` same-tool matches, mean argument-match fraction
  `0.489583`, `0` positive non-reference rewards, `0` markdown fences, and `0`
  truncated texts. TRL reported train runtime about `1467` seconds, train loss
  `-0.01278`, reward mean `-0.1391`, reward std `0.1333`, and
  `completions/clipped_ratio=0`. Compared with the unbalanced shaped-reward run,
  exact stayed `12/32`, same-tool matches improved from `17/32` to `20/32`,
  mean argument-match fraction improved from `0.4375` to `0.489583`, mean
  completion length dropped from `92.406` to `84.938`, and expected tool
  exposure shifted toward turn/check samples.
- The balanced 8-step diagnostic completed in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-toolbalanced8`, from
  commit `ea4d3fd`. Local result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_8step_cap48_tool_balanced/qwen_grpo_training_result.json`.
  It reports `status=complete`, `returncode=0`, `duration_s=3212.117`,
  `completion_log_sample_count=64`, `64/64` parsed actions, `25/64` exact
  reference actions, `36/64` same-tool matches, mean argument-match fraction
  `0.455729`, `0` positive non-reference rewards, `0` markdown fences, and `0`
  truncated texts. TRL reported train runtime about `2973` seconds, train loss
  `-0.01136`, reward mean `-0.1544`, reward std `0.1401`, step time `366.6`
  seconds, and `completions/clipped_ratio=0`. The remote adapter saved under
  `adapter/adapter_model.safetensors` and is about 87 MB. The top-level LoRA
  adapter files were copied back locally under the same job's `adapter/`
  directory; the optimizer checkpoint was left on the pod.
- The 8-step run is useful mainly as a bottleneck diagnostic. Compared with the
  balanced 4-step run, exact rate only moved from `12/32` (`0.375`) to `25/64`
  (`0.390625`), while same-tool rate dropped from `0.625` to `0.5625` and mean
  argument match dropped from `0.489583` to `0.455729`. The completion log shows
  strong `turn_by_angle` overuse: expected tools were `visual_servo_object=20`,
  `turn_by_angle=26`, `check_object_grounding=18`, but parsed tools were
  `visual_servo_object=8`, `turn_by_angle=48`, `check_object_grounding=8`.
  Exact-by-tool was `visual_servo_object=4/20`, `turn_by_angle=15/26`, and
  `check_object_grounding=6/18`.
- Commit `987ec94` adds an opt-in
  `--zero-reward-exact-action-bonus` GRPO ablation. The bonus is generic,
  applies only to exact reference actions whose source `candidate_step_reward`
  is present and zero, and is not exposed in the prompt. Missing source rewards
  remain non-positive. Tests in `sim/tests/test_qwen_grpo_training.py` cover the
  planner audit, generated train script, prompt non-leakage, generic tool names,
  and non-reference positive reward cap.
- The exact-bonus 4-step comparison completed in
  `/workspace/flat-disk-robot-code-train-20260614-grpo-exactbonus4`, from
  commit `987ec94`. Local result:
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_tool_balanced_exact_bonus005/qwen_grpo_training_result.json`.
  It reports `status=complete`, `returncode=0`, `duration_s=1626.631`,
  `completion_log_sample_count=32`, `32/32` parsed actions, `11/32` exact
  reference actions, `19/32` same-tool matches, mean argument-match fraction
  `0.458333`, `0` positive non-reference rewards, `0` markdown fences, and `0`
  truncated texts. TRL reported train runtime about `1397` seconds, train loss
  `-0.01657`, reward mean `-0.1313`, reward std `0.1537`, step time `344.5`
  seconds, and `completions/clipped_ratio=0`. The remote LoRA adapter saved and
  top-level adapter files were copied back locally under the same job's
  `adapter/` directory.
- The exact-bonus ablation did not improve this short diagnostic. Compared with
  the no-bonus balanced 4-step run, exact matches dropped from `12/32` to
  `11/32`, same-tool matches dropped from `20/32` to `19/32`, and mean argument
  match dropped from `0.489583` to `0.458333`. The persistent pattern remains
  turn overuse: expected tools were `visual_servo_object=4`, `turn_by_angle=14`,
  `check_object_grounding=14`, while parsed tools were `visual_servo_object=5`,
  `turn_by_angle=20`, `check_object_grounding=7`. Exact-by-tool was
  `visual_servo_object=0/4`, `turn_by_angle=6/14`, and
  `check_object_grounding=5/14`.
- A held-out completion-eval harness now exists for planned GRPO datasets:
  `flatdisk-sim-plan-qwen-grpo-completion-eval` and
  `flatdisk-sim-run-qwen-grpo-completion-eval`. It plans a separate eval job
  from an existing `qwen_grpo_training_job.json`, writes an eval-only prompt
  dataset, loads the base Qwen model plus an optional explicit PEFT adapter,
  generates action JSON completions, and scores them with the same generic
  exact/tool/argument metrics. Reference actions and reward sidecars are used
  only after generation for scoring; planner validation scans prompt messages
  for forbidden privileged tokens and scoring-sidecar leaks.
- A local smoke plan from the balanced 4-step job succeeded at
  `sim/scratch/open_vocab_nav_research_loop/qwen_grpo_completion_evals/bathroom_cross_run_lora_4step_cap48_tool_balanced_base_eval_smoke`.
  Command:
  `flatdisk-sim-plan-qwen-grpo-completion-eval --training-job sim/scratch/open_vocab_nav_research_loop/qwen_grpo_jobs/bathroom_cross_run_lora_4step_cap48_tool_balanced/qwen_grpo_training_job.json --output-dir sim/scratch/open_vocab_nav_research_loop/qwen_grpo_completion_evals/bathroom_cross_run_lora_4step_cap48_tool_balanced_base_eval_smoke --max-samples 4 --sample-stride 8 --max-new-tokens 48 --fail-on-not-ready`.
  The planned eval job reported `status=ready`, `sample_count=4`,
  `missing_image_count=0`, `forbidden_model_token_hits=[]`, and
  `sidecar_prompt_leak_hits=[]`; the runner dry-run also succeeded. This was
  not a real Qwen generation run yet.
- The working pod venv is `/workspace/flatdisk-grpo-venv`, created with
  `--system-site-packages` so it can see image Torch `2.9.1+cu128`; install
  `torchvision==0.24.1 --no-deps` to match that Torch build. Do not let pip pull
  a second Torch/CUDA stack.
- The generated GRPO script must keep prompts conversational, leave images in
  the separate `images` column, use PEFT LoRA, and cap generation length for
  smoke jobs. Full-model Adam OOMs on A40; uncapped generation makes "tiny"
  smoke runs unreasonably long.
- Next action is to improve action-choice learning signal and evaluation. The
  JSON format is stable, generic partial reward exists, and balancing improved
  early same-tool behavior, but doubling the balanced run to 8 steps did not
  solve exact action selection; the exact-action bonus ablation also did not
  help, and held-out completion-eval planning now exists. Useful next
  directions: run the completion eval on base Qwen and the saved LoRA adapters,
  collect more diverse grouped rollouts with stop/arrival and ambiguous
  grounding examples, inspect whether reference actions are too noisy or too
  correlated with turns, and only then scale to a 24-step balanced run.
- Future continuation should treat this as past verification, not a blocker.
  There is no known Runpod auth issue when `.env` is loaded as above, and no
  user action is needed before launching the next capped LoRA training attempt.
  Start from the pushed `codex/open-vocab-nav-research-loop` ref, regenerate the
  GRPO job with an explicit `--max-completion-length`, transfer only the job
  files plus dataset-referenced images, and run through
  `/workspace/flatdisk-grpo-venv`. Set `PYTHONPATH=sim/src` when invoking
  `flatdisk-sim-run-qwen-grpo-training` from a fresh checkout so the runner uses
  the checked-out source rather than an older editable install in the venv.

To fan out several planned trial-slice tasks, use the dispatcher. It is also
dry-run by default and skips incomplete prerequisites unless
`--ignore-prerequisites` is supplied:

```bash
uv run --project sim flatdisk-sim-runpod-dispatch \
  --name-prefix qwen-strategy-runpod-linux-v4-run- \
  --tag trial-slice \
  --max-workers 2 \
  --start-qwen-server \
  --terminate-after 4h
```
