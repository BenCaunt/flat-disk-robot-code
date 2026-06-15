# Flat Disk Robot Simulator

This directory contains a Zenoh-compatible simulator for the two-wheel flat disk robot. The simulator now uses AI2-THOR as the runtime and can run either generated ProcTHOR houses or named iTHOR indoor scenes while keeping the same topics as the physical firmware:

- `flatdisk/xiao/camera/jpeg`: binary `FDV1` header plus JPEG payload.
- `flatdisk/xiao/imu`: binary `FDI1` quaternion/accel/gyro payload.
- `flatdisk/xiao/status`: JSON robot/simulator status.
- `flatdisk/xiao/time_sync`: binary `FDSR` replies to `cmd/time_sync`.
- `flatdisk/xiao/cmd/motors/percent`, `cmd/motors/us`, `cmd/motors/stop`: motor commands.

The simulator intentionally does not publish `flatdisk/xiao/motors/encoders` and does not expose a pose estimate through the public API. Agent tools only receive camera frames, IMU yaw, and command results. Hidden pose is written only to simulator logs under `sim/scratch/`.

## Setup

```bash
cd sim
uv sync --extra dev --extra thor
```

ProcTHOR generation is optional because it is distributed separately from AI2-THOR:

```bash
uv sync --extra dev --extra thor --extra procthor
```

AI2-THOR downloads or launches a Unity build on first use. If ProcTHOR package resolution fails in a fresh Python environment, use the saved-house path below or install the upstream generator explicitly:

```bash
uv pip install 'procthor==0.0.1.dev2'
```

Useful upstream references:

- AI2-THOR initialization: <https://ai2thor.allenai.org/ithor/documentation/initialization/>
- AI2-THOR actions: <https://ai2thor.allenai.org/ithor/documentation/actions/navigation/>
- ProcTHOR repository: <https://github.com/allenai/procthor>

## Smoke Render

Render one camera frame without starting Zenoh:

```bash
cd sim
uv run flatdisk-sim-render-thor --backend ithor --scene FloorPlan301
uv run flatdisk-sim-render-thor --backend procthor --procthor-seed 42
```

Outputs are written to `sim/scratch/thor_render/`.

Camera defaults:

- image size: `640x480`
- camera height: `0.105 m`
- camera forward offset: `0.22 m`
- horizontal FOV: `68 deg`, converted to AI2-THOR/Unity vertical FOV at startup
- near/far planes: `0.03 m` / `20.0 m`

The robot mesh is not inserted into THOR yet. The sim currently uses the
AI2-THOR agent as the movement/collision proxy, but renders from an attached
third-party camera placed at the flat disk camera height. The URDF camera link
is much lower, roughly `0.033125 m`
above the floor if measured from the current mesh origins, so use this when you
want the physical mount estimate:

```bash
uv run flatdisk-sim-render-thor --backend ithor --scene FloorPlan301 --camera-height-m 0.033125
```

The physical URDF camera forward offset is roughly `0.102 m`, but the default
sim offset is `0.22 m` to keep the render camera clear of the AI2-THOR proxy
body and avoid self-artifacts. If you want the stricter physical extrinsic, pass:

```bash
uv run flatdisk-sim --backend ithor --scene FloorPlan201 --camera-forward-offset-m 0.102
```

If you have a real ESP32 calibration from
`scripts/checkerboard_calibration_logger.py`, pass it to derive the vertical FOV
from the calibrated focal length:

```bash
uv run flatdisk-sim-render-thor \
  --backend ithor \
  --scene FloorPlan301 \
  --camera-calibration ../captures/esp32_calibration/camera_calibration.json
```

## Run The Zenoh-Compatible Simulator

Start a generated ProcTHOR house:

```bash
cd sim
uv run flatdisk-sim --backend procthor --procthor-seed 42 --listen tcp/127.0.0.1:7447
```

Run a named iTHOR scene instead:

```bash
uv run flatdisk-sim --backend ithor --scene FloorPlan201 --listen tcp/127.0.0.1:7447
```

Scene shortcuts for `--backend ithor` are supported: `kitchens`, `living_rooms`, `bedrooms`, `bathrooms`, and `random`.

Run a saved ProcTHOR house JSON:

```bash
uv run flatdisk-sim --backend house-json --house-json path/to/house.json
```

The same camera controls are available on the Zenoh bridge:

```bash
uv run flatdisk-sim \
  --backend ithor \
  --scene FloorPlan201 \
  --camera-height-m 0.105 \
  --camera-forward-offset-m 0.22 \
  --field-of-view 68 \
  --field-of-view-axis horizontal
```

For debugging only, pass `--use-agent-camera` to render the raw AI2-THOR agent
camera. That view is much higher and does not match the flat disk camera.

Then use existing robot tooling from the repository root, for example:

```bash
python scripts/flatdisk_robot_mcp_server.py
python scripts/motor_gui_imu_heading.py
```

## Build A Privileged Semantic Topomap

Semantic topomaps are built offline with simulator-only access to reachable
floor positions, hidden poses, and object metadata. The saved map contains
images, image descriptors, semantic terms, and graph edges; the live robot API
only needs camera frames and a text goal.

Check runtime readiness first:

```bash
cd sim
uv run flatdisk-sim-nomad-topomap-doctor \
  --visualnav-repo /path/to/visualnav-transformer \
  --diffusion-policy-repo /path/to/diffusion_policy
```

```bash
cd sim
uv sync --extra dev --extra thor
uv run flatdisk-sim-build-semantic-topomap \
  --backend ithor \
  --scene FloorPlan401 \
  --output-dir scratch/semantic_topomaps/bathroom_401 \
  --max-positions 45 \
  --yaw-count 4 \
  --clean
```

For text-image CLIP search, add `--clip` when building and querying. The
default semantic search already uses privileged THOR object labels, so CLIP is
optional.

Query a route from a current frame to a semantic target:

```bash
uv run flatdisk-sim-query-semantic-topomap \
  --map-dir scratch/semantic_topomaps/bathroom_401 \
  --image scratch/semantic_topomaps/bathroom_401/images/000000.jpg \
  --goal toilet
```

Run the same topomap API against the real robot or simulator Zenoh stream:

```bash
python ../scripts/nomad_topomap_zenoh.py \
  --map-dir scratch/semantic_topomaps/bathroom_401 \
  --goal toilet \
  --visualnav-repo /path/to/visualnav-transformer \
  --diffusion-policy-repo /path/to/diffusion_policy \
  --checkpoint /Users/bencaunt/Downloads/nomad.pth
```

Add `--arm` only after route tracking and NoMaD inference look sane. Without
`--arm`, the script publishes status to
`flatdisk/xiao/nomad_topomap/status` but does not command the motors. Use
`--no-nomad` to test frame matching, semantic goal lookup, and route cursor
advancement without the upstream NoMaD dependency stack.

The same behavior is available as a small Python API:

```python
from flatdisk_sim.nomad_policy import NoMaDPolicy
from flatdisk_sim.semantic_topomap import SemanticTopomap
from flatdisk_sim.topomap_navigator import NavigatorConfig, SemanticTopomapNavigator

topomap = SemanticTopomap.load("scratch/semantic_topomaps/bathroom_401")
nomad = NoMaDPolicy(
    checkpoint="/Users/bencaunt/Downloads/nomad.pth",
    visualnav_repo="/path/to/visualnav-transformer",
    diffusion_policy_repo="/path/to/diffusion_policy",
)
navigator = SemanticTopomapNavigator(
    topomap,
    NavigatorConfig(goal="toilet"),
    command_policy=nomad,
)

sequence = navigator.get_sequence(current_frame).node_ids
step = navigator.drive_to_goal(current_frame, armed=True)
motor1, motor2 = step.command
```

For simulator evaluation over the same Zenoh interface:

```bash
uv run flatdisk-sim-evaluate-nomad-topomap \
  --map-dir scratch/semantic_topomaps/bathroom_401 \
  --goal toilet \
  --launch-sim \
  --backend ithor \
  --scene FloorPlan401 \
  --no-nomad \
  --duration 15 \
  --output-json scratch/nomad_topomap_eval/route_tracking.json
```

Remove `--no-nomad` after installing the NoMaD dependency stack, and add
`--arm` when you want the evaluator to publish motor commands. With
`--launch-sim`, the evaluator tails the simulator hidden-pose JSONL log and
reports route-goal state, command counts, collision count, and best/final
distance to the selected goal node.

Small verified iTHOR smoke commands:

```bash
uv run flatdisk-sim-render-thor \
  --backend ithor \
  --scene FloorPlan401 \
  --width 160 \
  --height 120 \
  --output-dir scratch/nomad_topomap_smoke/render

uv run flatdisk-sim-build-semantic-topomap \
  --backend ithor \
  --scene FloorPlan401 \
  --render-width 160 \
  --render-height 120 \
  --output-dir scratch/nomad_topomap_smoke/map \
  --max-positions 2 \
  --yaw-count 2 \
  --object-view-count 0 \
  --clean

uv run flatdisk-sim-query-semantic-topomap \
  --map-dir scratch/nomad_topomap_smoke/map \
  --image scratch/nomad_topomap_smoke/map/images/000000.jpg \
  --goal toilet

uv run flatdisk-sim-evaluate-nomad-topomap \
  --map-dir scratch/nomad_topomap_smoke/map \
  --goal toilet \
  --launch-sim \
  --backend ithor \
  --scene FloorPlan401 \
  --no-nomad \
  --duration 4 \
  --stop-on-route-goal
```

Route progress uses image-descriptor matching first. When NoMaD is loaded, the
navigator also uses NoMaD's distance head to advance interim goal frames; tune
this with `--nomad-close-threshold` and `--nomad-advance-margin`.

NoMaD inference constructs the model directly from the upstream
`robodhruv/visualnav-transformer` training modules and Stanford's
`diffusion_policy` UNet, avoiding the ROS-oriented deployment utilities. The
simulator package exposes PyTorch, `diffusers`, `efficientnet_pytorch`, and
`einops` as a `nomad` extra where possible, but `diffusion_policy` still needs
to be installed from its upstream repository or passed with
`--diffusion-policy-repo`.

## Run Agent Tasks

In another terminal:

```bash
cd sim
uv run flatdisk-sim-agent bathroom --planner openai
uv run flatdisk-sim-agent find-object --target "mug" --planner openai
```

Each run writes a report under `sim/scratch/<timestamp>_<task>/` with:

- `events.jsonl`: tool calls, image observations, IMU readings, and actions.
- `frames/`: raw camera frames returned to the agent.
- `episode_summary.json`: structured success/failure summary.
- `report.md`: human-readable report of what worked and what did not.

The heuristic visual planner was originally tuned for the old toy scene. For THOR scenes, use `--planner openai` or retune the local vision labels for the target objects in the selected scene.

## Evaluate Camera+IMU Text-Goal Navigation

Run the text-goal evaluator against the Zenoh AI2-THOR bridge. This launches
one simulator process per episode, connects through `AgentTools`, and gives the
policy only the same inputs available on the physical robot: a low RGB camera
frame, IMU yaw, the natural-language goal, and the policy's own recent action
history.

```bash
cd sim
uv sync --extra dev --extra thor --extra vlm
uv run flatdisk-sim-evaluate-text-goals --episodes bedroom_bed living_room_sofa
```

The evaluator writes one directory per episode under
`sim/scratch/thor_text_goal_eval/`. Each episode includes raw camera frames,
motion frame sheets, hidden-evaluator progress renders, `events.jsonl`, and
`episode_summary.json`. The aggregate run also writes `aggregate_summary.json`
and `report.md`.

The policy boundary is intentionally narrow:

- allowlist: camera frame, IMU yaw, text goal, and recent policy actions
- policy actions: `turn_by_angle` and `drive_straight`
- external supervisor action: `stop` after evaluator success, operator stop, or timeout
- denied to policy: map, pose, collision flags, encoders, status, scene name,
  simulator metadata, object coordinates, and target regions

Hidden THOR pose/object metadata is written only to the bridge log and read only
by `flatdisk_sim.evaluate_text_goals` for scoring and progress rendering. Each
episode uses a unique Zenoh namespace to avoid cross-talk with the real robot or
another simulator instance.

The same policy action contract can be driven by the real robot tools:

- observe: `latest_frame` plus `get_angle` / IMU yaw
- act: `turn_by_angle`, `drive_straight`, and supervisor/operator `stop`

No encoder topic is required by the policy or evaluator.

## Run A Policy Competition

Run the registered policies on held-out iTHOR scenes and rank them by
correctness first, then wall-clock time, then final target distance:

```bash
cd sim
uv run flatdisk-sim-compete-text-goals \
  --suite heldout \
  --policies control_vlm memory_vlm sprinter hf_scout
```

The policies are registered in `flatdisk_sim.policy_registry`:

- `control_vlm`: direct online VLM action selection.
- `memory_vlm`: VLM policy with non-privileged visual/action memory.
- `sprinter`: deterministic camera heuristic optimized for wall-clock time.
- `hf_scout`: deterministic scout with optional local Hugging Face hooks.

`hf_scout` defaults to no downloads and no model loading. Set
`FLATDISK_HF_SCOUT_ENABLE=1` and point `FLATDISK_HF_SCOUT_MODEL` or
`FLATDISK_HF_SCOUT_DEPTH_MODEL` at cached local models to try small VLM/depth
variants from a companion computer while keeping the same camera+IMU policy
interface.

## Run The LLM Harness Dashboard

The harness is a thin actor/critic wrapper around the robot tools. It can use a
deterministic smoke runner, a scripted open-vocabulary runner, live Codex, or a
local OpenAI-compatible VLM endpoint such as Qwen3-VL through MLX-VLM, vLLM, or
SGLang. All runners use the same Zenoh robot/simulator tool surface.

```bash
cd sim
uv sync --extra dev --extra harness
uv run flatdisk-sim-harness-dashboard --connect tcp/127.0.0.1:7447
```

Open <http://127.0.0.1:8765>, enter a slash-goal style prompt, and use
`Go`, `Pause`, `Resume`, `Stop`, or the teleop buttons. Motor-capable tools are
blocked unless the dashboard is launched with `--arm`. The dashboard shows the
camera frame, current mode, a live `Reasoning Trace` stream, `Tool I/O`, and
the memory tail. `Reasoning Trace` logs actor request start, prompt/image
attachments, Qwen/Codex output, parsed `thought`, selected action, critic
request/decision, safety-gate changes, and model response metadata when the
endpoint returns it. `Tool I/O` logs each selected tool call before execution
and the returned tool result as soon as it finishes.

To use live Codex calls:

```bash
uv run flatdisk-sim-harness-dashboard \
  --connect tcp/127.0.0.1:7447 \
  --live-codex \
  --rerun
```

To use a local Qwen3-VL planner on Apple Silicon, start the MLX-VLM
OpenAI-compatible server from the repo root:

```bash
uv run python -m mlx_vlm.server \
  --host 127.0.0.1 \
  --port 8080 \
  --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --max-tokens 512
```

Then point the dashboard at that endpoint:

```bash
uv run flatdisk-sim-harness-dashboard \
  --connect tcp/127.0.0.1:7447 \
  --runner qwen \
  --arm \
  --qwen-endpoint http://127.0.0.1:8080/v1/chat/completions
```

Each dashboard run writes `harness_events.jsonl`, `memory.jsonl`, prompts,
camera frames, and optionally `harness.rrd` under
`sim/scratch/llm_harness_dashboard/`.
Rerun recordings include a viewer blueprint for mode timeline, camera, yaw,
robot command logs, dedicated actor/critic LLM output logs, aggregate harness
events, and run metadata. The metadata document records the runner, critic,
model, reasoning effort, tool backend, relative-path boundary, and policy input
allowlist.

The prompt keeps stable tool/schema/context before dynamic task state for
cache-friendly ordering. The latest RGB camera frame is attached as the
authoritative observation. After every turn, drive, or visual-servo call, the
next actor call also receives a left-to-right motion strip made from evenly
spaced frames from the previous tool call. The legacy color/object detector has
been removed from the model-facing path; the only scalar frame summary is
`brightness_center`, which is low-level and advisory.

The actor may write non-privileged scratchpad fields through `memory_update` and
may request durable frame copies through `save_frames`, using `source=latest` or
`source=previous_motion`. The harness performs those writes into
`memory_frames/`; the model never gets arbitrary filesystem write access.

The tool contract includes normal bounded movement plus `visual_servo_object`,
which runs `scripts/object_drive_zenoh.py` for a short Florence/object-detector
servo segment against the same Zenoh namespace. The default detector is
`florence-mlx` on Apple Silicon after installing `mlx-vlm`; the fixed-box
detector option has been removed from the active harness/object-drive CLIs.

The tool contract also includes `query_topomap_memory`, a non-motion lookup
that is available when `AgentTools` is configured with
`--topomap-memory-map-dir`. It uses the latest RGB frame plus the current text
goal to return policy-safe image-match and route hints, including a contact
sheet, without returning poses, scene metadata, object metadata, or hidden THOR
state. Strict runs should use CLIP-backed topomaps via
`--topomap-memory-use-clip`; `--topomap-memory-allow-semantic-terms` is intended
for debugging maps whose saved semantic terms may have come from privileged
offline construction.

The live Qwen/Codex path preserves the model critic's decision. The deterministic
safety evaluator is retained only for explicit smoke/debug runners such as
`SafetyCriticRunner`.

Live Codex runs write role-specific JSON schemas under `codex_schemas/` and pass
the latest camera frame as a `--image` attachment. Model-facing prompts and
memory use paths relative to the policy run directory, while absolute paths are
kept only in internal event logs for the dashboard and artifact lookup.

Persona lessons from the policy competition are kept general in the harness:
memory must be sanitized before reuse, each loop already observes before actor
selection, and wall-clock time improves when the actor spends steps on bounded
motion rather than redundant observations.

Evaluate the harness through the Zenoh-compatible AI2-THOR bridge:

```bash
uv sync --extra dev --extra thor --extra harness
uv run flatdisk-sim-evaluate-harness-thor \
  --episodes living_room_sofa bedroom_bed \
  --runner qwen \
  --qwen-endpoint http://127.0.0.1:8080/v1/chat/completions \
  --rerun
```

Add topomap memory to a run when a saved map is available:

```bash
uv run flatdisk-sim-evaluate-harness-thor \
  --episodes living_room_sofa \
  --runner qwen \
  --qwen-endpoint http://127.0.0.1:8080/v1/chat/completions \
  --topomap-memory-map-dir scratch/semantic_topomaps/example_map \
  --topomap-memory-use-clip
```

Dependency-light long-horizon smoke demos that exercise the same motion-strip
and memory artifacts without a live Qwen server:

```bash
uv run flatdisk-sim-evaluate-harness-thor \
  --episodes living_room_sofa bathroom_toilet \
  --runner fast-demo \
  --max-steps 14 \
  --render-width 160 \
  --render-height 120 \
  --output-dir scratch/open_vocab_success_demo

uv run flatdisk-sim-evaluate-harness-thor \
  --episodes bathroom_toilet \
  --runner scripted-open-vocab \
  --max-steps 7 \
  --render-width 160 \
  --render-height 120 \
  --output-dir scratch/open_vocab_visual_servo_demo
```

Run a parallel research-loop sweep from a JSON config. Dry-run mode writes the
trial matrix plus Warmhub-ready schema/commit artifacts without launching THOR:

```bash
uv run flatdisk-sim-research-loop \
  --config ../experiments/2026-06-13-open-vocab-nav-research-loop/qwen_prompt_sweep.json \
  --output-dir scratch/open_vocab_nav_research_loop \
  --dry-run
```

Remove `--dry-run` to execute the matrix. Use `--parallelism N` to override the
config concurrency. Add `--preflight-endpoints` to fail fast with structured
trial exceptions when configured Qwen endpoints are unavailable. Add
`--preflight-only` to check endpoints and topomap-memory fixtures without
launching THOR episodes. Use `--commit-warmhub --init-warmhub-repo` only after
reviewing the generated `warmhub_shapes.json` and `warmhub_ops.json`.

The topomap-memory comparison config adds a CLIP-backed memory variant and uses
`{episode}` in the map path so one sweep can cover per-scene maps:

```bash
uv run flatdisk-sim-research-loop \
  --config ../experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep.json \
  --output-dir scratch/open_vocab_nav_research_loop \
  --dry-run
```

For Runpod/Linux workers, use the non-MLX sibling config:

```bash
uv run flatdisk-sim-research-loop \
  --config ../experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep_runpod_linux.json \
  --output-dir scratch/open_vocab_nav_research_loop \
  --dry-run
```

Executed research-loop runs export policy-training records automatically under
`training_export/`. To export an existing run directory manually:

```bash
uv run flatdisk-sim-export-nav-training \
  --input scratch/open_vocab_nav_research_loop/20260613_000000 \
  --output-dir scratch/open_vocab_nav_research_loop/training_export_manual \
  --experiment-id manual_open_vocab_export
```

The export separates model-facing prompt/action/tool data from evaluator-only
reward labels, so hidden THOR distance/success can be used for offline ranking
or future PPO/GRPO without becoming policy input. Trainer-facing files include
`policy_dataset_v1/policy_samples.jsonl`, the joined privileged
`policy_dataset_v1/evaluator_labels.jsonl`, `rollout_groups.jsonl`, and
`trajectory_preferences.jsonl`.

Agents can write coordination records directly to Warmhub:

```bash
uv run flatdisk-sim-research-warmhub task-list \
  --status planned \
  --limit 20

uv run flatdisk-sim-research-warmhub task-claim \
  --task qwen-topomap-memory-sweep-v1-preflight \
  --owner agent-name \
  --note "Starting preflight."

uv run flatdisk-sim-research-warmhub task-start \
  --task-id qwen-failure-pass-001 \
  --owner agent-name \
  --objective "Inspect failed Qwen runs and propose the next general prompt/tool variant" \
  --tag qwen --tag failure-analysis

uv run flatdisk-sim-research-warmhub task-finish \
  --task qwen-failure-pass-001 \
  --agent agent-name \
  --status complete \
  --summary "Logged findings and proposed the next experiment."
```

Use `task-start` for ad hoc work; use `task-claim` for planned queue work.
For command-backed planned tasks, a worker can claim, run one `notes.commands`
entry, and finish in one step:

```bash
uv run flatdisk-sim-research-warmhub task-run-command \
  --task qwen-topomap-memory-runpod-linux-v1-preflight \
  --agent agent-name \
  --command-index 0 \
  --log-file scratch/agent_logs/preflight.log
```

Runpod workers can run one planned slice through the repo-level wrapper:

```bash
TASK_ID=qwen-topomap-memory-sweep-v1-run-qwen_topomap_memory_clip-living_room_sofa \
CONFIG=../experiments/2026-06-13-open-vocab-nav-research-loop/qwen_topomap_memory_sweep_runpod_linux.json \
VARIANT=qwen_topomap_memory_clip \
EPISODE=living_room_sofa \
../scripts/runpod_open_vocab_nav_research_loop.sh
```

The wrapper defaults to the Runpod/Linux config, `UV_EXTRAS=thor`, and
`UV_WITH=torch,transformers`. Set `PREFLIGHT_ONLY=1` to validate endpoint and
topomap fixtures without launching simulator episodes. The local default Qwen
and Florence settings are MLX-shaped; use the Runpod config for
Linux-compatible OpenAI-style Qwen serving and `florence-transformers` phrase
grounding. Set `START_QWEN_SERVER=1` when the pod should start
`Qwen/Qwen3-VL-8B-Instruct` itself through `scripts/runpod_start_qwen_vllm.sh`;
leave it unset when an OpenAI-compatible endpoint is already running.

To generate a `runpodctl pod create` command from a planned Warmhub task:

```bash
uv run flatdisk-sim-runpod-launch-task \
  --task qwen-topomap-memory-runpod-linux-v1-preflight \
  --agent agent-name \
  --env WH_TOKEN="$WH_TOKEN" \
  --start-qwen-server \
  --terminate-after 4h
```

The launcher is dry-run by default and redacts sensitive environment values in
its output. `--start-qwen-server` injects the vLLM startup env, waits for the
local `/v1/models` endpoint, and attaches `/workspace/qwen_vllm.log` as task
evidence. Pass a Warmhub PAT to fresh pods with `--env WH_TOKEN="$WH_TOKEN"` so
the worker can claim tasks and commit results. `WARMHUB_TOKEN` and
`WARMHUB_API_KEY` are accepted as compatibility aliases and normalized to
`WH_TOKEN` in the pod. On `--launch`, the local launcher refuses to create a pod
unless worker Warmhub auth is present; the remote worker then runs `wh auth
status` and `wh repo describe "$WARMHUB_REPO" --json` before claiming or running
the task. Use `--launch` only after the worker code is committed and pushed,
`RUNPOD_API_KEY` is valid for pod management, and the pod can install or run the
`wh` CLI.

To fan out several planned trial slices from the Warmhub queue, use the
dispatcher:

```bash
uv run flatdisk-sim-runpod-dispatch \
  --name-prefix qwen-topomap-memory-runpod-linux-v1-run- \
  --tag trial-slice \
  --max-workers 2 \
  --env WH_TOKEN="$WH_TOKEN" \
  --start-qwen-server \
  --terminate-after 4h
```

The dispatcher is also dry-run by default, filters locally after querying
planned `AgentTask` records, and emits one pod command per selected task. On
`--launch`, it verifies `RUNPOD_API_KEY` is usable and that selected workers
will receive a Warmhub auth env before reserving any `AgentTask`; each remote
worker also checks Warmhub auth and repo reachability before task execution.

This path starts the same simulator bridge as the text-goal policy evaluator and
connects the harness through `AgentTools`. The live policy receives only camera
frame attachments, previous motion strips, camera-derived summaries, IMU yaw,
bounded tool results, and memory. Hidden THOR pose/object metadata is written under
`evaluator_hidden/`, outside the policy directory, and is used only by the
evaluator for scoring and progress renders.

When running tests from this workspace, prefer:

```bash
cd sim
uv run --no-sync python -X faulthandler -m pytest -q
```

The repo-root form `uv run --project sim python -m pytest` can import conda's
native `readline` extension before the local shim is on `sys.path`, which may
segfault with exit code 139 in this environment.

## Backend Notes

`flatdisk_sim.thor_backend.FlatDiskThorSim` owns the AI2-THOR controller. It maps motor percentages into small `RotateLeft`/`RotateRight` and `MoveAhead`/`MoveBack` actions, converts `event.frame` into the same JPEG packet format as the physical camera, and derives IMU yaw from the THOR agent rotation.

The bridge exposes no THOR metadata over Zenoh beyond health/status fields. Hidden pose and scene identifiers are logged only for debugging and dataset auditing.
