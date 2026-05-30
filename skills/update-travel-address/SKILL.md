---
name: update-travel-address
description: Update user location, travel address, timezone, or address-dependent project/profile settings after the user travels or temporarily changes where they are. Use when the user says they traveled, moved, are somewhere else, need their address/location changed for this session, or wants a repo/app/account/config updated with a temporary or permanent address.
---

# Update Travel Address

## Overview

Use this workflow to update address or location context without guessing, leaking, or over-recording personal address details. Treat "address" as ambiguous until the target is clear: it may mean current session location, timezone, a shipping/mailing address, an app profile, an environment variable, or a project config value.

## Workflow

1. Identify the target.
   - If the user names a file, app, service, profile, or config, use that target.
   - If they only say "change my address/location", ask one concise question for the destination target unless current context makes it obvious.
   - If this is only for the active Codex session, acknowledge the new location/timezone in the response; do not claim durable memory was changed.

2. Collect only the needed location details.
   - Prefer city/region/timezone when a precise street address is not required.
   - Ask for the exact street address only when updating a target that needs it.
   - Ask whether the change is temporary or permanent when that affects what should be edited.
   - Use absolute dates for effective or return dates.

3. Find the storage location.
   - Search the repo or target system for terms such as `address`, `location`, `timezone`, `profile`, `home`, `shipping`, `billing`, and `travel`.
   - Inspect surrounding code/config before editing so the update follows existing naming, formatting, and validation rules.
   - Preserve unrelated user changes in the worktree.

4. Make the smallest safe update.
   - Update only the chosen address/location fields and directly coupled values such as timezone or locale.
   - Do not store a full physical address in this skill file.
   - Do not echo a full street address in final output unless the user needs to verify the exact value.
   - If the target is an external account or live service, confirm the destination and permanence before making a real write.

5. Verify and report.
   - Run the narrowest relevant validation: config parse, tests, app health check, or a read-back of the changed profile/config.
   - Report what target was updated, whether the change is temporary or permanent, and any validation performed.
   - If only session context was updated, state the active city/region/timezone being used for the rest of the session.

## Good Prompts

- "I traveled to Austin. Update my location for this session."
- "Change my travel address in the project config to the address I just gave you."
- "I'm in Chicago until May 3, 2026. Use Central time for anything local."
- "Set my shipping address back to home."
