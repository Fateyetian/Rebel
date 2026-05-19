"""
ReBel (Reward Belief) Framework - Prompt Templates for ALFWorld

Belief schema:
  state.objects              : {obj_id: "location | in_hand | unknown"}
  state.states               : {obj_id: "heated | cooled | cleaned | sliced | none"}
  state.visited              : [receptacle_id, ...]
  state.unvisited_candidates : [receptacle_id, ...]
  task.phase                 : find | navigate | pickup | transform | place | done
  task.target                : current target object
  task.subgoal               : immediate one-sentence subgoal
  prediction                 : "If I execute [action], I expect [next-frame observation]."

Template usage:
  ALFWORLD_TEMPLATE_REBEL            — RL inference, step >= 1 (includes admissible_actions)
  ALFWORLD_TEMPLATE_NO_HIS_REBEL     — RL inference, step 0   (includes admissible_actions)
  ALFWORLD_TEMPLATE_REBEL_SFT        — SFT training, step >= 1 (no admissible_actions)
  ALFWORLD_TEMPLATE_NO_HIS_REBEL_SFT — SFT training, step 0   (no admissible_actions)
  ALFWORLD_REBEL_TAGGING_TEMPLATE    — Teacher-LLM annotation
"""


# ──────────────────────────────────────────────────────────────────────────────
# RL Inference  ·  Step >= 1  (with prior belief state, includes admissible_actions)
# ──────────────────────────────────────────────────────────────────────────────

ALFWORLD_TEMPLATE_REBEL = """You are an expert agent operating in the ALFRED embodied environment. Your task is: {task_description}. You have already taken {step_count} steps before this turn. The most recent {history_length} actions are: {action_history}. You are now at step {current_step}, and your current observation is: {current_observation}.

You must first update your belief state based on the current observation, then reason from the updated belief state, and finally select the most appropriate action. Prior observations are not explicitly provided, as their effective information has been compressed into the previous belief state.

The previous belief state is as follows:

<belief_prev>
{previous_belief}
</belief_prev>

**Step 1: Update Belief State**

Integrate the previous belief state with the current observation to fully update the physical state of the environment and the current task status.

**Step 2: Reason from Belief**

Based on the updated belief state, reason step by step to determine the most critical task phase to advance, and identify the optimal next action.

**Step 3: Action Selection**

The currently admissible actions are: [{admissible_actions}]. Considering the overall task goal, the updated belief state, and the current subgoal, select the action most beneficial for advancing the task. If the subgoal is uncertain, prioritize actions with the highest information gain (e.g., exploring unvisited receptacles) or lowest risk.

Now output strictly in the following format:

<belief>
{{
  "state": {{
    "objects": {{"[obj_id]": "[location | in_hand | unknown]"}},
    "states": {{"[obj_id]": "[heated | cooled | cleaned | sliced | none]"}},
    "visited": ["[receptacle_id]"],
    "unvisited_candidates": ["[receptacle_id]"]
  }},
  "task": {{
    "phase": "[find | navigate | pickup | transform | place | done]",
    "target": "[current target object]",
    "subgoal": "[immediate one-sentence subgoal]"
  }},
  "prediction": "If I execute [planned_action], I expect [next-frame expected observation]."
}}
</belief>

<think>
[2-3 sentences: based on the updated belief state, state what has been confirmed, what is still missing, what task phase is currently active, and why the selected action is the most optimal approach.]
</think>

<action>
[Exact action string selected from admissible actions]
</action>"""


# ──────────────────────────────────────────────────────────────────────────────
# RL Inference  ·  Step 0  (no prior belief state, includes admissible_actions)
# ──────────────────────────────────────────────────────────────────────────────

ALFWORLD_TEMPLATE_NO_HIS_REBEL = """You are an expert agent operating in the ALFRED embodied environment. Your task is: {task_description}. This is your first step. Your current observation is: {current_observation}.

You must construct your initial belief state from the current observation, then reason from it, and finally select the most appropriate action.

**Step 1: Construct Initial Belief State**

Based solely on the current observation and the task description, initialize the belief state. Record all objects and receptacles currently visible. Set objects whose locations are unconfirmed to "unknown". Identify likely unvisited receptacles that may be relevant to the task.

**Step 2: Reason from Belief**

Based on the initial belief state, determine the current task phase and the most appropriate first subgoal.

**Step 3: Action Selection**

The currently admissible actions are: [{admissible_actions}]. Select the action most beneficial for advancing toward the task goal.

Now output strictly in the following format:

<belief>
{{
  "state": {{
    "objects": {{"[obj_id]": "[location | in_hand | unknown]"}},
    "states": {{"[obj_id]": "[heated | cooled | cleaned | sliced | none]"}},
    "visited": ["[receptacle_id]"],
    "unvisited_candidates": ["[receptacle_id]"]
  }},
  "task": {{
    "phase": "[find | navigate | pickup | transform | place | done]",
    "target": "[current target object]",
    "subgoal": "[immediate one-sentence subgoal]"
  }},
  "prediction": "If I execute [planned_action], I expect [next-frame expected observation]."
}}
</belief>

<think>
[2-3 sentences: state the current task goal, what you can confirm from the initial observation, and why the selected first action is the most appropriate starting move.]
</think>

<action>
[Exact action string selected from admissible actions]
</action>"""


# ──────────────────────────────────────────────────────────────────────────────
# SFT Training  ·  Step >= 1  (with prior belief state, NO admissible_actions)
# ──────────────────────────────────────────────────────────────────────────────

ALFWORLD_TEMPLATE_REBEL_SFT = """You are an expert agent operating in the ALFRED embodied environment. Your task is: {task_description}. You have already taken {step_count} steps before this turn. The most recent {history_length} actions are: {action_history}. You are now at step {current_step}, and your current observation is: {current_observation}.

You must first update your belief state based on the current observation, then reason from the updated belief state, and finally select the most appropriate action. Prior observations are not explicitly provided, as their effective information has been compressed into the previous belief state.

The previous belief state is as follows:

<belief_prev>
{previous_belief}
</belief_prev>

**Step 1: Update Belief State**

Integrate the previous belief state with the current observation to fully update the physical state of the environment and the current task status.

**Step 2: Reason from Belief**

Based on the updated belief state, reason step by step to determine the most critical task phase to advance, and identify the optimal next action.

**Step 3: Action Selection**

Considering the overall task goal, the updated belief state, and the current subgoal, select the action most beneficial for advancing the task. If the subgoal is uncertain, prioritize actions with the highest information gain (e.g., exploring unvisited receptacles) or lowest risk.

Now output strictly in the following format:

<belief>
{{
  "state": {{
    "objects": {{"[obj_id]": "[location | in_hand | unknown]"}},
    "states": {{"[obj_id]": "[heated | cooled | cleaned | sliced | none]"}},
    "visited": ["[receptacle_id]"],
    "unvisited_candidates": ["[receptacle_id]"]
  }},
  "task": {{
    "phase": "[find | navigate | pickup | transform | place | done]",
    "target": "[current target object]",
    "subgoal": "[immediate one-sentence subgoal]"
  }},
  "prediction": "If I execute [planned_action], I expect [next-frame expected observation]."
}}
</belief>

<think>
[2-3 sentences: based on the updated belief state, state what has been confirmed, what is still missing, what task phase is currently active, and why the selected action is the most optimal approach.]
</think>

<action>
[Exact action string]
</action>"""


# ──────────────────────────────────────────────────────────────────────────────
# SFT Training  ·  Step 0  (no prior belief state, NO admissible_actions)
# ──────────────────────────────────────────────────────────────────────────────

ALFWORLD_TEMPLATE_NO_HIS_REBEL_SFT = """You are an expert agent operating in the ALFRED embodied environment. Your task is: {task_description}. This is your first step. Your current observation is: {current_observation}.

You must construct your initial belief state from the current observation, then reason from it, and finally select the most appropriate action.

**Step 1: Construct Initial Belief State**

Based solely on the current observation and the task description, initialize the belief state. Record all objects and receptacles currently visible. Set objects whose locations are unconfirmed to "unknown". Identify likely unvisited receptacles that may be relevant to the task.

**Step 2: Reason from Belief**

Based on the initial belief state, determine the current task phase and the most appropriate first subgoal.

**Step 3: Action Selection**

Select the action most beneficial for advancing toward the task goal.

Now output strictly in the following format:

<belief>
{{
  "state": {{
    "objects": {{"[obj_id]": "[location | in_hand | unknown]"}},
    "states": {{"[obj_id]": "[heated | cooled | cleaned | sliced | none]"}},
    "visited": ["[receptacle_id]"],
    "unvisited_candidates": ["[receptacle_id]"]
  }},
  "task": {{
    "phase": "[find | navigate | pickup | transform | place | done]",
    "target": "[current target object]",
    "subgoal": "[immediate one-sentence subgoal]"
  }},
  "prediction": "If I execute [planned_action], I expect [next-frame expected observation]."
}}
</belief>

<think>
[2-3 sentences: state the current task goal, what you can confirm from the initial observation, and why the selected first action is the most appropriate starting move.]
</think>

<action>
[Exact action string]
</action>"""


# ──────────────────────────────────────────────────────────────────────────────
# Annotation  ·  Teacher-LLM hindsight labeling
# ──────────────────────────────────────────────────────────────────────────────

ALFWORLD_REBEL_TAGGING_TEMPLATE = """You are an expert Embodied AI trajectory annotation and reasoning reconstruction specialist. Your task is not to execute environment actions directly, but to reconstruct the agent's belief update and local reasoning process at the current timestep, given the existing trajectory, current observation, and the previous belief state.

You must strictly adhere to World Model consistency: reason only based on information observable up to the current timestep, historical action outcomes, and confirmed facts. Do not use future information, knowledge outside the trajectory, or unverified assumptions. If the current observation conflicts with the previous belief state, trust the current observation and explicitly correct the old belief.

Your task is: {task_description}. You have already taken {step_count} steps before this turn. The most recent {history_length} actions are: {action_history}. You are now at step {current_step}, and your current observation is: {current_observation}.

Prior observations are not explicitly provided, as their effective information has been compressed into the previous belief state.

The previous belief state is as follows:

<belief_prev>
{previous_belief}
</belief_prev>

The current expert think is as follows:

<expert_think>
{expert_think}
</expert_think>

The current expert action is as follows:

<action_expert>
{expert_action}
</action_expert>

====================
Annotation Requirements
====================

1. First update the belief state based on the current observation.
2. Then, based on the updated belief state, briefly state what has been confirmed, what is still missing, what task phase is currently active, and why the expert action is justified.
3. Do not choose an action yourself; only reconstruct and explain the expert action.
4. Use only the current observation, historical action outcomes, and the previous belief state. Do not introduce future information or external knowledge.
5. Write "unknown" for any uncertain information in the belief; do not speculate.
6. prediction must describe the most likely verifiable change in the next observation after executing the expert action.

====================
Belief Update Rules
====================

- objects: Record the confirmed location or holding status of each object; write "unknown" if not confirmed.
- states: Record confirmed object states, such as heated / cooled / cleaned / sliced / none.
- visited: Record receptacles or areas that have actually been inspected and yielded useful information.
- unvisited_candidates: Record receptacles or areas not yet inspected but still potentially relevant to the current task.
- If the current observation conflicts with the previous belief, trust the current observation and explicitly correct the old state.
- The belief should retain only the minimal necessary information useful for the current decision.

Notes:
- target refers to the core object currently being processed in this phase.
- visited only includes receptacles or areas that have actually been inspected; once inspected, they count as visited.
- unvisited_candidates only includes receptacles or areas that have not yet been inspected but may still be relevant.

====================
Task Phase Determination
====================

phase must be exactly one of the following:

- find: The target object's location has NOT yet been confirmed. Transition OUT of find immediately when the current observation or any confirmed entry in state.objects reveals where the target is — do NOT wait until the agent physically arrives there.
- navigate: The target location IS known (confirmed in state.objects as a specific receptacle, not "unknown"), but the agent has NOT yet arrived. Use this phase for all "go to X" steps after the location is confirmed.
- pickup: The agent is standing at the target location and the target object is within reach (visible in current observation) OR the agent has just picked it up (obs contains "you pick up"). If gt_action is a "take" action, phase must be pickup.
- transform: The agent is holding the target (in_hand) and the task requires processing it (heat / cool / clean / slice). Use transform for all steps while holding and processing the object.
- place: The target has been processed (or no processing required) and the agent is navigating to or placing it at the destination receptacle.
- done: The task is complete.

Phase transition rules (apply strictly):
1. find → navigate: triggered the moment state.objects records the target's location (not "unknown"). Even if the agent hasn't moved yet, use navigate.
2. navigate → pickup: triggered when the agent arrives and the obs shows the target is reachable.
3. pickup → transform: triggered when the task requires processing AND the agent picks up the object.
4. pickup → place: triggered when no processing is needed AND the agent picks up the object.
5. transform → place: triggered when the processing action succeeds (obs confirms heated/cooled/cleaned/sliced).

====================
Output Format
====================

Output strictly in the following format:

<belief>
{{
  "state": {{
    "objects": {{"[obj_id]": "[location | in_hand | unknown]"}},
    "states": {{"[obj_id]": "[heated | cooled | cleaned | sliced | none]"}},
    "visited": ["[receptacle_id]"],
    "unvisited_candidates": ["[receptacle_id]"]
  }},
  "task": {{
    "phase": "[find | navigate | pickup | transform | place | done]",
    "target": "[current target object]",
    "subgoal": "[immediate one-sentence subgoal]"
  }},
  "prediction": "If I execute [expert_action], I expect [next-frame verifiable observation change]."
}}
</belief>

<think>
[2-3 sentences: state what has been confirmed, what is still missing, what task phase is currently active, and why the expert action is justified. Do not repeat the full history; do not introduce additional speculation.]
</think>"""


# ──────────────────────────────────────────────────────────────────────────────
# Template selector
# ──────────────────────────────────────────────────────────────────────────────

def get_prompt_template(
    template_type: str = "default",
    has_history: bool = True,
    has_plan: bool = False,
    sft: bool = False,
) -> str:
    """
    Return the appropriate prompt template for ALFWorld.

    Args:
        template_type: Ignored (kept for backward compatibility).
        has_history:   True for step >= 1 (prior belief available); False for step 0.
        has_plan:      Ignored (kept for backward compatibility).
        sft:           True → return SFT template (no admissible_actions placeholder).
                       False (default) → return RL template (includes admissible_actions).

    Returns:
        Prompt template string.
    """
    if sft:
        return ALFWORLD_TEMPLATE_REBEL_SFT if has_history else ALFWORLD_TEMPLATE_NO_HIS_REBEL_SFT
    return ALFWORLD_TEMPLATE_REBEL if has_history else ALFWORLD_TEMPLATE_NO_HIS_REBEL
