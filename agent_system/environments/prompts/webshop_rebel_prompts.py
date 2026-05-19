"""
WebShop ReBel (Reward Belief) Prompt Templates

Belief schema:
  state.target      : {attr: value}          — task requirement constraints
  state.confirmed   : {attr: verified_value} — attributes verified from observation
  state.unconfirmed : [attr, ...]            — attributes still pending confirmation
  state.product_id  : "ASIN" or null        — locked product identifier
  task.phase        : searching | browsing | viewing | selecting | buying | done
  task.target       : product_id or null
  task.subgoal      : immediate one-sentence goal
  prediction        : "If I [action], I expect [next observation]."

Template usage:
  WEBSHOP_REBEL_TEMPLATE_RL          — RL inference, step >= 1 (includes admissible_actions)
  WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL   — RL inference, step 0   (includes admissible_actions)
  WEBSHOP_REBEL_TEMPLATE_SFT         — SFT training, step >= 1 (no admissible_actions)
  WEBSHOP_REBEL_TEMPLATE_NO_HIS_SFT  — SFT training, step 0   (no admissible_actions)
  WEBSHOP_REBEL_TAGGING_TEMPLATE     — Teacher-LLM annotation
"""


# ──────────────────────────────────────────────────────────────────────────────
# RL Inference  ·  Step >= 1  (with prior belief state, includes admissible_actions)
# ──────────────────────────────────────────────────────────────────────────────

WEBSHOP_REBEL_TEMPLATE_RL = """You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is: {task_description}. You have already taken {step_count} steps before this turn. The most recent {history_length} actions are: {action_history}. You are now at step {current_step}, and your current observation is: {current_observation}.

The previous belief state is as follows:

<belief_prev>
{previous_belief}
</belief_prev>

You must first update your belief state based on the current observation, then reason from the updated belief state, and finally select the most appropriate and admissible action.

**Step 1: Update Belief State**

Integrate the previous belief state with the current observation to update the product goal, confirmed attributes, unconfirmed attributes, product identifier, and task phase. confirmed must only record information directly verified from observation, and all previously confirmed entries must be preserved.

**Step 2: Reason from Belief**

Based on the updated belief state, determine the current task phase, the current bottleneck, and the most critical attribute confirmation or product filtering step to advance. If information is insufficient or the subgoal is uncertain, prioritize actions that increase information certainty and reduce risk.

**Step 3: Action Selection**

The currently admissible actions are: [{admissible_actions}]. Select the one action most beneficial for advancing the task. If the subgoal is uncertain, prioritize actions with higher information gain and lower risk.

Now output strictly in the following format:

<belief>
{{
  "state": {{
    "target": {{"[attr]": "[value]"}},
    "confirmed": {{"[attr]": "[verified value]"}},
    "unconfirmed": ["[attr]"],
    "product_id": "[ASIN or null]"
  }},
  "task": {{
    "phase": "[searching | browsing | viewing | selecting | buying | done]",
    "target": "[product_id or null]",
    "subgoal": "[immediate one-sentence goal]"
  }},
  "prediction": "If I [planned action], I expect [consequence in next observation]."
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

WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL = """You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is: {task_description}. This is your first step. Your current observation is: {current_observation}.

You must construct your initial belief state from the task description and current observation, then reason from it, and finally select the most appropriate and admissible action.

**Step 1: Construct Initial Belief State**

Based on the task description, extract all target constraints (brand, color, price range, size, material, etc.) into state.target. Set confirmed to empty (nothing has been verified yet). Populate unconfirmed with all attributes that still need to be confirmed. Set product_id to null.

**Step 2: Reason from Belief**

Based on the initial belief state, determine the current task phase (should be "searching") and formulate the first subgoal: what to search for to find a matching product.

**Step 3: Action Selection**

The currently admissible actions are: [{admissible_actions}]. Select the action most beneficial for initiating the task.

Now output strictly in the following format:

<belief>
{{
  "state": {{
    "target": {{"[attr]": "[value]"}},
    "confirmed": {{}},
    "unconfirmed": ["[attr]"],
    "product_id": null
  }},
  "task": {{
    "phase": "searching",
    "target": null,
    "subgoal": "[immediate one-sentence goal]"
  }},
  "prediction": "If I [planned action], I expect [consequence in next observation]."
}}
</belief>

<think>
[2-3 sentences: state the task requirements extracted from the task description, what the initial search strategy is, and why the selected first action is the most appropriate starting move.]
</think>

<action>
[Exact action string selected from admissible actions]
</action>"""


# ──────────────────────────────────────────────────────────────────────────────
# SFT Training  ·  Step >= 1  (with prior belief state, NO admissible_actions)
# ──────────────────────────────────────────────────────────────────────────────

WEBSHOP_REBEL_TEMPLATE_SFT = """You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is: {task_description}. You have already taken {step_count} steps before this turn. The most recent {history_length} actions are: {action_history}. You are now at step {current_step}, and your current observation is: {current_observation}.

The previous belief state is as follows:

<belief_prev>
{previous_belief}
</belief_prev>

You must first update your belief state based on the current observation, then reason from the updated belief state, and finally select the most appropriate and admissible action.

**Step 1: Update Belief State**

Integrate the previous belief state with the current observation to update the product goal, confirmed attributes, unconfirmed attributes, product identifier, and task phase. confirmed must only record information directly verified from observation, and all previously confirmed entries must be preserved.

**Step 2: Reason from Belief**

Based on the updated belief state, determine the current task phase, the current bottleneck, and the most critical attribute confirmation or product filtering step to advance. If information is insufficient or the subgoal is uncertain, prioritize actions that increase information certainty and reduce risk.

**Step 3: Action Selection**

Considering the current belief state and subgoal, select the one action most beneficial for advancing the task. If the subgoal is uncertain, prioritize actions with higher information gain and lower risk.

Now output strictly in the following format:

<belief>
{{
  "state": {{
    "target": {{"[attr]": "[value]"}},
    "confirmed": {{"[attr]": "[verified value]"}},
    "unconfirmed": ["[attr]"],
    "product_id": "[ASIN or null]"
  }},
  "task": {{
    "phase": "[searching | browsing | viewing | selecting | buying | done]",
    "target": "[product_id or null]",
    "subgoal": "[immediate one-sentence goal]"
  }},
  "prediction": "If I [planned action], I expect [consequence in next observation]."
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

WEBSHOP_REBEL_TEMPLATE_NO_HIS_SFT = """You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is: {task_description}. This is your first step. Your current observation is: {current_observation}.

You must construct your initial belief state from the task description and current observation, then reason from it, and finally select the most appropriate and admissible action.

**Step 1: Construct Initial Belief State**

Based on the task description, extract all target constraints (brand, color, price range, size, material, etc.) into state.target. Set confirmed to empty (nothing has been verified yet). Populate unconfirmed with all attributes that still need to be confirmed. Set product_id to null.

**Step 2: Reason from Belief**

Based on the initial belief state, determine the current task phase (should be "searching") and formulate the first subgoal: what to search for to find a matching product.

**Step 3: Action Selection**

Select the action most beneficial for initiating the task.

Now output strictly in the following format:

<belief>
{{
  "state": {{
    "target": {{"[attr]": "[value]"}},
    "confirmed": {{}},
    "unconfirmed": ["[attr]"],
    "product_id": null
  }},
  "task": {{
    "phase": "searching",
    "target": null,
    "subgoal": "[immediate one-sentence goal]"
  }},
  "prediction": "If I [planned action], I expect [consequence in next observation]."
}}
</belief>

<think>
[2-3 sentences: state the task requirements extracted from the task description, what the initial search strategy is, and why the selected first action is the most appropriate starting move.]
</think>

<action>
[Exact action string]
</action>"""


# ──────────────────────────────────────────────────────────────────────────────
# Annotation  ·  Teacher-LLM hindsight labeling
# ──────────────────────────────────────────────────────────────────────────────

WEBSHOP_REBEL_TAGGING_TEMPLATE = """You are an expert autonomous agent annotation specialist operating in the WebShop e-commerce environment. Your task is not to execute environment actions directly, but to reconstruct the agent's belief update, local reasoning, and action selection process at the current timestep, given the existing trajectory, current observation, and the previous belief state.

You must strictly adhere to World Model consistency: reason only based on information observable up to the current timestep, historical action outcomes, and confirmed facts. Do not use future information, knowledge outside the trajectory, or unverified assumptions. If the current observation conflicts with the previous belief state, trust the current observation and explicitly correct the old belief.

Your task is: {task_description}. You have already taken {step_count} steps before this turn. The most recent {history_length} actions are: {action_history}. You are now at step {current_step}, and your current observation is: {current_observation}.

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
5. Write null for any uncertain information in the belief; do not speculate.
6. confirmed must only record information directly verified from observation, and all previously confirmed entries must be preserved.
7. prediction must describe the most likely verifiable change in the next observation after executing the expert action.

====================
Belief Update Rules
====================

- target: Record the product attributes or filtering goals that the current task truly requires, such as brand, color, price range, size, material, etc.
- confirmed: Record attributes and values that have been directly verified from the current or historical page observations. Important notes on what CAN and CANNOT be confirmed:
  - DIRECTLY OBSERVABLE (confirm immediately when seen): price, color options, size options, product name, color/size selections.
  - NOT IN PAGE OBS (WebShop product pages do NOT show feature text such as material, sleeve type, care instructions, fit type, closure, outsole, etc. in the interactive obs): when the agent transitions to the 'selecting' phase having chosen a product, mark these unverifiable text attributes as "inferred: <value from task>" — this accurately represents that the agent selected the product based on search intent, not direct feature verification.
  - All previously confirmed entries MUST be preserved in every subsequent step. Never drop a confirmed key.
- unconfirmed: Record attributes that are currently uncertain and still need confirmation. Once an attribute is confirmed (directly or inferred), remove it from unconfirmed.
- product_id: Record the ASIN of the currently locked product; write null if not yet locked.
- If the current observation conflicts with the previous belief, trust the current observation and explicitly correct the old state.
- The belief should retain only the minimal necessary information useful for the current decision.

====================
Task Phase Determination
====================

phase must be exactly one of the following:

- searching: The agent is on the search entry page or has just issued a search query. Use this when the current observation shows the search bar / initial instruction page.
- browsing: The agent is viewing a list of search results (multiple products shown). Use this immediately after a search returns results.
- viewing: The agent has clicked into a single product detail page (ASIN). Transition to viewing the moment a click[ASIN] action is taken or the observation shows a single product page.
- selecting: The agent has confirmed all required attributes on the detail page and is deciding to proceed to purchase.
- buying: The agent has clicked "Buy Now" or is on the purchase confirmation page. Transition to buying immediately when gt_action is click[Buy Now].
- done: The purchase is complete or the task goal has been fully achieved.

Phase transition rules (apply strictly):
1. searching → browsing: triggered when search results appear (observation lists multiple products).
2. browsing → viewing: triggered immediately when the agent clicks a product ASIN (click[B0...]).  Set product_id to that ASIN.
3. viewing → buying: triggered immediately when gt_action is click[Buy Now]. Do NOT keep phase as viewing.
4. confirmed entries from previous steps must ALL be preserved — never drop a previously confirmed attribute.
5. Once product_id is set to an ASIN, it must remain set for all subsequent steps unless explicitly changed to a different ASIN.

====================
Output Format
====================

Output strictly in the following format:

<belief>
{{
  "state": {{
    "target": {{"[attr]": "[value]"}},
    "confirmed": {{"[attr]": "[verified value]"}},
    "unconfirmed": ["[attr]"],
    "product_id": "[ASIN or null]"
  }},
  "task": {{
    "phase": "[searching | browsing | viewing | selecting | buying | done]",
    "target": "[product_id or null]",
    "subgoal": "[immediate one-sentence goal]"
  }},
  "prediction": "If I [expert_action], I expect [next-frame verifiable observation change]."
}}
</belief>

<think>
[2-3 sentences: state what has been confirmed, what is still missing, what task phase is currently active, and why the expert action is justified. Do not repeat the full history; do not introduce additional speculation.]
</think>"""


# ──────────────────────────────────────────────────────────────────────────────
# Template selector
# ──────────────────────────────────────────────────────────────────────────────

def get_webshop_prompt_template(has_history: bool = True, sft: bool = False) -> str:
    """
    Return the appropriate prompt template for WebShop.

    Args:
        has_history: True for step >= 1 (prior belief available); False for step 0.
        sft:         True → return SFT template (no admissible_actions placeholder).
                     False (default) → return RL template (includes admissible_actions).

    Returns:
        Prompt template string.
    """
    if sft:
        return WEBSHOP_REBEL_TEMPLATE_SFT if has_history else WEBSHOP_REBEL_TEMPLATE_NO_HIS_SFT
    return WEBSHOP_REBEL_TEMPLATE_RL if has_history else WEBSHOP_REBEL_TEMPLATE_NO_HIS_RL
