# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Optional
import re


def match_action_to_admissible(action: str, admissible_actions: List[str]) -> Tuple[str, bool]:
    """
    Intelligent action matching to handle ambiguous prepositions and minor variations.
    Handles "put X in/on Y" ambiguity, extra whitespace, and fuzzy prefix matching.
    """
    if not admissible_actions:
        return action, False

    if action in admissible_actions:
        return action, False

    if 'in/on' in action or 'on/in' in action:
        action_with_in = action.replace('in/on', 'in').replace('on/in', 'in')
        if action_with_in in admissible_actions:
            return action_with_in, True
        action_with_on = action.replace('in/on', 'on').replace('on/in', 'on')
        if action_with_on in admissible_actions:
            return action_with_on, True

    normalized_action = ' '.join(action.split())
    if normalized_action in admissible_actions:
        return normalized_action, True

    action_prefix = action.split()[0] if action.split() else ""
    if action_prefix:
        candidates = [a for a in admissible_actions if a.startswith(action_prefix)]
        if len(candidates) == 1:
            return candidates[0], True
        for candidate in candidates:
            if action in candidate or candidate in action:
                return candidate, True

    return action, False


def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids


def alfworld_projection_rebel(actions: List[str], action_pools: List[List[str]]) -> Tuple[List[str], List[int], List[str], List[bool]]:
    """
    ReBel projection for <belief>/<action> format. Validates belief JSON structure,
    action admissibility, and ordering. Returns 4-tuple for use with use_rebel=True.
    """
    import json

    actions_out = []
    valids = []
    beliefs = []
    action_available = [False] * len(actions)

    for i, output in enumerate(actions):
        valid = 1
        act_str = ""
        belief_text = ""

        if re.search(r'[\u4e00-\u9fff]', output):
            valid = 0

        belief_matches = re.findall(r"<belief>(.*?)</belief>", output, re.DOTALL | re.IGNORECASE)
        if len(belief_matches) != 1:
            valid = 0
        else:
            belief_text = belief_matches[0].strip()
            try:
                belief_json = belief_text.replace("'", '"')
                belief_data = json.loads(belief_json)
                has_world_model = 'world_model_update' in belief_data
                has_task_progress = 'task_progress_update' in belief_data
                has_exploration = 'exploration_map_update' in belief_data
                has_new_schema = 'state' in belief_data or 'task' in belief_data
                has_old_format = (
                    re.search(r'M_t:', belief_text, re.IGNORECASE) and
                    re.search(r'P_t:', belief_text, re.IGNORECASE) and
                    re.search(r'E_t:', belief_text, re.IGNORECASE)
                )
                if not (has_world_model or has_task_progress or has_exploration
                        or has_old_format or has_new_schema):
                    valid = 0
            except json.JSONDecodeError:
                if not (re.search(r'M_t:', belief_text, re.IGNORECASE) and
                        re.search(r'P_t:', belief_text, re.IGNORECASE) and
                        re.search(r'E_t:', belief_text, re.IGNORECASE)):
                    valid = 0
            if not belief_text or len(belief_text) < 10:
                valid = 0

        action_matches = re.findall(r"<action>([\s\S]*?)</action>", output, re.IGNORECASE)
        if len(action_matches) != 1:
            valid = 0
        else:
            act_str = action_matches[0].strip().lower()
            matched_action, _ = match_action_to_admissible(act_str, action_pools[i])
            act_str = matched_action
            if act_str in action_pools[i]:
                action_available[i] = True

        belief_pos = output.lower().find("<belief>")
        action_pos = output.lower().find("<action>")
        if belief_pos == -1 or action_pos == -1 or belief_pos > action_pos:
            valid = 0

        if not act_str:
            act_str = output.lower()[-30:]

        actions_out.append(act_str)
        valids.append(valid)
        beliefs.append(belief_text if valid else "")

    return actions_out, valids, beliefs, action_available
