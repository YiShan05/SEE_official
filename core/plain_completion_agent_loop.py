from __future__ import annotations

from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput


SELF_EVAL_END_MARKER = "[/SELF_EVAL]"


class PlainCompletionAgentLoop(AgentLoopBase):
    """Single-turn rollout for base-model plain completion prompts.

    verl's default single_turn_agent always applies a chat template to
    raw_prompt. That is correct for instruct/chat data but wrong for this
    core dataset, whose prompt column is already the exact completion
    prefix to feed to the base model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    def _prompt_to_text(self, raw_prompt: Any) -> str:
        if not isinstance(raw_prompt, str) or not raw_prompt:
            raise ValueError(
                "core expects raw_prompt to be the exact non-empty plain completion prompt string; "
                f"got {type(raw_prompt).__name__}"
            )
        return raw_prompt

    def _sampling_params_with_self_eval_stop(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        params = dict(sampling_params)
        stop = params.get("stop")
        if stop is None:
            params["stop"] = [SELF_EVAL_END_MARKER]
        elif isinstance(stop, str):
            if not stop:
                raise ValueError("sampling_params.stop must not be an empty string")
            params["stop"] = [stop] if stop == SELF_EVAL_END_MARKER else [stop, SELF_EVAL_END_MARKER]
        elif isinstance(stop, (list, tuple)):
            stops = list(stop)
            if not stops or not all(isinstance(item, str) and item for item in stops):
                raise ValueError(f"sampling_params.stop must be a non-empty string list, got {stop!r}")
            if SELF_EVAL_END_MARKER not in stops:
                stops.append(SELF_EVAL_END_MARKER)
            params["stop"] = stops
        else:
            raise ValueError(f"sampling_params.stop must be a string or string list, got {type(stop).__name__}")

        # Keep the closing marker in the decoded response; the format reward
        # expects a complete [SELF_EVAL]...[/SELF_EVAL] block.
        params["include_stop_str_in_output"] = True
        return params

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        prompt_text = self._prompt_to_text(kwargs["raw_prompt"])
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_ids) > self.prompt_length:
            raise ValueError(
                f"Prompt length {len(prompt_ids)} exceeds configured prompt_length={self.prompt_length}; "
                "refusing to truncate training data."
            )

        rollout_sampling_params = self._sampling_params_with_self_eval_stop(sampling_params)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=rollout_sampling_params,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_mask = [1] * len(output.token_ids)
        agent_output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={},
            num_turns=1,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        agent_output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return agent_output
