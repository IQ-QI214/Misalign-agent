"""Same completion tokens, different teacher context; no reference-answer leakage."""

import torch


def feedback_for(row, mode):
    if "teacher_feedback" in row:
        from .data import select_feedback
        return select_feedback(row, mode)
    if mode != "neutral":
        raise ValueError("Structured feedback is required for research feedback modes")
    return row.get("teacher_context", "Excellent")


def render_prompt(tokenizer, prompt, *, feedback=None, thinking=False, placement="user", reference=None):
    # This fixed-prefix context is an explicit adaptation, not the original paper's
    # answer-dependent hindsight template. Gold answers are never read here.
    if placement not in ("user", "assistant_prefill"):
        raise ValueError("Unknown privileged context placement")
    privileged = feedback
    if reference is not None:
        privileged = f"[Verified reference]\n{reference}\n\n{feedback}\nSolve the original task using your own reasoning."
    user = prompt if privileged is None or placement == "assistant_prefill" else f"{prompt}\n\n[Teacher-only feedback]\n{privileged}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking,
    )
    return rendered + (privileged + "\n\n" if privileged is not None and placement == "assistant_prefill" else "")


class TransferCollator:
    def __init__(self, tokenizer, max_prompt_length=1024, feedback_mode="neutral", thinking=False,
                 context_mode="feedback_only", placement="user"):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.feedback_mode = feedback_mode
        self.thinking = thinking
        if context_mode not in ("feedback_only", "verified_reference"):
            raise ValueError("Unknown context mode")
        self.context_mode = context_mode
        self.placement = placement

    def __call__(self, rows):
        # Microbatch 1 avoids unequal prompt padding/position behavior in the
        # hybrid recurrent architecture. Distributed global batch is recorded.
        if len(rows) != 1:
            raise ValueError("v1 supports per-device microbatch 1 only")
        row = rows[0]
        cue = feedback_for(row, self.feedback_mode)
        reference = None
        if self.context_mode == "verified_reference":
            reference = row.get("verified_solution")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("verified_reference requires an explicit verified_solution, never an implicit eval answer")
        result = {"record_ids": [row["id"]], "source_ids": [row["source_id"]], "feedback": [cue]}
        for role, feedback in [("student", None), ("teacher", cue)]:
            text = render_prompt(self.tokenizer, row["prompt"], feedback=feedback, thinking=self.thinking,
                                 placement=self.placement if role == "teacher" else "user",
                                 reference=reference if role == "teacher" else None)
            encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > self.max_prompt_length:
                raise ValueError(f"{role} prompt too long: {row['id']}; silent truncation is disabled")
            result[f"{role}_prompts"] = encoded["input_ids"]
            result[f"{role}_prompt_attention_mask"] = encoded["attention_mask"]
            result[f"{role}_prompt_length"] = encoded["input_ids"].shape[1]
        return result


def completion_mask(ids, eos_token_id):
    """Include the first EOS, mask everything after it, even when pad == EOS."""
    eos_ids = eos_token_id if isinstance(eos_token_id, (tuple, list)) else [eos_token_id]
    eos = torch.zeros_like(ids, dtype=torch.bool)
    for token in eos_ids:
        if token is not None:
            eos |= ids.eq(token)
    return (eos.long().cumsum(-1) - eos.long()).eq(0)
