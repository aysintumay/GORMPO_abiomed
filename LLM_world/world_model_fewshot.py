"""
Few-shot LLM world model — extends LLMWorldModel with nearest-neighbor retrieval.

Usage (drop-in replacement for LLMWorldModel):
    from LLM_world.world_model_fewshot import FewShotLLMWorldModel
    model = FewShotLLMWorldModel(model_id=..., device=..., k_shot=3)
    model.load_data(DATA_PATH)

At predict() time, the k nearest training trajectories (by L2 distance on the last
normalized context state) are prepended as alternating user/assistant turns before
the real query. This is more natural for chat models than embedding examples inside
a single user message.

Set k_shot=0 to run zero-shot (identical to LLMWorldModel).
"""

import numpy as np
import torch

from LLM_world.world_model import LLMWorldModel


class FewShotLLMWorldModel(LLMWorldModel):

    def __init__(self, k_shot: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k_shot = k_shot

        # retrieval index — populated in load_data()
        self._train_last_norm = None   # (N, 12) last context state, normalized
        self._train_ctx_norm  = None   # (N, T, 12) full context, normalized
        self._train_fc_norm   = None   # (N, H, 11) forecast without PL, normalized
        self._train_pl_int    = None   # (N,) P-level int

    # ── data loading ──────────────────────────────────────────────────────────

    def load_data(self, path: str):
        super().load_data(path)
        if self.k_shot > 0:
            self._build_retrieval_index()

    def _build_retrieval_index(self):
        """Iterate training set once to build arrays for fast L2 retrieval."""
        n = len(self.data_train)
        T = self.forecast_horizon
        print(f"[FewShot] Building retrieval index over {n} training samples ...")

        last_norm = np.empty((n, self.num_features), dtype=np.float32)
        ctx_norm  = np.empty((n, T, self.num_features), dtype=np.float32)
        fc_norm   = np.empty((n, T, self.num_features - 1), dtype=np.float32)
        pl_int    = np.empty(n, dtype=np.int32)

        for i in range(n):
            xi, pli, yi = self.data_train[i]
            xi_np = np.asarray(xi, dtype=np.float32)
            yi_np = np.asarray(yi, dtype=np.float32).reshape(T, self.num_features - 1)

            pl_raw  = float(np.asarray(pli).mean())
            pl_phys = pl_raw * float(self.std[-1]) + float(self.mean[-1])

            last_norm[i] = xi_np[-1]
            ctx_norm[i]  = xi_np
            fc_norm[i]   = yi_np
            pl_int[i]    = int(round(pl_phys))

        self._train_last_norm = last_norm
        self._train_ctx_norm  = ctx_norm
        self._train_fc_norm   = fc_norm
        self._train_pl_int    = pl_int
        print(f"[FewShot] Retrieval index ready ({n} examples).")

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _retrieve_examples(self, query_last_norm: np.ndarray, k: int) -> list:
        """
        k nearest training examples by L2 distance on last context state (normalized).
        Returns list of (ctx_phys, pl_int, fc_phys) tuples, closest first.
          ctx_phys : (T, 12) physical
          pl_int   : int P-level
          fc_phys  : (H, 12) physical — ground-truth forecast with PL column appended
        """
        diffs = self._train_last_norm - query_last_norm[None, :]   # (N, 12)
        dists = (diffs ** 2).sum(axis=1)                            # (N,)
        k_eff = min(k, len(dists))
        idxs  = np.argpartition(dists, k_eff)[:k_eff]
        idxs  = idxs[np.argsort(dists[idxs])]

        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()

        examples = []
        for idx in idxs:
            ctx_phys   = self._train_ctx_norm[idx] * s + m                  # (T, 12)
            pl_i       = int(self._train_pl_int[idx])
            fc_phys_11 = self._train_fc_norm[idx] * s[:-1] + m[:-1]        # (H, 11)
            pl_col     = np.full((self.forecast_horizon, 1), float(pl_i))
            fc_phys    = np.concatenate([fc_phys_11, pl_col], axis=1)       # (H, 12)
            examples.append((ctx_phys, pl_i, fc_phys))
        return examples

    # ── prompt helpers ────────────────────────────────────────────────────────

    def _format_query(self, ctx_phys: np.ndarray, p_level_int: int, include_answer_hint: bool = True) -> str:
        """Context + action formatted for a single turn (few-shot or real query)."""
        ctx_bins = self._to_bins(ctx_phys)
        rows     = "\n".join(" ".join(str(v) for v in row) for row in ctx_bins)
        pl_bin   = int(self._to_bins(np.array([[0.0] * 11 + [float(p_level_int)]]))[0, -1])

        msg = (
            f"## Context ({len(ctx_bins)} timesteps, oldest first):\n"
            f"{rows}\n\n"
            f"## Action: new Pump Level = P{p_level_int} (bin {pl_bin})\n\n"
            f"## Predict next {self.forecast_horizon} timesteps as a JSON array of arrays:"
        )
        if include_answer_hint:
            hint = "\n[\n" + ",\n".join(
                "  [" + ", ".join("<int>" for _ in range(self.num_features)) + "]"
                for _ in range(self.forecast_horizon)
            ) + "\n]"
            msg += hint
        return msg

    def _format_answer(self, fc_phys: np.ndarray, p_level_int: int) -> str:
        """Ground-truth forecast as a JSON bin array for the assistant turn."""
        fc_bins = self._to_bins(fc_phys).copy()
        pl_bin  = int(self._to_bins(np.array([[0.0] * 11 + [float(p_level_int)]]))[0, -1])
        fc_bins[:, -1] = pl_bin
        return "[\n" + ",\n".join(
            "  [" + ", ".join(str(v) for v in row) + "]"
            for row in fc_bins
        ) + "\n]"

    def _build_few_shot_messages(self, query_last_norm: np.ndarray) -> list:
        """Return list of (user_str, assistant_str) pairs for the k nearest examples."""
        examples = self._retrieve_examples(query_last_norm, self.k_shot)
        return [
            (
                self._format_query(ctx_ex, pl_ex, include_answer_hint=False),
                self._format_answer(fc_ex, pl_ex),
            )
            for ctx_ex, pl_ex, fc_ex in examples
        ]

    # ── LLM inference (override to support few-shot message history) ─────────

    def _run_llm(self, query_msg: str, few_shot_messages: list | None = None, n: int = 1) -> list:
        """
        Generate n completions. Samples are drawn one at a time to keep KV-cache
        memory bounded regardless of prompt length (important for large k_shot).
        few_shot_messages: list of (user_str, assistant_str) pairs prepended as
                           alternating turns before the real query.
        """
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        if few_shot_messages:
            for user_str, asst_str in few_shot_messages:
                messages.append({"role": "user",      "content": user_str})
                messages.append({"role": "assistant", "content": asst_str})
        messages.append({"role": "user", "content": query_msg})

        encoded = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids.to(self.llm.device)
        else:
            input_ids = encoded.to(self.llm.device)

        results = []
        with torch.no_grad():
            for _ in range(n):
                out = self.llm.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=self.temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                results.append(
                    self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                )
        return results

    # ── public API (overrides) ────────────────────────────────────────────────

    def _build_user_prompt(self, ctx_physical: np.ndarray, p_level_int: int) -> str:
        """Override: few-shot examples are injected via _run_llm, not here."""
        return self._format_query(ctx_physical, p_level_int, include_answer_hint=True)

    def predict(self, context_norm: torch.Tensor, p_level_int: int):
        ctx_physical = self._unnorm_context(context_norm)
        query_msg    = self._format_query(ctx_physical, p_level_int, include_answer_hint=True)

        few_shot_msgs = None
        if self.k_shot > 0 and self._train_last_norm is not None:
            query_last    = context_norm[-1].cpu().numpy()
            few_shot_msgs = self._build_few_shot_messages(query_last)

        raws    = self._run_llm(query_msg, few_shot_messages=few_shot_msgs, n=self.num_samples)
        samples = [p for raw in raws if (p := self._parse_response(raw, p_level_int)) is not None]

        if not samples:
            print(f"[FewShotLLM] All {self.num_samples} responses unparseable; using fallback.")
            samples = [self._fallback(context_norm, p_level_int)]

        arr       = np.stack(samples)
        mean_phys = arr.mean(axis=0)
        std_phys  = arr.std(axis=0)

        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()

        mean_norm = torch.tensor((mean_phys - m) / s, dtype=torch.float32)
        std_norm  = torch.tensor(std_phys / s,         dtype=torch.float32)
        return mean_norm, std_norm

    def sample_multiple(self, x: torch.Tensor, pl, num_samples: int = 10) -> torch.Tensor:
        if isinstance(pl, int):
            p_level_int = pl
        else:
            p_level_int = int(round(self.unnorm_pl(
                pl.float().mean() if isinstance(pl, torch.Tensor) else torch.tensor(pl)
            ).item()))

        ctx_physical  = self._unnorm_context(x.squeeze(0))
        query_msg     = self._format_query(ctx_physical, p_level_int, include_answer_hint=True)

        few_shot_msgs = None
        if self.k_shot > 0 and self._train_last_norm is not None:
            query_last    = x.squeeze(0)[-1].cpu().numpy()
            few_shot_msgs = self._build_few_shot_messages(query_last)

        m = self.mean[self.columns].detach().cpu().numpy()
        s = self.std[self.columns].detach().cpu().numpy()

        raws    = self._run_llm(query_msg, few_shot_messages=few_shot_msgs, n=num_samples)
        samples = []
        for raw in raws:
            pred = self._parse_response(raw, p_level_int)
            if pred is not None:
                norm = torch.tensor((pred - m) / s, dtype=torch.float32)
                samples.append(norm.unsqueeze(0))

        if not samples:
            fallback = self._fallback(x.squeeze(0), p_level_int)
            norm     = torch.tensor((fallback - m) / s, dtype=torch.float32)
            samples  = [norm.unsqueeze(0)] * num_samples

        return torch.stack(samples)
