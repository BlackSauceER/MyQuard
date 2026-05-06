import math
import torch
from typing import List, Optional, Sequence, Union


class GaitPhaseGenerator:
    """
    Phase generator for gait-conditioned quadruped locomotion.

    Foot order:
        0: FR
        1: FL
        2: HR
        3: HL

    Output:
        phase_signal: [num_envs, 8]
            [sin_FR, cos_FR, sin_FL, cos_FL, sin_HR, cos_HR, sin_HL, cos_HL]

        swing_mask: [num_envs, 4]
            1.0 means expected swing phase
            0.0 means expected stance phase

        phi: [num_envs, 4]
            phase value of each leg, in [0, 2π]
    """

    GAIT_PARAMS = {
        "walk": {
            "phase_offsets": [0.0, 0.5 * math.pi, 1.5 * math.pi, math.pi],
            "swing_ratio": 0.25,
            "frequency": 0.8,
        },
        "trot": {
            "phase_offsets": [0.0, math.pi, math.pi, 0.0],
            "swing_ratio": 0.5,
            "frequency": 1.8,
        },
        "gallop": {
            "phase_offsets": [0.0, 1.6 * math.pi, 0.8 * math.pi, 0.4 * math.pi],
            "swing_ratio": 0.75,
            "frequency": 2.8,
        },
    }

    def __init__(
        self,
        num_envs: int,
        device: Union[str, torch.device],
        dt: float,
        gait: str,
        random_init_phase: bool = True,
    ):
        if gait not in self.GAIT_PARAMS:
            raise ValueError(f"Unknown gait type: {gait}")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = float(dt)
        self.gait = gait

        params = self.GAIT_PARAMS[gait]

        frequency = float(params["frequency"])
        period = 1.0 / frequency

        self.frequency = torch.full(
            (num_envs, 1),
            frequency,
            dtype=torch.float32,
            device=self.device,
        )

        self.period = torch.full(
            (num_envs, 1),
            period,
            dtype=torch.float32,
            device=self.device,
        )

        self.phase_offsets = torch.tensor(
            params["phase_offsets"],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 4).expand(num_envs, 4).clone()

        self.swing_ratio = torch.full(
            (num_envs, 4),
            float(params["swing_ratio"]),
            dtype=torch.float32,
            device=self.device,
        )

        if random_init_phase:
            self.phase_time = torch.rand(num_envs, device=self.device) * self.period.squeeze(-1)
        else:
            self.phase_time = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #

    def _normalize_env_ids(self, env_ids):
        if env_ids is None:
            return None
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return env_ids

    def _to_env_tensor(
        self,
        value,
        name: str,
        env_ids=None,
        shape_last: int = 4,
    ) -> torch.Tensor:
        """
        Convert scalar / list / tensor into [num_selected_envs, shape_last].

        Accepted shapes:
            scalar
            [1]
            [shape_last]
            [num_envs, shape_last]
            [len(env_ids), shape_last]
        """
        env_ids = self._normalize_env_ids(env_ids)

        if env_ids is None:
            n = self.num_envs
        else:
            n = len(env_ids)

        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float32, device=self.device)
        else:
            value = value.to(device=self.device, dtype=torch.float32)

        if value.dim() == 0:
            value = value.view(1, 1).expand(n, shape_last)

        elif value.dim() == 1:
            if value.numel() == 1:
                value = value.view(1, 1).expand(n, shape_last)
            elif value.numel() == shape_last:
                value = value.view(1, shape_last).expand(n, shape_last)
            elif value.numel() == n and shape_last == 1:
                value = value.view(n, 1)
            elif value.numel() == self.num_envs and shape_last == 1 and env_ids is not None:
                value = value[env_ids].view(n, 1)
            else:
                raise ValueError(
                    f"{name} has invalid shape {tuple(value.shape)}; "
                    f"expected scalar, [{shape_last}], [{n}], "
                    f"[{n}, {shape_last}], or [{self.num_envs}, {shape_last}]"
                )

        elif value.dim() == 2:
            if value.shape == (self.num_envs, shape_last) and env_ids is not None:
                value = value[env_ids]
            elif value.shape == (n, shape_last):
                pass
            elif value.shape == (1, shape_last):
                value = value.expand(n, shape_last)
            else:
                raise ValueError(
                    f"{name} has invalid shape {tuple(value.shape)}; "
                    f"expected [{n}, {shape_last}] or [{self.num_envs}, {shape_last}]"
                )

        else:
            raise ValueError(f"{name} has invalid dim {value.dim()}")

        return value

    def _get_template_offsets(self) -> torch.Tensor:
        """
        Returns:
            template_offsets: [3, 4]
                row 0: walk
                row 1: trot
                row 2: gallop
        """
        offsets = [
            self.GAIT_PARAMS["walk"]["phase_offsets"],
            self.GAIT_PARAMS["trot"]["phase_offsets"],
            self.GAIT_PARAMS["gallop"]["phase_offsets"],
        ]

        return torch.tensor(
            offsets,
            dtype=torch.float32,
            device=self.device,
        )  # [3, 4]

    def _interpolate_offsets_sincos(
        self,
        weights: torch.Tensor,
        template_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate phase offsets in sin/cos space to avoid angle wrapping jumps.

        Args:
            weights: [N, K], usually K=3 for walk/trot/gallop.
                     Each row should sum to 1.
            template_offsets: [K, 4], offsets of gait templates.

        Returns:
            phase_offsets: [N, 4], in [0, 2π)
        """
        weights = weights.to(device=self.device, dtype=torch.float32)

        if template_offsets is None:
            template_offsets = self._get_template_offsets()
        else:
            template_offsets = template_offsets.to(device=self.device, dtype=torch.float32)

        if weights.dim() != 2:
            raise ValueError(f"weights must be [N, K], got {tuple(weights.shape)}")

        if template_offsets.dim() != 2 or template_offsets.shape[1] != 4:
            raise ValueError(
                f"template_offsets must be [K, 4], got {tuple(template_offsets.shape)}"
            )

        if weights.shape[1] != template_offsets.shape[0]:
            raise ValueError(
                f"weights K={weights.shape[1]} does not match "
                f"template_offsets K={template_offsets.shape[0]}"
            )

        # Normalize weights for safety.
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)

        sin_templates = torch.sin(template_offsets)  # [K, 4]
        cos_templates = torch.cos(template_offsets)  # [K, 4]

        sin_blend = weights @ sin_templates  # [N, 4]
        cos_blend = weights @ cos_templates  # [N, 4]

        offsets = torch.atan2(sin_blend, cos_blend)
        offsets = torch.remainder(offsets, 2.0 * math.pi)

        return offsets

    # --------------------------------------------------------------------- #
    # Parameter update APIs
    # --------------------------------------------------------------------- #

    def set_gait(
        self,
        gait: str,
        env_ids=None,
        reset_phase: bool = False,
        preserve_phase: bool = True,
    ):
        """
        Set parameters from predefined walk/trot/gallop templates.

        Args:
            gait: "walk", "trot", or "gallop"
            env_ids: selected envs. If None, update all envs.
            reset_phase:
                If True, reset phase_time to 0 after updating params.
            preserve_phase:
                If True, preserve normalized phase when frequency / period changes.
        """
        if gait not in self.GAIT_PARAMS:
            raise ValueError(f"Unknown gait type: {gait}")

        params = self.GAIT_PARAMS[gait]

        frequency = torch.full(
            (1, 1),
            float(params["frequency"]),
            dtype=torch.float32,
            device=self.device,
        )

        swing_ratio = torch.full(
            (1, 4),
            float(params["swing_ratio"]),
            dtype=torch.float32,
            device=self.device,
        )

        phase_offsets = torch.tensor(
            params["phase_offsets"],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 4)

        self.set_params(
            frequency=frequency,
            swing_ratio=swing_ratio,
            phase_offsets=phase_offsets,
            env_ids=env_ids,
            reset_phase=reset_phase,
            preserve_phase=preserve_phase,
        )

        if env_ids is None:
            self.gait = gait

    def set_params(
        self,
        frequency=None,
        period=None,
        swing_ratio=None,
        phase_offsets=None,
        env_ids=None,
        reset_phase: bool = False,
        preserve_phase: bool = True,
    ):
        """
        Directly update physical phase-generator parameters.

        Args:
            frequency:
                scalar / [N] / [N, 1], unit Hz.
            period:
                scalar / [N] / [N, 1], unit second.
                Do not provide frequency and period at the same time.
            swing_ratio:
                scalar / [4] / [N, 4].
                1.0 means full swing, 0.0 means no swing.
                Internally clamped to [0.05, 0.95].
            phase_offsets:
                [4] / [N, 4], radians.
            env_ids:
                selected env indices.
            reset_phase:
                If True, set phase_time to 0 after updating.
            preserve_phase:
                If True, preserve normalized base phase t / T when period changes.
        """
        if frequency is not None and period is not None:
            raise ValueError("Only one of frequency or period can be provided.")

        env_ids = self._normalize_env_ids(env_ids)

        if env_ids is None:
            target = slice(None)
            n = self.num_envs
        else:
            target = env_ids
            n = len(env_ids)

        old_period = self.period[target].squeeze(-1).clone()
        old_phase_ratio = torch.remainder(
            self.phase_time[target] / torch.clamp(old_period, min=1e-6),
            1.0,
        )

        if frequency is not None:
            freq = self._to_env_tensor(
                frequency,
                name="frequency",
                env_ids=env_ids,
                shape_last=1,
            )
            freq = torch.clamp(freq, min=1e-4)
            self.frequency[target] = freq
            self.period[target] = 1.0 / freq

        if period is not None:
            T = self._to_env_tensor(
                period,
                name="period",
                env_ids=env_ids,
                shape_last=1,
            )
            T = torch.clamp(T, min=1e-4)
            self.period[target] = T
            self.frequency[target] = 1.0 / T

        if swing_ratio is not None:
            gamma = self._to_env_tensor(
                swing_ratio,
                name="swing_ratio",
                env_ids=env_ids,
                shape_last=4,
            )
            gamma = torch.clamp(gamma, min=0.05, max=0.95)
            self.swing_ratio[target] = gamma

        if phase_offsets is not None:
            offsets = self._to_env_tensor(
                phase_offsets,
                name="phase_offsets",
                env_ids=env_ids,
                shape_last=4,
            )
            offsets = torch.remainder(offsets, 2.0 * math.pi)
            self.phase_offsets[target] = offsets

        if reset_phase:
            self.phase_time[target] = 0.0
        elif preserve_phase:
            new_period = self.period[target].squeeze(-1)
            self.phase_time[target] = old_phase_ratio * new_period
        else:
            new_period = self.period[target].squeeze(-1)
            self.phase_time[target] = torch.remainder(
                self.phase_time[target],
                new_period,
            )

    def set_params_from_cmd(
        self,
        phase_cmd: torch.Tensor,
        env_ids=None,
        mode: str = "simple",
        reset_phase: bool = False,
        preserve_phase: bool = True,
        frequency_range: Sequence[float] = (0.6, 3.2),
        swing_ratio_range: Sequence[float] = (0.25, 0.75),
    ):
        """
        Map selector raw output to valid phase-generator parameters.

        Recommended first version:
            mode="simple"
            phase_cmd shape: [N, 3]
                raw_frequency
                raw_swing_ratio
                raw_gait_alpha

            alpha in [0, 1]:
                0.0 -> walk
                0.5 -> trot
                1.0 -> gallop

            phase_offsets are interpolated in sin/cos space.

        Alternative:
            mode="template_softmax"
            phase_cmd shape: [N, 5]
                raw_frequency
                raw_swing_ratio
                template_logits_walk
                template_logits_trot
                template_logits_gallop

            template weights = softmax(template_logits)
            phase_offsets are interpolated in sin/cos space.
        """
        env_ids = self._normalize_env_ids(env_ids)

        phase_cmd = phase_cmd.to(device=self.device, dtype=torch.float32)

        if env_ids is not None and phase_cmd.shape[0] == self.num_envs:
            phase_cmd = phase_cmd[env_ids]

        f_min, f_max = float(frequency_range[0]), float(frequency_range[1])
        gamma_min, gamma_max = float(swing_ratio_range[0]), float(swing_ratio_range[1])

        if f_min <= 0.0 or f_max <= f_min:
            raise ValueError(f"Invalid frequency_range: {frequency_range}")

        if not (0.0 < gamma_min < gamma_max < 1.0):
            raise ValueError(f"Invalid swing_ratio_range: {swing_ratio_range}")

        if mode == "simple":
            if phase_cmd.shape[-1] != 3:
                raise ValueError(
                    f"simple mode expects phase_cmd dim 3, "
                    f"got {phase_cmd.shape[-1]}"
                )

            raw_f = phase_cmd[:, 0:1]
            raw_gamma = phase_cmd[:, 1:2]
            raw_alpha = phase_cmd[:, 2:3]

            frequency = f_min + (f_max - f_min) * torch.sigmoid(raw_f)

            swing_ratio_scalar = gamma_min + (gamma_max - gamma_min) * torch.sigmoid(raw_gamma)
            swing_ratio = swing_ratio_scalar.expand(-1, 4)

            alpha = torch.sigmoid(raw_alpha)  # [N, 1], 0 walk, 0.5 trot, 1 gallop

            # Piecewise template weights:
            # alpha in [0, 0.5]: walk -> trot
            # alpha in [0.5, 1]: trot -> gallop
            walk_weight = torch.clamp(1.0 - 2.0 * alpha, min=0.0, max=1.0)
            trot_weight = 1.0 - torch.abs(2.0 * alpha - 1.0)
            trot_weight = torch.clamp(trot_weight, min=0.0, max=1.0)
            gallop_weight = torch.clamp(2.0 * alpha - 1.0, min=0.0, max=1.0)

            weights = torch.cat(
                [walk_weight, trot_weight, gallop_weight],
                dim=-1,
            )  # [N, 3]

            phase_offsets = self._interpolate_offsets_sincos(weights)

        elif mode == "template_softmax":
            if phase_cmd.shape[-1] != 5:
                raise ValueError(
                    f"template_softmax mode expects phase_cmd dim 5, "
                    f"got {phase_cmd.shape[-1]}"
                )

            raw_f = phase_cmd[:, 0:1]
            raw_gamma = phase_cmd[:, 1:2]
            template_logits = phase_cmd[:, 2:5]

            frequency = f_min + (f_max - f_min) * torch.sigmoid(raw_f)

            swing_ratio_scalar = gamma_min + (gamma_max - gamma_min) * torch.sigmoid(raw_gamma)
            swing_ratio = swing_ratio_scalar.expand(-1, 4)

            weights = torch.softmax(template_logits, dim=-1)
            phase_offsets = self._interpolate_offsets_sincos(weights)

        else:
            raise ValueError(f"Unknown phase command mode: {mode}")

        self.set_params(
            frequency=frequency,
            swing_ratio=swing_ratio,
            phase_offsets=phase_offsets,
            env_ids=env_ids,
            reset_phase=reset_phase,
            preserve_phase=preserve_phase,
        )

    # --------------------------------------------------------------------- #
    # Runtime APIs
    # --------------------------------------------------------------------- #

    def reset(self, env_ids: Union[List[int], torch.Tensor], random_init_phase: bool = True):
        env_ids = self._normalize_env_ids(env_ids)

        if env_ids is None or len(env_ids) == 0:
            return

        if random_init_phase:
            self.phase_time[env_ids] = (
                torch.rand(len(env_ids), device=self.device)
                * self.period[env_ids].squeeze(-1)
            )
        else:
            self.phase_time[env_ids] = 0.0

    def step(self, env_ids=None):
        """
        Advance phase time by dt.

        Args:
            env_ids:
                If None, step all envs.
                Otherwise, step selected envs only.
        """
        env_ids = self._normalize_env_ids(env_ids)

        if env_ids is None:
            self.phase_time = torch.remainder(
                self.phase_time + self.dt,
                self.period.squeeze(-1),
            )
        else:
            self.phase_time[env_ids] = torch.remainder(
                self.phase_time[env_ids] + self.dt,
                self.period[env_ids].squeeze(-1),
            )

    def get_phase(self):
        """
        Returns:
            phase_signal: [num_envs, 8]
            swing_mask: [num_envs, 4]
            phi: [num_envs, 4]
        """
        t = self.phase_time.view(-1, 1)  # [num_envs, 1]
        T = self.period                  # [num_envs, 1]
        gamma = self.swing_ratio         # [num_envs, 4]

        gamma = torch.clamp(gamma, min=0.05, max=0.95)

        offset_ratio = self.phase_offsets / (2.0 * math.pi)  # [num_envs, 4]

        # u is normalized phase in one period, with per-leg phase offset.
        u = torch.remainder(t / T + offset_ratio, 1.0)  # [num_envs, 4]

        # Piecewise phase mapping:
        # swing:  u in [0, gamma)      -> phi in [0, π)
        # stance: u in [gamma, 1)      -> phi in [π, 2π)
        phi_swing = math.pi * u / gamma
        phi_stance = math.pi + math.pi * (u - gamma) / (1.0 - gamma)

        phi = torch.where(u < gamma, phi_swing, phi_stance)

        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        swing_mask = (u < gamma).float()

        phase_signal = torch.stack(
            [
                sin_phi[:, 0], cos_phi[:, 0],
                sin_phi[:, 1], cos_phi[:, 1],
                sin_phi[:, 2], cos_phi[:, 2],
                sin_phi[:, 3], cos_phi[:, 3],
            ],
            dim=-1,
        )

        return phase_signal, swing_mask, phi

    def get_expected_contact(self):
        """
        Returns:
            expected_contact: [num_envs, 4]
                1.0 means expected stance/contact.
                0.0 means expected swing/no contact.
        """
        _, swing_mask, _ = self.get_phase()
        return 1.0 - swing_mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gait = "walk"
    dt = 0.02

    phase_gen = GaitPhaseGenerator(
        num_envs=1,
        device="cpu",
        dt=dt,
        gait=gait,
        random_init_phase=False,
    )

    # Example: selector raw command.
    # simple mode:
    #   raw_frequency, raw_swing_ratio, raw_gait_alpha
    # raw_gait_alpha:
    #   sigmoid(raw)=0.0 -> walk
    #   sigmoid(raw)=0.5 -> trot
    #   sigmoid(raw)=1.0 -> gallop
    #
    # raw_alpha=0 -> sigmoid=0.5 -> trot

    phase_fr, phase_fl, phase_hr, phase_hl = [], [], [], []

    period_x3 = 1.0 / GaitPhaseGenerator.GAIT_PARAMS[gait]['frequency'] * 3
    sample_num = int(period_x3 / dt)
    alpha = 0.0
    for i in range(sample_num):
        phase_signal, _, _ = phase_gen.get_phase()
        phase_gen.step()

        # phase_cmd = torch.tensor([[0.0,  0.0,  alpha]], dtype=torch.float32)
        # phase_gen.set_params_from_cmd(phase_cmd, mode="simple")

        phase_fr.append(phase_signal[0, 0].item())
        phase_fl.append(phase_signal[0, 2].item())
        phase_hr.append(phase_signal[0, 4].item())
        phase_hl.append(phase_signal[0, 6].item())

        alpha += 0.01

    plt.plot(range(sample_num), phase_fr, label="FR")
    plt.plot(range(sample_num), phase_fl, label="FL")
    plt.plot(range(sample_num), phase_hr, label="HR")
    plt.plot(range(sample_num), phase_hl, label="HL")
    plt.legend()
    plt.show()