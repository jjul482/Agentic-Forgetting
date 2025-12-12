import os
from typing import Tuple
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


DEFAULT_VLM_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

def make_atari_env(env_id: str = "ALE/Breakout-v5") -> gym.Env:
    """
    Create a Gymnasium Atari environment that returns RGB frames
    via env.render().
    """
    env = gym.make(env_id, render_mode="rgb_array")
    return env

def get_gym_frame(env: gym.Env, obs: np.ndarray | None = None) -> np.ndarray:
    """
    Standard way to get a pixel frame from a Gymnasium Atari env.

    - If render_mode="rgb_array" was set, env.render() returns (H,W,3) uint8.
    - We ignore `obs` here and just rely on render() to keep it simple.
    """
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. Make sure the env was created with render_mode='rgb_array'."
        )
    return frame

def atari_frame_to_pil(frame: np.ndarray) -> Image.Image:
    """
    Convert an Atari frame (HWC, CHW, or grayscale) into a 3-channel PIL.Image.
    Expected value range: 0–255 (uint8) or 0–1 (float).
    """
    if frame.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {frame.shape}")

    # If CHW (C,H,W) with C in {1,3}, convert to HWC
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] < frame.shape[-1]:
        frame = np.transpose(frame, (1, 2, 0))

    # If grayscale, expand to 3 channels
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)

    # Normalize dtype
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0)
            frame = (frame * 255).astype("uint8")
        else:
            frame = frame.astype("uint8")

    return Image.fromarray(frame)


class AtariVLM:
    """
    Thin wrapper around a Hugging Face VLM (LLaVA v1.6 Mistral) that:
      - downloads the model snapshot (on first use)
      - runs inference on (task_prompt, atari_frame)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VLM_ID,
        device: str = None,
        cache_dir: str = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        model_id : HF repo id of the VLM.
        device   : "cuda", "cpu", or None for auto.
        cache_dir: Optional local path to store the snapshot.
        dtype    : torch dtype for the model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model_id = model_id

        print(f"[AtariVLM] Loading VLM '{model_id}' on {self.device}...")

        self.processor = LlavaNextProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        ).to(self.device)

        self.model.eval()
        print("[AtariVLM] Loaded.")

    @torch.inference_mode()
    def describe_state(
        self,
        task_prompt: str,
        atari_frame: np.ndarray,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """
        Run the VLM on a single Atari frame and task prompt.

        task_prompt: e.g. "You are an RL assistant. Describe threats and goals for the agent."
        atari_frame: numpy array representing the game screen.
        """
        image = atari_frame_to_pil(atari_frame)

        # LLaVA v1.6 expects a specific prompt format; simplest is an [INST] template
        prompt = f"[INST] <image>\n{task_prompt} [/INST]"

        inputs = self.processor(
            image=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
        )

        # Decode full sequence and strip the prompt part if desired
        full_text = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        return full_text

    @torch.inference_mode()
    def snapshot_hidden(
        self,
        task_prompt: str,
        atari_frame: np.ndarray,
    ) -> Tuple[torch.Tensor, str]:
        """
        Optional: take a 'snapshot' of internal hidden states for representation learning.

        Returns:
           (last_hidden_state, decoded_text)
        """
        image = atari_frame_to_pil(atari_frame)
        prompt = f"[INST] <image>\n{task_prompt} [/INST]"

        inputs = self.processor(
            image=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, dim)
        # Optionally also get a short generation:
        gen_ids = self.model.generate(**inputs, max_new_tokens=32)
        text = self.processor.decode(gen_ids[0], skip_special_tokens=True)

        return last_hidden.detach().cpu(), text

def show_frame_blocking(frame: np.ndarray, title: str = "Atari Frame"):
    plt.figure(figsize=(4, 4))
    plt.imshow(frame)
    plt.axis("off")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # --- Example usage with a dummy Atari frame ---
    gym.register_envs(ale_py)
    env = make_atari_env("ALE/Freeway-v5")
    obs, info = env.reset()

    vlm = AtariVLM()

    task = (
        "You see an Atari game screen. "
        "Describe the current situation, the player's objective, "
        "and which objects look dangerous for the agent."
    )

    done = False
    truncated = False

    while not (done or truncated):
        # 3) Get pixel frame from Gymnasium
        frame = get_gym_frame(env, obs)

        # --- NEW: show the frame and block until user closes the window ---
        show_frame_blocking(frame, title="Current Atari State")

        # 4) Ask VLM about this state
        description = vlm.describe_state(task, frame)
        print("\nVLM description:\n", description)

        # 5) Take a random action just for the example
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)


    env.close()