from typing import Any
import torch
from transformer_lens.model_bridge import TransformerBridge
from config import MODEL_NAME, RESID_POST_HOOK, DEFAULT_PREPEND_BOS, DEFAULT_TOP_K, SANITY_PROMPT, SANITY_EXPECTED_TOKEN

from dotenv import load_dotenv
load_dotenv()


def load_model(model_name: str = MODEL_NAME):
    """Load a TransformerLens bridge model ready for inference."""
    device = get_device()
    model = TransformerBridge.boot_transformers(model_name, device=device)
    model.enable_compatibility_mode()  # gives HookedTransformer-equivalent numerics

    model.eval()
    return model


def get_device() -> str:
    '''Get the device to use for the model.'''
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_resid_post_hook_name(layer: int) -> str:
    """Hook name for residual stream after block `layer`."""
    return RESID_POST_HOOK.format(layer=layer)


def get_tokens(
    model,
    prompt: str,
    prepend_bos: bool = DEFAULT_PREPEND_BOS,
):
    """Convert text to a token tensor using the BOS policy."""
    return model.to_tokens(prompt, prepend_bos=prepend_bos)


def topk_next_tokens(
    model,
    tokens: torch.Tensor,
    k: int = DEFAULT_TOP_K,
) -> tuple[list[str], list[float]]:
    """Return top-k next-token strings and logits for the final position."""
    with torch.inference_mode():
        logits = model(tokens)
        next_logits = logits[0, -1]
        top = torch.topk(next_logits, k)

    top_candidates = [model.to_string(tid.item()) for tid in top.indices]
    top_values = [float(v) for v in top.values]

    return top_candidates, top_values


def get_resid_cache(model, tokens: torch.Tensor) -> dict:
    """Run a forward pass and cache residual-stream (resid_post) activations."""
    def _is_resid_post(name: str) -> bool:
        return name.endswith("hook_resid_post")

    with torch.inference_mode():
        logits, cache = model.run_with_cache(tokens, names_filter=_is_resid_post)
    
    return cache


def final_token_activation(
    cache: dict,
    layer: int,
    batch_idx: int = 0,
    pos_idx: int = -1,
) -> torch.Tensor:
    """Residual-stream activation at one token position"""
    hook_name = get_resid_post_hook_name(layer)
    return cache[hook_name][batch_idx, pos_idx, :]


def check_next_token_prediction(model, prompt: str, expected: str) -> torch.Tensor:
    """check if expected token appears in top-k next-token predictions."""
    tokens = get_tokens(model, prompt)
    top_strs, _ = topk_next_tokens(model, tokens)

    print("check if expected token appears in top-k next-token predictions:")
    print(f"Prompt: {prompt}")
    print(f"Top {len(top_strs)} candidates: {top_strs}")

    assert any(expected in s for s in top_strs), (
        f"Expected {expected!r} among top predictions. Got: {top_strs}"
    )
    print(f"PASS: {expected!r} found among top predictions.")
    return tokens


def check_resid_activations(model, tokens: torch.Tensor):
    """check if resid_post cache exists at every layer with the right shape."""
    print("check if resid_post cache exists at every layer with the right shape:")
    cache = get_resid_cache(model, tokens)

    expected_shape = (1, tokens.shape[1], model.cfg.d_model)

    print(f"{'Layer':<8}{'Hook name':<30}{'Shape':<25}")
    for layer in range(model.cfg.n_layers):
        hook_name = get_resid_post_hook_name(layer)
        act = cache[hook_name]
        print(f"{layer:<8}{hook_name:<30}{str(tuple[Any, ...](act.shape)):<25}")
        assert tuple(act.shape) == expected_shape, (
            f"Layer {layer}: expected {expected_shape}, got {tuple(act.shape)}"
        )

    print(
        f"PASS: activations extracted at all {model.cfg.n_layers} layers, "
        f"shape matches expected {expected_shape}."
    )


def main() -> None:
    model = load_model()
    print(f"\nModel loaded: {model.cfg.model_name}")
    print(f"Number of layers: {model.cfg.n_layers}")
    print(f"Residual stream dimension (d_model): {model.cfg.d_model}")

    tokens = check_next_token_prediction(model, SANITY_PROMPT, SANITY_EXPECTED_TOKEN)
    check_resid_activations(model, tokens)


if __name__ == "__main__":
    main()