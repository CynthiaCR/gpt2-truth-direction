MODEL_NAME = "openai-community/gpt2"
RESID_POST_HOOK = "blocks.{layer}.hook_resid_post"
DEFAULT_PREPEND_BOS = False
DEFAULT_TOP_K = 5
SANITY_PROMPT = "The capital of France is"
SANITY_EXPECTED_TOKEN = "Paris"
HOOK_RESID_POST = "hook_resid_post"
