​If we fine-tune one of Meta's open-weights models (e.g., Muse Glimmer) via LoRA/SFT and serve it in our app, does that fully count towards the Meta model usage requirement?

​Is a distillation pipeline (using Muse Spark via Meta Model API to generate synthetic training data \rightarrow SFT onto Muse Glimmer) considered an encouraged architecture?

If so, ​are there any official Meta-recommended frameworks (e.g., TorchTitan/Unsloth) or hosting partner stacks (Modal/Baseten/Fireworks) you suggest we follow?

Fine-tuning Muse Glimmer is allowed and counts as using a Meta model, but judges cannot assess the fine-tuned model itself, so the use case matters more. Post-training docs: https://dev.meta.ai/docs/muse-glimmer/post-training
Using Muse Spark to generate data for SFT on Glimmer is a fine architecture; if a Meta model is only used to train or build your system, ideally demo it calling a live Meta model too, since Use of Meta AI Tooling is scored.