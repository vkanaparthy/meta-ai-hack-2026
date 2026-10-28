​If we fine-tune one of Meta's open-weights models (e.g., Muse Glimmer) via LoRA/SFT and serve it in our app, does that fully count towards the Meta model usage requirement?

​Is a distillation pipeline (using Muse Spark via Meta Model API to generate synthetic training data \rightarrow SFT onto Muse Glimmer) considered an encouraged architecture?

If so, ​are there any official Meta-recommended frameworks (e.g., TorchTitan/Unsloth) or hosting partner stacks (Modal/Baseten/Fireworks) you suggest we follow?