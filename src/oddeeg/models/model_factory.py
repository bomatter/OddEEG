import importlib
import torch


def create_model(
    model_name,
    dataset_name,
    task: str | None = None,
    training_mode: str = "discriminative",
    n_times: int | None = None,
    checkpoint: str = None,
    device: str = "cpu",
    compile: bool = False,  # whether to use torch.compile
    **model_kwargs,  # forwarded to the model module's create_model
):
    if training_mode == "discriminative" and task is None:
        raise ValueError("Task must be specified for discriminative models.")
    
    try:
        model_module = importlib.import_module(f"oddeeg.models.{model_name.lower()}")
        model = model_module.create_model(
            dataset_name=dataset_name,
            task=task,
            n_times=n_times,
            training_mode=training_mode,
            **model_kwargs,
        )
    except ModuleNotFoundError:
        raise ValueError(f"Model '{model_name}' is not available. Add a module in oddeeg/models/")

    # Move model to device
    model.to(device)

    # Restore checkpoint if provided
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        # Remove "_orig_mod" prefix, which is added when a torch.compile-wrapped model is saved
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)

    # Compile model if requested
    if compile:
        model = torch.compile(model)

    return model