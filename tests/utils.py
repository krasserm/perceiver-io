import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_equal_size(source_model, target_model, expected_size):
    assert count_parameters(source_model) == expected_size
    assert count_parameters(target_model) == expected_size


def random_input(b=2, n=6):
    prompt = torch.randint(6, 262, size=(b, n))
    mask = torch.ones_like(prompt, dtype=torch.int64)
    return {"input_ids": prompt, "attention_mask": mask}
