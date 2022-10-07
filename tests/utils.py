def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_equal_size(source_model, target_model, expected_size):
    assert count_parameters(source_model) == expected_size
    assert count_parameters(target_model) == expected_size
