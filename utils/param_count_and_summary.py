from torchsummary import summary

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_summary(model, input_shape, device='cpu'):
    summary(model, input_size=input_shape)
    print(f"Trainable Parameters: {count_params(model)}")
