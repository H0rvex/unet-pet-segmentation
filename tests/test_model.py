import torch

from unet_pet_seg.model import DecoderBlock, EncoderBlock, UNet

_EXPECTED_PARAMS = 7_703_107


def test_encoder_block_output_shapes():
    block = EncoderBlock(3, 64)
    x = torch.randn(2, 3, 128, 128)
    skip, pooled = block(x)
    assert skip.shape == (2, 64, 128, 128)
    assert pooled.shape == (2, 64, 64, 64)


def test_decoder_block_output_shape():
    block = DecoderBlock(512, 256)
    x = torch.randn(2, 512, 16, 16)
    skip = torch.randn(2, 256, 32, 32)
    assert block(x, skip).shape == (2, 256, 32, 32)


def test_unet_output_shape():
    model = UNet(num_classes=3)
    model.eval()
    x = torch.randn(4, 3, 128, 128)
    assert model(x).shape == (4, 3, 128, 128)


def test_unet_output_shape_batch1():
    model = UNet(num_classes=3)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    assert model(x).shape == (1, 3, 128, 128)


def test_unet_output_shape_256px():
    model = UNet(num_classes=3)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    assert model(x).shape == (2, 3, 256, 256)


def test_param_count():
    model = UNet(num_classes=3)
    params = sum(p.numel() for p in model.parameters())
    assert abs(params - _EXPECTED_PARAMS) / _EXPECTED_PARAMS < 0.05, (
        f"Param count {params} deviates >5% from expected {_EXPECTED_PARAMS}"
    )


def test_gradient_flows_to_all_parameters():
    model = UNet(num_classes=3)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    loss = model(x).sum()
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"No gradient for: {no_grad}"


def test_output_has_no_nan():
    model = UNet(num_classes=3)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    assert not torch.isnan(model(x)).any()
