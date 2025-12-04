
import unittest
import torch
from loofyloo_prime.model import LoofylooPrime
from loofyloo_prime.core.multimodal import MultimodalFoundation, CrossAttention
from loofyloo_prime.core.moe import MoELayer

class TestModel(unittest.TestCase):
    def test_multimodal_foundation(self):
        vocab_size = 1000
        embed_dim = 128
        foundation = MultimodalFoundation(vocab_size, embed_dim)
        text_input = torch.randint(0, vocab_size, (1, 10))
        image_input = torch.randn(1, 2048)
        audio_input = torch.randn(1, 1024)
        output = foundation(text_input, image_input, audio_input)
        self.assertEqual(output.shape, (1, 10, embed_dim))

    def test_cross_attention(self):
        embed_dim = 128
        cross_attention = CrossAttention(embed_dim)
        text_features = torch.randn(1, 10, embed_dim)
        other_features = torch.randn(1, 1, embed_dim)
        output = cross_attention(text_features, other_features)
        self.assertEqual(output.shape, (1, 10, embed_dim))

    def test_moe_layer(self):
        input_dim = 128
        output_dim = 128
        num_experts = 4
        moe_layer = MoELayer(input_dim, output_dim, num_experts)
        dummy_input = torch.randn(1, 10, input_dim)
        output = moe_layer(dummy_input)
        self.assertEqual(output.shape, (1, 10, output_dim))

    def test_loofyloo_prime(self):
        vocab_size = 1000
        embed_dim = 128
        num_experts = 4
        model = LoofylooPrime(vocab_size, embed_dim, num_experts)
        text_input = torch.randint(0, vocab_size, (1, 10))
        image_input = torch.randn(1, 2048)
        audio_input = torch.randn(1, 1024)
        output = model(text_input, image_input, audio_input)
        self.assertEqual(output.shape, (1, 10, embed_dim))

if __name__ == '__main__':
    unittest.main()
