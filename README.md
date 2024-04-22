# MicroRWKV
This is a custom architecture for the nanoRWKV project from [here](https://github.com/BlinkDL/nanoRWKV). The architecture is based on the original nanoRWKV architecture, but with some modifications.

## Model Structure
RWKV_TimeMix -> RWKV_ChannelMix -> Sliding Window Attention -> GroupedQAttention -> TinyMoE

Here is a brief description of each component:
1. RWKV_TimeMix: This component applies a time-based mixing operation to the input, which helps the model capture temporal dependencies.
2. RWKV_ChannelMix: The channel-based mixing operation is performed in this module, allowing the model to learn better representations across different feature channels.
3. Sliding Window Attention: This attention mechanism operates on a sliding window of the input, enabling the model to efficiently capture local and global dependencies.
4. GroupedQAttention: This attention module applies a grouped approach to the query, key, and value computations, improving the model's ability to capture multi-headed attention.
5. TinyMoE: The Tiny Mixture of Experts (TinyMoE) layer is a lightweight and efficient implementation of a Mixture of Experts (MoE) mechanism, which can help the model learn specialized representations.

## Detailed Explanation
1. RWKV_TimeMix:
This module applies a time-based mixing operation to the input, which helps the model capture temporal dependencies.
It uses several learnable parameters, such as `time_maa_k`, `time_maa_v`, `time_maa_r`, and `time_maa_g`, to control the mixing process.
The module also applies a time-decay mechanism using the time_decay parameter, which allows the model to give more importance to recent inputs.
The output of this module is then passed through a series of linear layers, including the receptance, key, value, and gate layers.

2. RWKV_ChannelMix:
This module performs a channel-based mixing operation on the input, allowing the model to learn better representations across different feature channels.
It uses a time-shift operation and learnable parameters, such as `time_maa_k` and `time_maa_r`, to control the mixing process.
The module applies a key, value, and receptance linear layers to the mixed input, and the output is then passed through a sigmoid activation function.

3. Sliding Window Attention:
This attention mechanism operates on a sliding window of the input, enabling the model to efficiently capture both local and global dependencies.
The module computes the query, key, and value matrices using a linear layer, and then applies a sliding window attention operation to the input.
The output of the sliding window attention is then passed through a final linear layer to produce the final output.

4. GroupedQAttention:
This attention module applies a grouped approach to the query, key, and value computations, improving the model's ability to capture multi-headed attention.
The module first computes the query, key, value, and weight matrices using a single linear layer, and then splits these matrices into groups.
The attention computation is then performed on each group, and the results are concatenated and passed through a final linear layer.

5. TinyMoE:
The Tiny Mixture of Experts (TinyMoE) layer is a lightweight and efficient implementation of a Mixture of Experts (MoE) mechanism, which can help the model learn specialized representations.
The module computes attention scores using a linear layer, and then applies these scores to a set of expert networks to produce the final output.
The module also includes an auxiliary loss term that encourages the experts to learn diverse representations, improving the overall performance of the model.

## Usage (Inference)
To use this model for inference, you can follow these steps:
1. Download and paste model weights in the `out` directory.
2. Copy and paste the values like: `block_size`, `vocab_size`, etc from the table into the class GPTConfig in `generate.py`.
3. Then run the following command:
```python
python generate.py --prompt="One day" --max_num_tokens=50 --model_name="ckpt-500"
```
Explain: 
This command will generate text based on the input prompt "One day" using the model weights stored in the `out` directory. The `max_num_tokens` parameter specifies the maximum number of tokens to generate, and the `model_name` parameter specifies the name of the model weights file to load. For `model_name`, you can specify the name of the model weights file without the extension, like "ckpt-500" or "ckpt-1000" or only "ckpt".

## Tables
| name_model   | BLOCK_SIZE | VOCAB_SIZE | N_LAYER | N_HEAD | N_EMBD | NUM_EXPERTS | NUM_ACTIVE_EXPERTS | EXPERT_DIM | DIM | DROPOUT | BIAS  | DATASET         |
|--------------|------------|------------|---------|--------|--------|-------------|--------------------|------------|-----|---------|-------|-----------------|
| ckpt-500.pth | 1024       | 50304      | 8       | 8      | 768    | 4           | 4                  | 512        | 768 | 0.0     | False | tinystories_15k |

## Results
Prompt: One day

Generated text: One day: Sharing positive bought Isabel a rainbow hug. Her name was an vitamins, so only one favorite thing to cheer she were.

Lily picked up a hay and proudly went to a small portion. She was very happened. When Tommy said it

Generated text length: 227 | Inference time: 3 seconds

## Dependencies
- torch
- numpy
- tiktoken

## Conclusion
The MicroRWKV model is a custom neural network architecture that combines several cutting-edge techniques, such as time-based and channel-based mixing, sliding window attention, grouped attention, and a Tiny Mixture of Experts (TinyMoE) layer. These components work together to enhance the model's ability to capture both local and global dependencies, as well as to learn specialized representations. The combination of these techniques results in a powerful and efficient model that can be used for a variety of natural language processing tasks.
