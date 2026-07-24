"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    vocab = {}

    for token in specials:
        if token not in vocab:
            vocab[token] = len(vocab)

    for sentence in sentences:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    return {id_: token for token, id_ in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    unk_id = token_to_id[unk_token]

    return [token_to_id.get(token, unk_id) for token in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    return [id_to_token[idx] for idx in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    if len(ids) >= max_len:
        return ids[:max_len]

    padding_needed = max_len - len(ids)
    return ids + [pad_id] * padding_needed

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    even_indices = torch.arange(0, d_model, 2, dtype=torch.float)

    div_term = torch.exp(even_indices * -(math.log(10000) / d_model))

    return div_term

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len, dtype=torch.float).unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe[:, 0::2] = torch.sin(position * div_term)

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len, d_model, dtype=torch.float)

    position = build_position_index_column(max_len)

    div_term = compute_positional_div_term(d_model)

    pe = fill_even_indices_with_sin(pe, position, div_term)

    pe = fill_odd_indices_with_cos(pe, position, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    seq_len = embedded_batch.size(1)

    pe_slice = positional_encoding[:seq_len,:]

    return embedded_batch + pe_slice

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    base_mask = (token_ids != pad_id)

    return base_mask.unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    base_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    
    lower_triangle = torch.tril(base_mask)

    return lower_triangle.view(1, 1, seq_len, seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    k_transpose = key.transpose(-2, -1)

    return torch.matmul(query, k_transpose)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    weights = F.softmax(masked_scores, dim=-1)

    return torch.nan_to_num(weights, nan=0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    d_k = key.shape[-1]

    raw_scores = compute_raw_attention_scores(query, key)

    scaled_scores = scale_attention_scores(raw_scores, d_k)

    if mask is not None:
        scaled_scores = mask_attention_scores_with_neg_inf(scaled_scores, mask)

    attention_weights = softmax_attention_weights(scaled_scores)

    context = apply_attention_weights_to_values(attention_weights, value)

    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.size()

    d_k = d_model // num_heads

    return tensor.view(B, L, num_heads, d_k)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B, num_heads, L, d_k = multi_head_tensor.size()

    return multi_head_tensor.transpose(1, 2).contiguous().view(B, L, num_heads * d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    out = torch.matmul(x, weight.transpose(-1, -2))

    if bias is not None:
        out = out + bias
    
    return out

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)

    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q_split = split_last_dim_into_heads(q, num_heads)
    q_h = transpose_heads_before_sequence(q_split)

    k_split = split_last_dim_into_heads(k, num_heads)
    k_h = transpose_heads_before_sequence(k_split)

    v_split = split_last_dim_into_heads(v, num_heads)
    v_h = transpose_heads_before_sequence(v_split)

    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    context, attention_weights = scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)

    return context, attention_weights

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    merged_context = merge_heads_back_to_model_dim(context)

    output = apply_linear_projection(merged_context, w_o, b_o)

    return output

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q = apply_linear_projection(query, w_q, bias=None)
    k = apply_linear_projection(key, w_k, bias=None)
    v = apply_linear_projection(value, w_v, bias=None)

    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)

    context, attention_weights = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)

    output = merge_heads_and_project_output(context, w_o, b_o=None)

    return output

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    linear_projection = torch.matmul(x, w1) + b1

    return torch.relu(linear_projection)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return torch.matmul(hidden, w2) + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)

    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = x.mean(dim=-1, keepdim=True)

    variance = x.var(dim=-1, unbiased=False, keepdim=True)

    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, variance = compute_layer_norm_mean_and_variance(x)

    normalized_x = (x - mean) / torch.sqrt(variance + eps)

    return gamma * normalized_x + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    summed_output = residual_input + sublayer_output

    return normalize_and_scale_with_gamma_beta(summed_output, gamma, beta, eps=eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    return (x * keep_mask) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    attn_output = assemble_multi_head_attention_forward(
        query=x,
        key=x,
        value=x,
        w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, num_heads=num_heads, mask=src_mask
    )

    final_output = apply_residual_add_and_norm(
        residual_input=x, sublayer_output=attn_output, gamma=gamma, beta=beta
    )

    return final_output

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    ffn_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)

    final_output = apply_residual_add_and_norm(x, ffn_output, gamma, beta)

    return final_output

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    attn_out = encoder_layer_self_attention_sublayer(
        x,
        layer_params['w_q'],
        layer_params['w_k'],
        layer_params['w_v'],
        layer_params['w_o'],
        layer_params['attn_gamma'],
        layer_params['attn_beta'],
        num_heads,
        src_mask
    )

    final_out = encoder_layer_feed_forward_sublayer(
        attn_out,
        layer_params['w1'],
        layer_params['b1'],
        layer_params['w2'],
        layer_params['b2'],
        layer_params['ffn_gamma'],
        layer_params['ffn_beta']
    )

    return final_out

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    hidden_state = x

    for layer_params in encoder_layer_params_list:
        hidden_state = assemble_encoder_layer(
            hidden_state,
            layer_params,
            num_heads,
            src_mask
        ) 

    return hidden_state

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    attn_output = assemble_multi_head_attention_forward(
        y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask
    )

    final_output = apply_residual_add_and_norm(y, attn_output, gamma, beta)

    return final_output

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    
    safe_mask = src_mask
    if safe_mask is not None:
        # Explicitly reshape the mask to (B, 1, 1, T_src) to broadcast cleanly
        B, T_src = safe_mask.shape[0], safe_mask.shape[-1]
        safe_mask = safe_mask.view(B, 1, 1, T_src)

    # CRITICAL FIX: Pass 'safe_mask' instead of 'src_mask'
    attn_output = assemble_multi_head_attention_forward(
        y, encoder_output, encoder_output, 
        w_q, w_k, w_v, w_o, 
        num_heads, safe_mask
    )

    # Apply the residual connection and layer normalization
    final_output = apply_residual_add_and_norm(y, attn_output, gamma, beta)

    return final_output

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    ffn_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)

    final_output = apply_residual_add_and_norm(y, ffn_output, gamma, beta)
    return final_output

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # 1) Masked self-attention over the target stream (uses tgt_mask)
    y_self_attn = decoder_layer_masked_self_attention_sublayer(
        y,
        layer_params['w_q_self'], layer_params['w_k_self'],
        layer_params['w_v_self'], layer_params['w_o_self'],
        layer_params['self_gamma'], layer_params['self_beta'],
        num_heads, tgt_mask
    )

    # 2) Cross-attention: queries from target, keys/values from encoder_output (uses src_mask)
    y_cross_attn = decoder_layer_cross_attention_sublayer(
        y_self_attn, encoder_output,
        layer_params['w_q_cross'], layer_params['w_k_cross'],
        layer_params['w_v_cross'], layer_params['w_o_cross'],
        layer_params['cross_gamma'], layer_params['cross_beta'],
        num_heads, src_mask
    )

    # 3) Position-wise feed-forward (no mask)
    final_output = decoder_layer_feed_forward_sublayer(
        y_cross_attn,
        layer_params['w1'], layer_params['b1'],
        layer_params['w2'], layer_params['b2'],
        layer_params['ffn_gamma'], layer_params['ffn_beta']
    )

    return final_output

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    for layer_params in decoder_layer_params_list:
        y = assemble_decoder_layer(
            y, encoder_output, layer_params, num_heads, src_mask, tgt_mask
        )
    
    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output,output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.t()

# Step 50 - apply_log_softmax_over_vocab
import torch.nn.functional as F

def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return F.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    token_embedding = model_params['token_embedding']
    d_model = token_embedding.size(-1)

    # 1. Embed source and target token ids, then scale by sqrt(d_model)
    src_embedded = scale_embeddings_by_sqrt_d_model(token_embedding[src_ids], d_model)
    tgt_embedded = scale_embeddings_by_sqrt_d_model(token_embedding[tgt_ids], d_model)

    # 2. Add sinusoidal positional encoding (one PE table, big enough for both sides)
    max_len = max(src_ids.size(1), tgt_ids.size(1))
    positional_encoding = build_sinusoidal_positional_encoding(max_len, d_model)
    src_embedded = add_positional_encoding_to_embeddings(src_embedded, positional_encoding)
    tgt_embedded = add_positional_encoding_to_embeddings(tgt_embedded, positional_encoding)

    # 3. Build masks
    src_mask = build_padding_mask(src_ids, pad_id)                              # (B, 1, 1, L_src)
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)                       # (B, 1, 1, L_tgt)
    causal_mask = build_causal_mask(tgt_ids.size(1))                            # (1, 1, L_tgt, L_tgt)
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, causal_mask)   # (B, 1, L_tgt, L_tgt)

    # 4. Run encoder and decoder stacks
    encoder_output = stack_encoder_layers(src_embedded, model_params['encoder_layers'], num_heads, src_mask)
    decoder_output = stack_decoder_layers(tgt_embedded, encoder_output, model_params['decoder_layers'],
                                           num_heads, src_mask, tgt_mask)

    # 5. Project to vocab logits, then log-softmax
    logits = apply_final_output_projection(decoder_output, model_params['output_projection'])
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    d_model_scale = 1.0 / math.sqrt(d_model)
    d_ff_scale = 1.0 / math.sqrt(d_ff)

    return {
        'w_q': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_k': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_v': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_o': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),

        'w1': (torch.randn(d_model, d_ff) * d_model_scale).requires_grad_(True),
        'b1': torch.zeros(d_ff, requires_grad=True),
        'w2': (torch.randn(d_ff, d_model) * d_ff_scale).requires_grad_(True),
        'b2': torch.zeros(d_model, requires_grad=True),

        'attn_gamma': torch.ones(d_model, requires_grad=True),
        'attn_beta': torch.zeros(d_model, requires_grad=True),
        'ffn_gamma': torch.ones(d_model, requires_grad= True),
        'ffn_beta': torch.zeros(d_model, requires_grad=True)

    }

# Step 53 - init_decoder_layer_parameters
import torch
import math

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    d_model_scale = 1.0 / math.sqrt(d_model)
    d_ff_scale = 1.0 / math.sqrt(d_ff)

    return {
        'w_q_self': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_k_self': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_v_self': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_o_self': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),

        'w_q_cross': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_k_cross': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_v_cross': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),
        'w_o_cross': (torch.randn(d_model, d_model) * d_model_scale).requires_grad_(True),

        'w1': (torch.randn(d_model, d_ff) * d_model_scale).requires_grad_(True),
        'b1': torch.zeros(d_ff, requires_grad=True),
        'w2': (torch.randn(d_ff, d_model) * d_ff_scale).requires_grad_(True),
        'b2': torch.zeros(d_model, requires_grad=True),

        'self_gamma': torch.ones(d_model, requires_grad=True),
        'self_beta': torch.zeros(d_model, requires_grad=True),
        'cross_gamma': torch.ones(d_model, requires_grad=True),
        'cross_beta': torch.zeros(d_model, requires_grad=True),
        'ffn_gamma': torch.ones(d_model, requires_grad=True),
        'ffn_beta': torch.zeros(d_model, requires_grad=True)

    }

# Step 54 - init_embedding_and_projection_parameters
import torch
import math

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    scale = 1.0 / math.sqrt(d_model)

    src_embedding = (torch.randn(vocab_size, d_model) * scale).requires_grad_(True)
    tgt_embedding = (torch.randn(vocab_size, d_model) * scale).requires_grad_(True)

    if tie_weights:
        output_projection = tgt_embedding
    else:
        output_projection = (torch.randn(vocab_size, d_model) * scale).requires_grad_(True)

    return {
        'src_embedding': src_embedding,
        'tgt_embedding': tgt_embedding,
        'output_projection': output_projection
    }

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

