import jax
import jax.numpy as jnp
import snax
import optax

from lm import RNNLanguageModel, HMMLanguageModel

def make_prompt_inference_dataset(
        key,
        model,
        prompt_len,
        suffix_len,
        num_suffixes):
  key, subkey = jax.random.split(key)
  prompt, _ = model.sample_and_log_prob(subkey, prompt_len)
  keys = jax.random.split(key, num=num_suffixes)
  sample_suffix = lambda k: model.sample_and_log_prob(k, suffix_len, prefix=prompt)
  suffixes, _ = jax.vmap(sample_suffix)(keys)
  return prompt, suffixes

def rws_loss(q, key, suffixes, log_p_fn, num_samples=10, prompt_len=10):
  num_suffixes, suffix_len = suffixes.shape
  keys = jax.random.split(key, num=num_samples)
  # [num_samples, prompt_len]
  sampled_prompts, log_qs = jax.vmap(
      q.sample_and_log_prob, in_axes=(0, None))(keys, prompt_len)

  # [num_samples * num_suffixes, prompt_len]
  tiled_prompts = jnp.tile(sampled_prompts, [num_suffixes, 1])
  # [num_samples * num_suffixes]
  tiled_log_qs = jnp.tile(log_qs, [num_suffixes])

  # [num_samples * num_suffixes, suffix_len]
  tiled_suffixes = jnp.reshape(jnp.tile(suffixes, [1, num_samples]),
                               [num_samples * num_suffixes, suffix_len])

  # Concatenate the sampled prompts with the suffixes.
  full_seqs = jnp.concatenate([tiled_prompts, tiled_suffixes], axis=1)

  full_seq_len = prompt_len + suffix_len
  full_seq_lens = jnp.full([num_samples * num_suffixes], full_seq_len)

  # [num_samples * num_suffixes]
  log_ps = log_p_fn(full_seqs, full_seq_lens)
  assert log_ps.shape == (num_samples * num_suffixes,)
  log_ws_raw = log_ps - tiled_log_qs
  assert log_ws_raw.shape == (num_samples * num_suffixes,)
  log_ws = jnp.reshape(log_ws_raw, [num_suffixes, num_samples])

  normalized_ws = jax.nn.softmax(log_ws, axis=1)
  normalized_ws_sg = jax.lax.stop_gradient(normalized_ws)
  losses = normalized_ws_sg * jnp.reshape(tiled_log_qs, [num_suffixes, num_samples])
  return - jnp.mean(jnp.sum(losses, axis=1))

def train_prompt_inference(
    key,
    prompt_len,
    suffix_len,
    num_suffixes,
    vocab_size,
    embedding_dim,
    model_hdims,
    rws_num_samples,
    num_train_steps,
    lr,
    optimizer=optax.adam):
  # Create the dataset
  sticky_mat = (jnp.eye(vocab_size) - 1.) * 2.
  true_model = HMMLanguageModel(key, vocab_size, M=sticky_mat)

  key, subkey = jax.random.split(key)

  true_prompt, suffixes = make_prompt_inference_dataset(
      key, true_model, prompt_len, suffix_len, num_suffixes)

  log_p_fn = lambda s, _: jax.vmap(true_model.log_prob)(s)

  # Make the prompt model.
  key, subkey = jax.random.split(key)
  q = RNNLanguageModel(subkey, snax.LSTM, vocab_size,
                             embedding_dim, model_hdims)

  # Create and initialize the optimizer
  opt = optimizer(lr)
  opt_state = opt.init(q)

  def loss(model, key):
    return rws_loss(model, key, suffixes, log_p_fn,
                    num_samples=rws_num_samples,
                    prompt_len=prompt_len)

  @jax.jit
  def train_step(key, model, cur_opt_state):
    loss_val, grads = jax.value_and_grad(loss, argnums=0)(model, key)
    updates, new_opt_state = opt.update(grads, cur_opt_state, model)
    new_model = optax.apply_updates(model, updates)
    return loss_val, new_model, new_opt_state

  for t in range(num_train_steps):
    key, subkey = jax.random.split(key)
    loss_val, q, opt_state = train_step(subkey, q, opt_state)
    if t % 10 == 0:
      log_p_true_prompt = q.log_prob(true_prompt, prompt_len)
      print(f"Step {t} loss: {loss_val:0.4f}, log p(true prompt): {log_p_true_prompt:0.4f}")
      key, subkey = jax.random.split(key)
      print(f"  sample: {q.sample_and_log_prob(key, prompt_len)}")


train_prompt_inference(
        key = jax.random.PRNGKey(0),
        prompt_len = 3,
        suffix_len = 17,
        num_suffixes = 16,
        vocab_size = 2,
        embedding_dim = 4,
        model_hdims = [32],
        rws_num_samples = 32,
        num_train_steps = 10_000,
        lr= 1e-3)
