import functools
import jax
import jax.numpy as jnp
import snax
import equinox as eqx
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
import optax


class HMMLanguageModel(eqx.Module):

  M: jnp.ndarray
  p0: jnp.ndarray
  vocab_size: int = eqx.static_field()

  def __init__(self, key, vocab_size: int, M=None):
    self.vocab_size = vocab_size
    if M is None:
      self.M = jax.random.normal(key, (vocab_size, vocab_size))
    else:
      self.M = M
    self.p0 = jnp.zeros((vocab_size,), dtype=jnp.float32)

  def log_prob(self, tokens):
    assert tokens.ndim == 1
    # [num_tokens]
    inputs = tokens[:-1]
    targets = tokens[1:]
    # [num_tokens, vocab_size]
    logits = self.M[inputs]
    # [num_tokens, vocab_size]
    all_log_probs = jax.nn.log_softmax(logits, axis=1)
    # [num_tokens]
    token_log_probs = jax.vmap(lambda lp, t: lp[t])(all_log_probs, targets)
    p0_log_prob = tfd.Categorical(logits=self.p0).log_prob(tokens[0])
    return jnp.sum(token_log_probs) + p0_log_prob

  def sample_and_log_prob(self, key, num_new_tokens, prefix=None):
    if prefix is not None:
      init_input = prefix[-1]
      init_lp = jnp.array(0.)
    else:
      p0_dist = tfd.Categorical(logits=self.p0)
      key, subkey = jax.random.split(key)
      init_input = p0_dist.sample(seed=subkey)
      init_lp = p0_dist.log_prob(init_input)

    def scan_body(carry, _):
      prev_sample, lp, key = carry
      logits = self.M[prev_sample]
      out_dist = tfd.Categorical(logits=logits)
      key, subkey = jax.random.split(key)
      out_tok = out_dist.sample(seed=subkey)
      out_tok_lp = out_dist.log_prob(out_tok)
      return (out_tok, lp + out_tok_lp, key), out_tok

    out_carry, samples = jax.lax.scan(
        scan_body,
        (init_input, init_lp, key),
        xs=None,
        length=num_new_tokens-1)

    samples = jnp.concatenate([init_input[jnp.newaxis], samples], axis=0)
    return samples, out_carry[1]


class RNNLanguageModel(eqx.Module):

  rnn: snax.RNN
  embedding_mat: jnp.ndarray
  vocab_size: int = eqx.static_field()
  embedding_dim: int = eqx.static_field()

  def __init__(self,
               key,
               rnn_class,
               vocab_size: int,
               embedding_dim: int,
               hdims):
    k1, k2 = jax.random.split(key)
    self.rnn = rnn_class(k1, embedding_dim, hdims + [embedding_dim])
    self.embedding_mat = jax.random.normal(k2, shape=[vocab_size, embedding_dim])
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim

  def log_prob(self, tokens, len):
    assert tokens.ndim == 1
    # Embed the input sequence, [num_tokens, embedding_dim]
    embedded_toks = jnp.take(self.embedding_mat, tokens, axis=0)
    # Pad the tokens with the 'start token', and clip the end.
    inputs = jnp.pad(embedded_toks, ((1,0),(0,0)))[:-1]
    # Get the rnn output [num_tokens, embedding_dim]
    _, rnn_outputs = self.rnn(inputs)
    # Multiply RNN outs by embedding matrix to
    # get logits of shape [num_tokens, vocab_size]
    logits = rnn_outputs @ self.embedding_mat.T
    raw_log_probs = jax.nn.log_softmax(logits, axis=1)
    log_probs = jax.vmap(lambda lps, t: lps[t])(raw_log_probs, tokens)
    return jnp.sum(log_probs)

  def sample_and_log_prob(self, key, num_new_tokens, prefix=None):
    if prefix is not None:
      embedded_prefix = jnp.take(self.embedding_mat, prefix, axis=0)
      # Pad the tokens with the 'start token', and clip the end.
      inputs = jnp.pad(embedded_prefix, ((1,0),(0,0)))[:-1]
      # Get the state of the rnn
      rnn_state, _ = self.rnn(inputs)
      final_state = jax.tree_util.tree_map(lambda x: x[-1], rnn_state)
      next_input = embedded_prefix[-1]
    else:
      final_state = self.rnn.initial_state()
      next_input = jnp.zeros([self.embedding_dim], dtype=jnp.float32)

    def scan_body(carry, _):
      prev_state, prev_out_embedded, lp, key = carry
      # rnn_outs is [embedding_dim]
      new_state, rnn_outs = self.rnn.one_step(prev_state, prev_out_embedded)
      assert rnn_outs.shape == (self.embedding_dim,)
      # Multiply rnn_outs by embedding matrix to get logits
      logits = self.embedding_mat @ rnn_outs
      out_dist = tfd.Categorical(logits=logits)
      key, subkey = jax.random.split(key)
      out_tok = out_dist.sample(seed=subkey)
      out_tok_lp = out_dist.log_prob(out_tok)
      out_embedded = self.embedding_mat[out_tok]
      return (new_state, out_embedded, lp + out_tok_lp, key), out_tok

    out_carry, samples = jax.lax.scan(
        scan_body,
        (final_state, next_input, 0., key),
        xs=None,
        length=num_new_tokens)

    return samples, out_carry[-2]


def ds_itr_gen(ds_arr, batch_size, key):
  N = ds_arr.shape[0]
  assert N % batch_size == 0, "batch size does not evenly divide dataset size."
  batches_per_epoch = N // batch_size
  inds = jax.random.permutation(key, N)
  inds = inds.reshape([batches_per_epoch, batch_size])

  ind_fn = jax.jit(jax.vmap(lambda j: ds_arr[j]))

  for i in range(batches_per_epoch):
    yield ind_fn(inds[i])


def make_lm_dataset(key, batch_size, seq_len, num_seqs, vocab_size, embedding_dim, q_hdims):
  key, subkey = jax.random.split(key)
  model = RNNLanguageModel(
      subkey, snax.LSTM, vocab_size, embedding_dim, q_hdims)
  keys = jax.random.split(key, num=num_seqs)
  seqs, _ = jax.vmap(model.sample_and_log_prob, in_axes=(0, None))(keys, seq_len)
  ds_itr_fn = functools.partial(ds_itr_gen, seqs, batch_size)
  return ds_itr_fn, seqs


def pretrain_lm(
    seed = 0,
    batch_size = 16,
    seq_len = 32,
    num_seqs = 2_000,
    vocab_size = 6,
    embedding_dim = 4,
    lm_hdims = [128, 128],
    lr = 1e-4,
    epochs = 2):
  key = jax.random.PRNGKey(seed)
  key, sk1, sk2 = jax.random.split(key, num=3)
  ds, seqs = make_lm_dataset(
      sk1, batch_size, seq_len, num_seqs, vocab_size, embedding_dim, lm_hdims)
  model = RNNLanguageModel(
      sk2, snax.LSTM, vocab_size, embedding_dim, lm_hdims)

  # Create and initialize the optimizer
  opt = optax.adam(lr)
  opt_state = opt.init(model)

  def batch_loss(m, seqs):
    seq_len = seqs.shape[1]
    lps = jax.vmap(m.log_prob, in_axes=(0, None))(seqs, seq_len)
    return - jnp.mean(lps)

  @jax.jit
  def train_step(m, batch, opt_state):
    loss_val, grads = jax.value_and_grad(batch_loss, argnums=0)(m, batch)
    updates, new_opt_state = opt.update(grads, opt_state, m)
    new_model = optax.apply_updates(m, updates)
    return loss_val, new_model, new_opt_state

  sample = jax.jit(lambda m, k: m.sample_and_log_prob(k, seq_len))

  for e in range(epochs):
    key, subkey = jax.random.split(key)
    for i, batch in enumerate(ds(subkey)):
      key, subkey = jax.random.split(key)
      loss_val, model, opt_state = train_step(model, batch, opt_state)
      if i % 20 == 0:
        print(f"loss: {loss_val:0.4f}")
        #print(f"model sample {sample(model, subkey)}")
