import os
import openai
openai.organization = os.getenv('OPENAI_ORG')
openai.api_key = os.getenv('OPENAI_KEY')
assert openai.organization is not None
assert openai.api_key is not None

from transformers import GPT2TokenizerFast
import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
import numpy as onp

def make_enc_dec():
  tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

  def itos(ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return tokenizer.convert_tokens_to_string(tokens)

  def stoi(string):
    tokens = tokenizer(string)['input_ids']

  return itos, stoi

itos, stoi = make_enc_dec()

def gpt_logprobs(ids_and_lens):
  ids, lens = ids_and_lens
  assert ids.ndim == 2
  assert lens.ndim == 1
  num_prompts, max_num_tokens = ids.shape

  # Decode all prompts
  prompts = []
  for i in range(ids.shape[0]):
    cropped = ids[i][:lens[i]]
    tok_list = [int(cropped[j]) for j in range(lens[i])]
    prompts.append(itos(tok_list))

  # Call OpenAI API
  out = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompts,
      max_tokens=0,
      temperature=1.,
      logprobs=1,
      echo=True)

  # Parse output.
  outs = []
  for lp in out['choices']:
    assert lp['index'] == len(outs)
    raw_lps = onp.array(lp['logprobs']['token_logprobs'], dtype=onp.float32)
    padded_lps = onp.pad(raw_lps, (0, max_num_tokens - raw_lps.shape[0]))
    outs.append(padded_lps)
  return onp.array(outs)


@jax.jit
def batch_gpt_logprobs(ids, lens):
  ids = jnp.reshape(ids, [-1, ids.shape[-1]])
  lens = jnp.reshape(lens, [-1])
  result_type = jnp.ones_like(ids, dtype=jnp.float32)
  out = hcb.call(gpt_logprobs, (ids, lens), result_shape=result_type)
  out = jnp.reshape(out, ids.shape)
  return out
