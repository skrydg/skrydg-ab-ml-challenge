import tensorflow as tf

def concatenate(datasets):
  assert(len(datasets) > 0)

  result = datasets[0]

  for i in range(1, len(datasets)):
    result = result.concatenate(datasets[i])

  return result
