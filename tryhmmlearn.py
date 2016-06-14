import sys
import numpy as np

from hmmlearn import hmm
from seqfile import Seqfile

np.random.seed(seed=None)

num_states = int(sys.argv[1])
seq_path = sys.argv[2]

with open(seq_path) as f:
    sf = Seqfile(f)

seq = (i - 1 for i in sf.seq)
seq = np.fromiter(seq, np.int64)
seq = np.atleast_2d(seq).T

model = hmm.MultinomialHMM(n_components=num_states)

# r = model.fit(seq, [5, 5])
r = model.fit(seq)

sys.stderr.write("""\
Training sequence length: {0}
Alphabet size: {0}
""".format(seq.size, model.n_features))
# print(repr(model.startprob_))
# print(repr(model.transmat_))
# print(repr(model.emissionprob_))

symbols, _states = model.sample(100)

for num in np.squeeze(symbols):
    sys.stderr.write("{0} ".format(num))

print("")
