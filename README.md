## Installing dependencies

    $ pip install -r requirements.txt

## Text preprocessing

    $ cat /tmp/SSJ/T/K/{L,S}/* \
      | ./text/collect \
      | ./text/sentences \
      | ./text/filter \
      > corpus.txt

    $ cat corpus.txt \
      | ./text/segment \
      > segments.txt

    $ cat segments.txt \
      | ./text/count_words_per_line \
      | sort -n
      > segment_lengths.txt

## Show segment length distribution

    $ python stats/segment_length_distribution.py segment_lengths.txt

## Running tests

    $ pip install pytest
    $ env PYTHONPATH=src py.test
