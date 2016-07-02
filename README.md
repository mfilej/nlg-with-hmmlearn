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
      > counts.txt

## Running tests

    $ pip install pytest
    $ env PYTHONPATH=src py.test
