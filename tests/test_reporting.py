from machineguard.reporting import describe_removed_duplicate


def test_describes_which_feature_was_kept_and_how_strongly_they_match():
       line = describe_removed_duplicate("rms", "std", 0.98)
       assert line == "rms removed: duplicate of std (correlation 0.98)"

def test_uses_the_names_and_the_sign_it_is_given():
       line = describe_removed_duplicate("kurtosis", "skewness", -0.97)
       assert line == "kurtosis removed: duplicate of skewness (correlation -0.97)"