Problems:

1) **final blocks** is the final draft, however the block deflation does not work as expected: After the first eigenvals converge, the algorithms stagnates. Most likely due to the fact that the already determined eigenvals correspond to almost the whole span! Thus there's nothing left to iterate over!

2) **Davidson deflated improve** takes out single eigenvalues as the algorithm progresses

3) **adaptive deflation** bislang bestfunktionierenden Algorithmus mit dem von Tobias

4) **final deflation algorithm**, same as adaptive but more compact

5) **new_concept, basierend auf dem paper von sadkane und sidje, works ok but I have the problem with small basis


