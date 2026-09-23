# Day 1

*21 Sept 2026*

## The idea

You tell me which model you're using and how you're shrinking it, I tell you how much accuracy
you'll lose. Right now people either trust the vendor's number or spend a day of GPU time
checking it themselves.

I gave myself one night to find out if it was even possible.

## What happened

I read for the first couple of hours instead of coding, which saved me the week.

Neural Magic had already run half a million evaluations on this. Buried in their numbers was the
problem: the accuracy changes are tiny. Usually under three points. Then I worked out how much
measurement error is in these benchmarks anyway, and on some of them it's one to two points.

So I'd be trying to predict a 3-point signal buried in 2 points of noise. I wrote that down
before I built anything.

## Results

Scraped 102 Red Hat model cards into 850 real before/after evaluations.

The machine learning lost to guessing. Ridge regression did worse than just predicting the
average. Gradient boosting did worse than that. Every feature I added made it worse.

The only thing that beat the average was a six-row lookup table on compression method. The whole
useful model is twelve numbers.

But the ranges worked. I froze it, pointed it at model families it had never seen, and the real
answer landed inside the 90% range 90.1% of the time. 118 out of 131. Then I rebuilt the scoring
from scratch, no shared code, because a number that clean makes me suspicious. Same answer.

## Then I tried to break it

Spent the last stretch attacking my own result. Found four things I'd overstated. Two of my
"unseen" model families weren't unseen. A model I claimed to have tested had been silently thrown
away by a validation check. One benchmark was being read as a completely different benchmark,
mixing scores of 95 in with scores of 15. And a filter I'd written had quietly excluded two thirds
of the available data.

The headline held up. But at 90.1% on 131 rows, not the tidier version I'd written first. I
changed the docs, not the framing.

## Where that leaves it

It's not a predictor. It's a risk range. You pick a compression method, it tells you where your
accuracy will probably land, and how well that promise has actually been tested.

It's honest about failing, too. It can say "this looks fine." It can't say "don't do this" yet.
And the big losses, the ones that actually matter, are exactly where it misses. I compressed a 1B
model hard and it blew through the range on four of six benchmarks.

I'd rather ship that with the failure modes on the label.

## Tomorrow

Two thirds of the data is still sitting there because of my own filter. Half a day of work, and
it's the only thing that fixes the one method where the range is genuinely unreliable.

Then I need to find out if anyone wants it, which I can't do from my laptop.
