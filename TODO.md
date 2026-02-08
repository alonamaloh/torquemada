# Torquemada TODO

## Flexible move input (UI)

The current click-based move input requires the user to click exact squares in
sequence. Instead, each legal move should have an associated **mask** of
relevant squares (start, end, intermediate squares, captured pieces). As the
player clicks squares, we narrow down the set of matching moves. Once the
clicked squares uniquely determine a move, we play it immediately.

This makes input more forgiving — the player can click any distinctive square
of the intended move, in any order, rather than needing to follow the exact
path.

## Retrain evaluation network with 5-piece adjudication

Currently self-play training games are adjudicated using tablebases whenever
the position reaches tablebase range. The problem: the separate DTM network
for 6-7 piece positions is trained on all tablebase entries rather than
positions that actually arise in games, so it doesn't perform well.

**Plan:**
- Regenerate training data, but only adjudicate when pieces <= 5 (not 6-7).
- This forces the main evaluation network to learn 6-7 piece positions from
  game contexts, which should improve its accuracy for those piece counts.
- The separate DTM network can be kept or dropped — need to test both
  approaches. If kept, train it on 6-7 piece positions that actually occur
  in self-play games rather than exhaustive tablebase entries.
