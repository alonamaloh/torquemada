# Tareas

## Engine-vs-engine mode
Add a mode where the engine plays both sides automatically, so users can watch the engine play against itself.

## Pondering (think on opponent's time)
Allow the engine to think during the opponent's turn. When the opponent makes a move, if it matches the predicted move, the search continues seamlessly; otherwise, the search restarts.

## Permanent brain (instant response mode)
A mode where the engine thinks on the opponent's time about *all* possible opponent moves in parallel (or sequentially). When the opponent plays, the engine responds immediately with the pre-computed best reply.

## Unified "¡Juega!" button
Replace the "Parar" button with a "¡Juega!" button that adapts to context:
- **Engine is thinking**: Stop search, play best move found so far.
- **Human's turn**: Switch so the engine takes over this side and starts thinking.
- **Analysis mode**: Play the top move on the board, stay in analysis mode.
- **Engine-vs-engine mode**: Stop search, play now, continue the match.
