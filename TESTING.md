# UI Testing Checklist

## Basic Play
- [ ] New game starts, white pieces respond to clicks
- [ ] Click a piece, see orange highlights for valid destinations
- [ ] Click destination to make a move — hear sound
- [ ] Engine responds with its move — see animation, hear sound
- [ ] Multi-jump captures animate step by step with sound on each jump
- [ ] Last move highlighted in yellow after each move

## Move Input Edge Cases
- [ ] Click wrong square, then click a valid piece — resets selection
- [ ] Multi-jump with repeated landing squares works (sequential clicking)
- [ ] Click squares out of order in a multi-capture — flexible input resolves

## Undo / Redo
- [ ] Undo while idle — takes back move, you play the side now to move
- [ ] Undo while engine is thinking — stops search, undoes
- [ ] Undo while pondering — stops ponder, undoes
- [ ] Redo works after undo
- [ ] Redo disabled while engine is thinking
- [ ] Making a new move clears redo stack

## "Juega!" Button
- [ ] When idle on your turn: engine takes your side, plays a move
- [ ] When engine is thinking: stops search (engine plays best found so far)
- [ ] When pondering with PV: plays the top PV move, switches sides
- [ ] Button disabled when game is over

## New Game Dialog
- [ ] "Blancas" — you play white, board not flipped
- [ ] "Negras" — you play black, board flipped, engine moves first
- [ ] "Ambos" — two-player mode, no engine moves

## Mode Buttons (Motor B / Motor N / 2J)
- [ ] Switching to "Motor B" (engine white) — engine plays if it's white's turn
- [ ] Switching to "Motor N" (engine black) — engine plays if it's black's turn
- [ ] Switching to "2J" — no engine moves, pondering if enabled

## Edit Mode
- [ ] Click "Editar" — controls switch, piece selector appears
- [ ] Click squares to place/remove pieces
- [ ] Cycle through piece types by clicking same square type
- [ ] "Limpiar" clears the board
- [ ] Side-to-move toggle (B/N) works
- [ ] "Hecho" — exits edit, position applied, you play the side to move

## Pondering & Analysis
- [ ] Enable pondering checkbox — background search starts
- [ ] Analysis display checkbox — eval bar + search info appear
- [ ] Depth, score, nodes, nps, PV update in real time
- [ ] Root moves shown when pondering in engine mode
- [ ] Book moves shown when in book position
- [ ] Disabling pondering stops search, hides info

## Time Control
- [ ] Click "+3s/mov" — dialog opens with current value
- [ ] Change value, OK — label updates
- [ ] Engine respects time bank (fast moves early, longer later)
- [ ] Time display counts down during search

## Match Play
- [ ] "Partida" in new game dialog starts match mode
- [ ] Toolbar changes (resign button visible, normal controls hidden)
- [ ] Colors alternate between games
- [ ] Game over shows W/D/L stats dialog
- [ ] "OK" exits match mode, restores previous settings
- [ ] Resign button asks confirmation, records loss

## Draw Handling
- [ ] Engine offers draw in drawish positions (few pieces, proven draw)
- [ ] Accept — game ends as draw
- [ ] Decline — engine plays its move, no more offers this game
- [ ] 60-move rule triggers automatic draw
- [ ] Threefold repetition triggers automatic draw

## Game History
- [ ] Stats button opens history dialog
- [ ] Saved games listed with date, moves preview, result
- [ ] Select a game, click "Analizar" — loads game, rewinds to start
- [ ] Redo through the loaded game move by move
- [ ] Delete a game from history

## Board
- [ ] Flip button rotates board 180°
- [ ] Eval bar flips orientation with board
- [ ] Sound checkbox toggles move sounds
- [ ] Opening book checkbox toggles book usage

## Preferences Persistence
- [ ] Enable pondering + analysis, reload page — both still enabled
- [ ] Disable sound, reload — still disabled
- [ ] Disable opening book, reload — still disabled

## Tablebase Download
- [ ] "Descargar DTM" button opens download dialog (if OPFS available)
- [ ] "Descargar CWDL" button opens download dialog
