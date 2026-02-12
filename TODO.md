# Torquemada - Planned Features

## 1. Analysis Mode

A new game mode alongside "Engine White", "Engine Black", and "2 Players".

### Behavior
- Player controls both sides (like "2 Players")
- Engine continuously analyzes the current position in the background
- Search info (depth, score, PV, NPS) is displayed live, as when the engine thinks
- When the player makes a move, the current search is stopped and a new analysis begins on the resulting position
- No time management: engine searches indefinitely (or to a large fixed depth) until interrupted
- Full undo/redo support; after undo/redo, engine restarts analysis on the restored position
- No auto-play: the engine never makes a move, it only analyzes

### UI
- Add an "Analysis" button to the mode selector (alongside Engine White / Engine Black / 2 Players)
- When active, search info panel is always visible
- Consider showing an evaluation bar on the side of the board (stretch goal)
- The mode button shows active state like other mode buttons

### Implementation Notes
- In `game-controller.js`: new `humanColor` value or a separate `analysisMode` flag
- Engine search with `softTime=Infinity, hardTime=Infinity` (or very large values)
- On each move/undo/redo: `stopSearch()` then start new background search
- Search results update the display but do NOT trigger a move
- `_engineMove()` should not be called; instead a new `_analyzePosition()` method
- Opening book should be disabled in analysis mode (we want real engine evaluation)

---

## 2. Game Recording (Persistence)

Save completed match-play games to browser storage so they survive page reloads.

### What to Store Per Game
- `id`: Unique identifier (timestamp-based, e.g., `Date.now()`)
- `date`: ISO 8601 timestamp of game start
- `moves`: Array of move notations (e.g., `["11-15", "23-19", "8-11", ...]`)
- `result`: `"white"` | `"black"` | `"draw"`
- `resultReason`: `"no_moves"` | `"resignation"` | `"repetition"` | `"50_moves"` | ...
- `playerColor`: `"white"` | `"black"`

### Storage Strategy
- Use `localStorage` with key `torquemada-games`
- Store as JSON array of game objects
- Each game is small (~200-500 bytes), so localStorage's ~5MB limit allows thousands of games
- If we ever hit limits, consider migrating to IndexedDB (but unlikely needed)

### When to Record
- Automatically on game-over during match play only
- Do NOT record single games, 2-player games, or analysis sessions
- Do NOT record incomplete/abandoned games (only finished games)

### Implementation Notes
- New module: `game-storage.js` with `saveGame(gameData)`, `getGames()`, `deleteGame(id)`, `clearGames()`
- Hook into the `onGameOver` callback in `main.js`
- Build the game record from `gameController.history` (extract move notations)

---

## 3. Stats & Game History Dialog (replaces current Stats dialog)

The current "Stats" button is only visible during match play. We change this:
- The Stats button moves to the **main toolbar** (visible when NOT in match play)
- During match play, the Stats button is hidden (matches are in progress; stats shown at game end)
- The dialog combines W/D/L counters with the full game history list

### UI Design
- Opens a modal dialog (consistent with existing dialog style)
- **Top section**: W/D/L match stats counters (same as current stats dialog)
- **Below**: Scrollable list of recorded match games, most recent first
- Each entry shows on one line:
  - Date and time (e.g., "2025-06-15 14:30")
  - First few moves as preview (e.g., "11-15 23-19 8-11 ...")
  - Result icon/text: W (green), D (gray), L (red) from the player's perspective
  - Player color indicator (small white/black circle)
- Clicking a game selects it (highlighted row)
- Buttons at the bottom:
  - "Analyze" - loads the selected game in analysis mode (disabled if none selected)
  - "Delete" - deletes the selected game
  - "Close" - closes the dialog
- Optional: "Clear All" button to delete all recorded games and reset stats (with confirmation)

### Responsive
- On desktop: dialog is ~500px wide, list height ~400px with scroll
- On mobile: dialog is full-width with appropriate padding

### Stats Derivation
- W/D/L counters are derived from the stored game list (no separate counter storage)
- This replaces the current `torquemada-match-stats` localStorage key
- Migration: on first load, if old stats exist but no games, show legacy counters with a note

---

## 4. Load Game for Analysis

When a game is selected from the history and "Load" is pressed:

### Behavior
- Switch to Analysis mode automatically
- Set up the initial position (standard or custom if the game had one)
- Replay all moves from the game into the history
- Navigate to the final position
- The user can then use undo/redo to step through the game
- Engine analyzes whatever position is currently on the board
- The redo stack is populated with all moves, so the user can step forward/backward through the entire game

### Implementation Approach
- Option A (simpler): Load the initial position, then replay all moves programmatically to build the history. The user starts at the final position and can undo back.
- Option B (richer): Load the initial position, populate the full history array directly, position the "cursor" at move 0, and let the user step forward with redo. This way the user starts at the beginning.
- **Preferred: Option B** - more natural for game review. User sees the game from the start and steps forward.

### Implementation Notes
- New method in `game-controller.js`: `loadGame(moves, initialPosition?)` that:
  1. Resets the board to the initial position
  2. Replays all moves to build the internal history
  3. Then undoes all of them so the position is at move 0
  4. The redo stack now contains the full game
- Alternatively, directly populate `history` and `redoStack` arrays (but must keep engine board in sync)

---

## Implementation Order

Suggested order to build these features incrementally:

1. **Analysis Mode** - Independent feature, no storage needed. Provides the foundation for game review.
2. **Game Storage Module** - Create `game-storage.js`, hook into game-over events. Games start accumulating.
3. **Game History Dialog** - UI to view recorded games. Can test with games recorded in step 2.
4. **Load Game for Analysis** - Connect history viewer to analysis mode. The full loop is complete.

Each step is independently useful and testable.
