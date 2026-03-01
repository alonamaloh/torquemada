/**
 * Pure game model — board, history, undo/redo, legal moves, game-over detection.
 * No knowledge of human/engine turns, no async search, no UI.
 * Communicates via EventTarget events.
 */

export class GameState extends EventTarget {
    constructor(engine) {
        super();
        this.engine = engine;

        // Board state (plain object, not a JSBoard)
        this.board = null;  // {white, black, kings, whiteToMove, pieceCount, nReversible}

        // History
        this.history = [];      // Array of {board, move, notation}
        this.redoStack = [];    // Stack of undone moves for redo

        // Legal moves for current position
        this.legalMoves = [];

        // Position tracking for threefold repetition
        this.positionCounts = new Map();  // key → count

        // Game over state
        this.gameOver = false;
        this.winner = null;       // 'white'|'black'|'draw'|null
        this.gameOverReason = null;
    }

    /**
     * Initialize: fetch starting position and legal moves
     */
    async init() {
        this.board = await this.engine.getInitialBoard();
        this._incrementPosition(this.board);
        await this._updateLegalMoves();
    }

    /**
     * Start a new game from the initial position
     */
    async newGame() {
        this.history = [];
        this.redoStack = [];
        this.gameOver = false;
        this.winner = null;
        this.gameOverReason = null;
        this.positionCounts = new Map();

        this.board = await this.engine.getInitialBoard();
        this._incrementPosition(this.board);
        await this._updateLegalMoves();

        this.dispatchEvent(new CustomEvent('newGame', { detail: { board: this.board } }));
    }

    /**
     * Make a move. Does NOT trigger engine search — callers react via events.
     * @param {Object} move - Move object from engine (with from_xor_to, captures, notation, etc.)
     * @returns {Object} The new board state
     */
    async makeMove(move) {
        // Clear redo stack since we're branching
        this.redoStack = [];

        // Save current position in history
        this.history.push({
            board: { ...this.board },
            move: move,
            notation: move.notation
        });

        // Execute move via engine
        const newBoard = await this.engine.makeMove(this.board, move);
        this.board = newBoard;
        this._incrementPosition(newBoard);

        // Update legal moves (also checks game-over)
        await this._updateLegalMoves();

        const detail = {
            move,
            board: this.board,
            gameOver: this.gameOver,
            winner: this.winner,
            reason: this.gameOverReason
        };
        this.dispatchEvent(new CustomEvent('move', { detail }));

        return this.board;
    }

    /**
     * Undo the last move
     * @returns {boolean} true if undo was performed
     */
    async undo() {
        if (this.history.length === 0) return false;

        // Decrement position count for current board
        this._decrementPosition(this.board);

        // Pop last entry, push to redo stack
        const last = this.history.pop();
        this.redoStack.push(last);

        // Restore previous board
        this.board = { ...last.board };

        // Reset game over
        this.gameOver = false;
        this.winner = null;
        this.gameOverReason = null;

        // Update legal moves
        await this._updateLegalMoves();

        this.dispatchEvent(new CustomEvent('undo', {
            detail: { board: this.board, undoneMove: last.move }
        }));

        return true;
    }

    /**
     * Redo the last undone move
     * @returns {boolean} true if redo was performed
     */
    async redo() {
        if (this.redoStack.length === 0) return false;

        const entry = this.redoStack.pop();

        // Re-add to history
        this.history.push(entry);

        // Execute the move via engine
        const newBoard = await this.engine.makeMove(this.board, entry.move);
        this.board = newBoard;
        this._incrementPosition(newBoard);

        // Update legal moves (also checks game-over)
        await this._updateLegalMoves();

        this.dispatchEvent(new CustomEvent('redo', {
            detail: { board: this.board, move: entry.move }
        }));

        return true;
    }

    /**
     * Set a custom board position (from edit mode)
     */
    async setPosition(white, black, kings, whiteToMove) {
        this.history = [];
        this.redoStack = [];
        this.gameOver = false;
        this.winner = null;
        this.gameOverReason = null;
        this.positionCounts = new Map();

        // Build board data — engine.makeMove expects nReversible, default to 0
        this.board = { white, black, kings, whiteToMove, pieceCount: 0, nReversible: 0 };
        // Get the real pieceCount from the engine
        const moves = await this.engine.getLegalMoves(this.board);
        // We need to reconstruct to get pieceCount; use a dummy makeMove round-trip?
        // Actually, we can compute pieceCount locally:
        this.board.pieceCount = popcount(white) + popcount(black);

        this._incrementPosition(this.board);
        await this._updateLegalMoves();

        this.dispatchEvent(new CustomEvent('positionChanged', { detail: { board: this.board } }));
    }

    /**
     * Load a game from move notations. Replays all moves, then rewinds to start.
     * @param {string[]} moves - Array of move notation strings
     */
    async loadGame(moves) {
        // Reset to initial position
        this.history = [];
        this.redoStack = [];
        this.gameOver = false;
        this.winner = null;
        this.gameOverReason = null;
        this.positionCounts = new Map();

        this.board = await this.engine.getInitialBoard();
        this._incrementPosition(this.board);

        // Replay all moves to build history
        for (const notation of moves) {
            const legalMoves = await this.engine.getLegalMoves(this.board);
            const parsed = await this.engine.parseMove(this.board, notation);
            if (!parsed) {
                console.warn('loadGame: could not parse move:', notation);
                break;
            }

            const move = legalMoves.find(m =>
                m.from_xor_to === parsed.from_xor_to && m.captures === parsed.captures
            );
            if (!move) {
                console.warn('loadGame: illegal move:', notation);
                break;
            }

            this.history.push({
                board: { ...this.board },
                move: move,
                notation: move.notation
            });

            const newBoard = await this.engine.makeMove(this.board, move);
            this.board = newBoard;
            this._incrementPosition(newBoard);
        }

        // Rewind: move entire history to redo stack and reset to initial
        this.redoStack = this.history.reverse();
        this.history = [];

        // Reset to initial position
        this.board = await this.engine.getInitialBoard();
        this.positionCounts = new Map();
        this._incrementPosition(this.board);
        await this._updateLegalMoves();

        this.dispatchEvent(new CustomEvent('gameLoaded', {
            detail: { board: this.board, moveCount: this.redoStack.length }
        }));
    }

    /**
     * Parse a notation string and make the move if legal
     * @returns {boolean} true if the move was made
     */
    async parseAndMakeMove(notation) {
        const parsed = await this.engine.parseMove(this.board, notation);
        if (!parsed) return false;

        const legalMove = this.legalMoves.find(m =>
            m.from_xor_to === parsed.from_xor_to && m.captures === parsed.captures
        );
        if (!legalMove) return false;

        await this.makeMove(legalMove);
        return true;
    }

    /**
     * Get move history for display
     * @returns {{history: string[], redo: string[]}}
     */
    getMoveHistoryForDisplay() {
        const historyMoves = this.history.map(h => h.notation);
        const redoMoves = this.redoStack.slice().reverse().map(h => h.notation);
        return { history: historyMoves, redo: redoMoves };
    }

    /**
     * Check if undo/redo are available
     */
    getUndoRedoState() {
        return {
            canUndo: this.history.length > 0,
            canRedo: this.redoStack.length > 0
        };
    }

    /**
     * Get game notation (PGN-like)
     */
    getNotation() {
        let notation = '';
        for (let i = 0; i < this.history.length; i++) {
            const moveNum = Math.floor(i / 2) + 1;
            if (i % 2 === 0) {
                notation += `${moveNum}. `;
            }
            notation += this.history[i].notation + ' ';
        }
        return notation.trim();
    }

    /**
     * Probe DTM tablebase for current position
     */
    async getDTM() {
        return await this.engine.probeDTM(this.board);
    }

    /**
     * Set game over externally (e.g., draw accepted, resignation)
     */
    setGameOver(winner, reason) {
        this._setGameOver(winner, reason);
    }

    // --- Internal ---

    /**
     * Get a position key for repetition tracking.
     * The engine stores the board flipped so side-to-move is always "white"
     * internally, so the (white, black, kings) triple already encodes whose
     * turn it is — no need to include whiteToMove separately.
     */
    _positionKey(board) {
        return `${board.white},${board.black},${board.kings}`;
    }

    _incrementPosition(board) {
        const key = this._positionKey(board);
        const count = (this.positionCounts.get(key) || 0) + 1;
        this.positionCounts.set(key, count);
        return count;
    }

    _decrementPosition(board) {
        const key = this._positionKey(board);
        const count = (this.positionCounts.get(key) || 1) - 1;
        if (count <= 0) {
            this.positionCounts.delete(key);
        } else {
            this.positionCounts.set(key, count);
        }
    }

    async _updateLegalMoves() {
        this.legalMoves = await this.engine.getLegalMoves(this.board);

        // Check for game over: no legal moves
        if (this.legalMoves.length === 0) {
            this._setGameOver(
                this.board.whiteToMove ? 'black' : 'white',
                'sin jugadas'
            );
            this.dispatchEvent(new CustomEvent('legalMovesUpdated', {
                detail: { moves: this.legalMoves }
            }));
            return;
        }

        // Check for 60-move rule
        if (this.board.nReversible >= 60) {
            this._setGameOver('draw', '60 jugadas sin progreso');
            this.dispatchEvent(new CustomEvent('legalMovesUpdated', {
                detail: { moves: this.legalMoves }
            }));
            return;
        }

        // Check for threefold repetition
        const key = this._positionKey(this.board);
        if ((this.positionCounts.get(key) || 0) >= 3) {
            this._setGameOver('draw', 'triple repetición');
            this.dispatchEvent(new CustomEvent('legalMovesUpdated', {
                detail: { moves: this.legalMoves }
            }));
            return;
        }

        this.dispatchEvent(new CustomEvent('legalMovesUpdated', {
            detail: { moves: this.legalMoves }
        }));
    }

    _setGameOver(winner, reason) {
        this.gameOver = true;
        this.winner = winner;
        this.gameOverReason = reason;
        this.dispatchEvent(new CustomEvent('gameOver', {
            detail: { winner, reason }
        }));
    }
}

/**
 * Count set bits in a 32-bit integer
 */
function popcount(n) {
    n = n >>> 0;
    n = n - ((n >> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
    return (((n + (n >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
