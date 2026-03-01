/**
 * Game controller - manages game state, history, and coordinates UI with engine
 */

const _v = new URL(import.meta.url).searchParams.get('v') || '';
const _q = _v ? `?v=${_v}` : '';
const { BoardUI } = await import(`./board-ui.js${_q}`);
const { playMoveSound } = await import(`./sound.js${_q}`);

export class GameController {
    constructor(canvas, engine, statusElement = null) {
        this.boardUI = new BoardUI(canvas);
        this.engine = engine;
        this.statusElement = statusElement;

        // Game state
        this.history = [];           // Array of { board, move, notation }
        this.redoStack = [];         // Stack of undone moves for redo
        this.currentIndex = -1;      // Current position in history
        this.legalMoves = [];        // Current legal moves

        // Settings
        this.humanColor = 'white';   // 'white', 'black', or 'both' (human vs human)
        this.secondsPerMove = 3.0;   // Time budget per move
        this.secondsLeft = 0;        // Time bank (accumulates/drains)
        this.useBook = true;         // Use opening book
        this.autoPlay = true;        // Engine plays automatically
        this.ponderEnabled = false;  // Pondering: engine thinks during opponent's turn

        // Position tracking for threefold repetition
        this.positionCounts = new Map();  // key: "white,black,kings" → count

        // State flags
        this.state = 'idle';         // 'idle' | 'thinking' | 'pondering'
        this.gameOver = false;
        this.winner = null;          // 'white', 'black', or 'draw'

        // Callbacks
        this.onMove = null;          // (move, board) => void
        this.onGameOver = null;      // (winner, reason) => void
        this.onThinkingStart = null;
        this.onThinkingEnd = null;
        this.onStatusUpdate = null;
        this.onSearchInfo = null;    // (info) => void - called with search results
        this.onModeChange = null;    // (humanColor) => void - called when mode changes
        this.onTimeUpdate = null;    // (secondsLeft) => void - called when time bank changes
        this.onDrawOffer = null;     // async () => boolean - called when engine offers a draw

        // Current principal variation from search
        this.currentPV = [];

        // Flexible move input state
        this._selectedMask = 0;      // 32-bit mask of selected squares
        this.partialPath = [];       // kept for _makeMove animation compatibility
        this.clickedSquares = new Set();  // kept for _clearInputState compatibility

        // Set up board click handler
        this.boardUI.onClick = (square) => this._handleSquareClick(square);
    }

    /**
     * Whether the engine is busy (thinking or pondering).
     * Used by UI for button state checks.
     */
    get isThinking() {
        return this.state !== 'idle';
    }

    /**
     * Initialize game controller
     */
    async init() {
        // Wait for engine to be ready
        if (!this.engine.isReady) {
            await this.engine.init();
        }

        // Reset to initial position
        await this.newGame();
    }

    /**
     * Stop any running search and wait for it to finish.
     * The interrupted search result is discarded.
     */
    async abortSearch() {
        if (this.state === 'idle') return;
        this._aborting = true;
        this.engine.stopSearch();
        while (this.state !== 'idle') {
            await this._sleep(50);
        }
        this._aborting = false;
    }

    /**
     * Start a new game
     */
    async newGame() {
        await this.abortSearch();

        // If engine was playing white, switch so engine plays black
        // This prevents auto-start and lets the user make the first move
        if (this.humanColor === 'black') {
            this.humanColor = 'white';
            if (this.onModeChange) this.onModeChange(this.humanColor);
        }

        this.history = [];
        this.redoStack = [];
        this.currentIndex = -1;
        this.gameOver = false;
        this.winner = null;
        this.currentPV = [];
        this.positionCounts = new Map();
        this.secondsLeft = 0;
        this._notifyTime();
        this._clearInputState();
        this.boardUI.setSelected(null);
        this.boardUI.clearLastMove();

        // Reset engine board
        const board = await this.engine.resetBoard();
        this._updateFromBoard(board);

        // Count initial position
        this._incrementPosition(board);

        // Get legal moves
        await this._updateLegalMoves();

        this._updateStatus('Nueva partida');

        // Engine never plays white at start (we switched above if needed)
        // So no need to trigger engine move here
    }

    /**
     * Get a position key for repetition tracking.
     * The engine stores the board flipped so side-to-move is always "white"
     * internally, so the (white, black, kings) triple already encodes whose
     * turn it is — no need to include whiteToMove separately.
     */
    _positionKey(board) {
        return `${board.white},${board.black},${board.kings}`;
    }

    /**
     * Increment position count and return the new count
     */
    _incrementPosition(board) {
        const key = this._positionKey(board);
        const count = (this.positionCounts.get(key) || 0) + 1;
        this.positionCounts.set(key, count);
        return count;
    }

    /**
     * Decrement position count
     */
    _decrementPosition(board) {
        const key = this._positionKey(board);
        const count = (this.positionCounts.get(key) || 1) - 1;
        if (count <= 0) {
            this.positionCounts.delete(key);
        } else {
            this.positionCounts.set(key, count);
        }
    }

    /**
     * Update board UI from engine board state
     */
    _updateFromBoard(board) {
        this.boardUI.setPosition(board.white, board.black, board.kings, board.whiteToMove);
    }

    /**
     * Update legal moves from engine
     */
    async _updateLegalMoves() {
        this.legalMoves = await this.engine.getLegalMoves();
        this.boardUI.setLegalMoves(this.legalMoves);

        // Check for game over (no legal moves = loss for side to move)
        if (this.legalMoves.length === 0) {
            const board = await this.engine.getBoard();
            this._setGameOver(board.whiteToMove ? 'black' : 'white', 'sin jugadas');
            return;
        }

        // Check for draw conditions
        const board = await this.engine.getBoard();
        if (board.nReversible >= 60) {
            this._setGameOver('draw', '60 jugadas sin progreso');
            return;
        }
        const key = this._positionKey(board);
        if ((this.positionCounts.get(key) || 0) >= 3) {
            this._setGameOver('draw', 'triple repetición');
            return;
        }
    }

    /**
     * Set game over state and update status
     * @param {string} winner - 'white', 'black', or 'draw'
     * @param {string} reason - reason for game end (for callback)
     */
    _setGameOver(winner, reason) {
        this.gameOver = true;
        this.winner = winner;

        // Format status message
        if (winner === 'draw') {
            const msg = reason ? `¡Tablas — ${reason}!` : '¡Tablas!';
            this._updateStatus(msg);
        } else {
            const winnerName = winner === 'white' ? 'blancas' : 'negras';
            this._updateStatus(`¡Ganan ${winnerName}!`);
        }

        if (this.onGameOver) {
            this.onGameOver(winner, reason);
        }
    }

    // No _computeMoveMask needed — the engine provides move.mask directly.

    /**
     * Filter moves to those whose path starts with the current partialPath.
     * Used for sequential mode (clicking in path order, including repeated squares).
     */
    _filterByPathPrefix(moves) {
        if (this.partialPath.length === 0) return moves;
        return moves.filter(m => {
            if (!m.path || m.path.length < this.partialPath.length) return false;
            for (let i = 0; i < this.partialPath.length; i++) {
                if (m.path[i] !== this.partialPath[i]) return false;
            }
            return true;
        });
    }

    /**
     * Check if all moves represent the same engine move
     * (same from_xor_to and captures — just different path orderings)
     */
    _areAllSameEngineMove(moves) {
        if (moves.length <= 1) return true;
        const first = moves[0];
        return moves.every(m =>
            m.from_xor_to === first.from_xor_to && m.captures === first.captures
        );
    }

    /**
     * Get moves whose mask contains all selected squares.
     */
    _getMovesMatchingSelection() {
        const sel = this._selectedMask;
        return this.legalMoves.filter(m => ((m.mask & sel) >>> 0) === sel);
    }

    /**
     * Try to extend partialPath for in-order visualization.
     * If partialPath is empty and the square is a piece's from, start the path.
     * Otherwise extend if the square matches the next path entry in any matching move.
     */
    _tryExtendPartialPath(square, matchingMoves) {
        if (this.partialPath.length === 0) {
            if (matchingMoves.some(m => m.from === square)) {
                this.partialPath = [square];
            }
            return;
        }
        const nextIdx = this.partialPath.length;
        for (const m of matchingMoves) {
            if (m.path && m.path.length > nextIdx && m.path[nextIdx] === square) {
                this.partialPath.push(square);
                return;
            }
        }
        // Out of order — keep existing partial path, don't extend
    }

    /**
     * Update board highlights:
     * - Red outlines on selected squares
     * - Orange outlines on unselected squares in matching moves' masks
     * - If partialPath is active, show piece moving and captures fading
     */
    _updateHighlights() {
        let matchingMoves = this._getMovesMatchingSelection();

        // In sequential mode, also filter by path prefix
        if (this.partialPath.length > 0) {
            const prefixFiltered = this._filterByPathPrefix(matchingMoves);
            if (prefixFiltered.length > 0) matchingMoves = prefixFiltered;
        }

        // Collect all unselected squares from matching move masks
        let validMask = 0;
        for (const m of matchingMoves) {
            validMask |= m.mask;
        }
        validMask = (validMask & ~this._selectedMask) >>> 0;

        // In sequential mode, also include next path positions
        // (may be already-selected squares that need to be clicked again)
        if (this.partialPath.length > 0) {
            const nextIdx = this.partialPath.length;
            for (const m of matchingMoves) {
                if (m.path && m.path.length > nextIdx) {
                    validMask |= (1 << (m.path[nextIdx] - 1));
                }
            }
            validMask = validMask >>> 0;
        }

        const validSquares = [];
        for (let i = 0; i < 32; i++) {
            if (validMask & (1 << i)) validSquares.push(i + 1);
        }

        // Selected squares for red outlines
        const selectedSquares = [];
        for (let i = 0; i < 32; i++) {
            if (this._selectedMask & (1 << i)) selectedSquares.push(i + 1);
        }

        // Batch updates to avoid multiple renders
        this.boardUI.legalMoves = [];  // suppress default green highlights
        this.boardUI.outOfOrderClicks = selectedSquares;  // red outlines
        this.boardUI.flexibleHighlights = validSquares;   // orange outlines
        if (this.partialPath.length > 0) {
            this.boardUI.selectedSquare = this.partialPath[0];
            this.boardUI.partialPath = this.partialPath;
        } else {
            this.boardUI.selectedSquare = null;
            this.boardUI.partialPath = [];
        }
        this.boardUI.render();
    }

    /**
     * Handle square click — flexible move input.
     *
     * Maintains a set of selected squares (as a 32-bit mask). Each click adds
     * a square. If exactly one engine move matches, execute it. If multiple
     * match, wait. If none match, restart with just this square; if still
     * nothing, clear.
     */
    async _handleSquareClick(square) {
        if (this.gameOver) return;

        // Stop pondering so the worker is free for getBoard/makeMove calls
        if (this.state === 'pondering') {
            await this.abortSearch();
        }

        if (this.state === 'thinking') return;

        const board = await this.engine.getBoard();
        if (!this._isHumanTurn(board.whiteToMove)) return;

        const bit = (1 << (square - 1)) >>> 0;

        // Already selected this square — try sequential mode (for repeated landing squares)
        if (this._selectedMask & bit) {
            if (this.partialPath.length > 0) {
                const pathMoves = this._filterByPathPrefix(this._getMovesMatchingSelection());
                const nextIdx = this.partialPath.length;
                const extended = pathMoves.filter(m =>
                    m.path && m.path.length > nextIdx && m.path[nextIdx] === square
                );
                if (extended.length > 0) {
                    this.partialPath.push(square);
                    if (this._areAllSameEngineMove(extended)) {
                        const move = extended[0];
                        const animate = move.path && move.path.length > 2 && this.partialPath.length < move.path.length;
                        this._clearInputState(false);
                        await this._makeMove(move, true, animate);
                        return;
                    }
                    this._updateHighlights();
                    return;
                }
            }
            return;
        }

        // Try adding this square to the selection
        const newMask = (this._selectedMask | bit) >>> 0;
        this._selectedMask = newMask;
        let matchingMoves = this._getMovesMatchingSelection();

        if (matchingMoves.length === 0) {
            // No moves match with full set — restart with just this square
            this._selectedMask = bit;
            this.partialPath = [];
            matchingMoves = this._getMovesMatchingSelection();

            if (matchingMoves.length === 0) {
                // This square is in no move at all — clear everything
                this._clearInputState();
                return;
            }
        }

        // Try to extend partial path for in-order visualization
        this._tryExtendPartialPath(square, matchingMoves);

        // Exactly one engine move (possibly multiple path orderings) → execute
        if (this._areAllSameEngineMove(matchingMoves)) {
            const move = matchingMoves[0];
            const animate = move.path && move.path.length > 2 && this.partialPath.length < move.path.length;
            this._clearInputState(false);
            await this._makeMove(move, true, animate);
            return;
        }

        // Multiple moves still possible — show highlights and wait
        this._updateHighlights();
    }

    /**
     * Clear all input state and reset board highlights
     * @param {boolean} render - if false, skip re-rendering (caller will render)
     */
    _clearInputState(render = true) {
        this._selectedMask = 0;
        this.partialPath = [];
        this.clickedSquares = new Set();
        this.boardUI.selectedSquare = null;
        this.boardUI.partialPath = [];
        this.boardUI.outOfOrderClicks = [];
        this.boardUI.flexibleHighlights = [];
        this.boardUI.lastMove = null;
        if (render) {
            this.boardUI.setLegalMoves(this.legalMoves);
        } else {
            this.boardUI.legalMoves = this.legalMoves;
        }
    }

    /**
     * Check if it's human's turn
     */
    _isHumanTurn(whiteToMove) {
        if (this.humanColor === 'both') return true;
        if (this.humanColor === 'white') return whiteToMove;
        return !whiteToMove;
    }

    /**
     * Make a move (human or engine)
     * @param {boolean} triggerAutoPlay - whether to auto-play engine's response
     */
    async _makeMove(move, triggerAutoPlay = true, animate = false) {
        // Stop pondering if active (human just moved)
        if (this.state === 'pondering') {
            await this.abortSearch();
        }

        // Clear redo stack and PV since we're making a new move
        this.redoStack = [];
        this.currentPV = [];

        // Store current position in history
        const prevBoard = await this.engine.getBoard();
        this.history.push({
            board: { ...prevBoard },
            move: move,
            notation: move.notation
        });
        this.currentIndex = this.history.length - 1;

        // Animate multi-capture moves step by step (only for engine moves)
        if (animate && move.path && move.path.length > 2) {
            // Show each jump with a delay
            for (let i = 1; i < move.path.length; i++) {
                const partialPath = move.path.slice(0, i + 1);
                this.boardUI.setPartialPath(partialPath);
                this.boardUI.render();
                playMoveSound();
                await this._sleep(200);
            }
            // Keep partial path set until board is updated to avoid flash
        } else {
            playMoveSound();
        }

        // Make the move
        const newBoard = await this.engine.makeMove(move);
        // Track position for repetition detection
        this._incrementPosition(newBoard);
        // Clear input state before updating board (no render yet)
        this._selectedMask = 0;
        this.partialPath = [];
        this.clickedSquares = new Set();
        this.boardUI.partialPath = [];
        this.boardUI.outOfOrderClicks = [];
        this.boardUI.flexibleHighlights = [];
        this._updateFromBoard(newBoard);
        this.boardUI.setSelected(null);

        // Callback
        if (this.onMove) {
            this.onMove(move, newBoard);
        }

        // Update legal moves (also checks for game over)
        // setLegalMoves clears lastMove, so set it after
        await this._updateLegalMoves();
        this.boardUI.setLastMove(move.from, move.to);

        // Update status only if game is not over
        if (!this.gameOver) {
            const side = newBoard.whiteToMove ? 'blancas' : 'negras';
            this._updateStatus(`Mueven ${side}`);
        }

        // If game not over and it's engine's turn, make engine move
        if (triggerAutoPlay && !this.gameOver && this.autoPlay && !this._isHumanTurn(newBoard.whiteToMove)) {
            await this._engineMove();
        } else if (this.ponderEnabled && !this.gameOver) {
            // Start pondering if enabled and game isn't over
            await this._startPondering();
        }
    }

    /**
     * Sleep helper for animations
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Engine makes a move
     * @param {boolean} triggerAutoPlay - whether to auto-play after this move
     */
    async _engineMove(triggerAutoPlay = true) {
        if (this.gameOver || this.state !== 'idle') return;

        this.state = 'thinking';
        this._triggerAutoPlay = triggerAutoPlay;  // Store for use after search completes
        if (this.onThinkingStart) this.onThinkingStart();
        this._updateStatus('Motor pensando...');

        try {
            // Time control: bank time, compute soft/hard limits
            this.secondsLeft += this.secondsPerMove;
            const softTime = this.secondsLeft / 8;
            const hardTime = this.secondsLeft;
            console.log('Starting search: softTime=', softTime.toFixed(2), 'hardTime=', hardTime.toFixed(2));
            const startTime = Date.now();
            const startSecondsLeft = this.secondsLeft;

            // Live countdown timer (updates display every 100ms)
            const countdownTimer = setInterval(() => {
                const elapsed = (Date.now() - startTime) / 1000;
                const projected = Math.max(0, startSecondsLeft - elapsed);
                if (this.onTimeUpdate) this.onTimeUpdate(projected);
            }, 100);

            // Progress callback for iterative deepening updates
            const onProgress = (progressResult) => {
                this._reportSearchInfo(progressResult);
            };

            let result;
            try {
                result = await this.engine.search(softTime, hardTime, onProgress);
            } finally {
                clearInterval(countdownTimer);
            }
            console.log('Search result:', result);

            // Debit elapsed time from bank
            const elapsedSeconds = (Date.now() - startTime) / 1000;
            this.secondsLeft -= elapsedSeconds;
            if (this.secondsLeft < 0.1) this.secondsLeft = 0.1;
            this._notifyTime();

            // Abort if search was cancelled (new game, edit mode, etc.)
            if (this._aborting) return;

            if (result.error) {
                console.error('Engine error:', result.error);
                this._updateStatus('Error del motor: ' + result.error);
                return;
            }

            // Report final result
            this._reportSearchInfo(result);

            // Ensure minimum 200ms so user can see what happened
            const elapsed = Date.now() - startTime;
            if (elapsed < 200) {
                await this._sleep(200 - elapsed);
            }

            // Draw offer: proven draw (score in [-10000, +10000]), not book, 4 or fewer pieces
            if (this.onDrawOffer && Math.abs(result.score) <= 10000 && !result.book) {
                const board = await this.engine.getBoard();
                if (board.pieceCount <= 4) {
                    const accepted = await this.onDrawOffer();
                    if (accepted) {
                        this.state = 'idle';
                        if (this.onThinkingEnd) this.onThinkingEnd();
                        this._setGameOver('draw', 'tablas aceptadas');
                        return;
                    }
                }
            }

            // Make the move
            if (result.bestMove && result.bestMove.from_xor_to && result.bestMove.from_xor_to !== 0) {
                console.log('Making move:', result.bestMove);
                // Reset thinking state BEFORE making move, so recursive engine moves can proceed
                this.state = 'idle';
                if (this.onThinkingEnd) this.onThinkingEnd();
                await this._makeMove(result.bestMove, this._triggerAutoPlay, true);  // animate=true for engine moves
                return;  // _makeMove handles the next engine move if needed
            } else {
                console.warn('No valid bestMove in result:', result.bestMove);
                this._updateStatus('No se encontró jugada');
            }
        } catch (err) {
            console.error('Search exception:', err);
            this._updateStatus('Error de búsqueda: ' + err.message);
        } finally {
            this.state = 'idle';
            if (this.onThinkingEnd) this.onThinkingEnd();
        }
    }

    /**
     * Start pondering (background search of current position).
     * Runs an open-ended search that reports progress but never plays a move.
     */
    async _startPondering() {
        if (this.gameOver || this.state !== 'idle') return;

        this.state = 'pondering';
        if (this.onThinkingStart) this.onThinkingStart();

        try {
            const onProgress = (result) => {
                this._reportSearchInfo(result);
            };

            // analyzeMode=true (search even w/ 1 move)
            // ponderMode: full window for all root moves only when engine is playing (not 2-player)
            const ponderMode = this.humanColor !== 'both';
            const result = await this.engine.search(999999, 999999, onProgress, true, ponderMode);

            // Abort if search was cancelled
            if (this._aborting) return;

            if (result && !result.error) {
                this._reportSearchInfo(result);
            }
        } catch (err) {
            if (!this._aborting) {
                console.error('Ponder error:', err);
            }
        } finally {
            this.state = 'idle';
            if (this.onThinkingEnd) this.onThinkingEnd();
        }
    }

    /**
     * Report search info (used for both progress updates and final result)
     */
    _reportSearchInfo(result) {
        this.currentPV = result.pv || [];

        // Format score for display
        let scoreStr = '?';
        if (result.score !== undefined) {
            if (Math.abs(result.score) > 29000) {
                // Mate scores
                const mateIn = Math.ceil((30000 - Math.abs(result.score)) / 2);
                scoreStr = result.score > 0 ? `M${mateIn}` : `-M${mateIn}`;
            } else if (Math.abs(result.score) <= 10000) {
                // Proven draw: score in [-10000, +10000]
                const val = (result.score / 100).toFixed(2);
                scoreStr = (result.score >= 0 ? '+' : '') + val + '(tablas)';
            } else {
                // Undecided: strip ±10000 offset
                const raw = result.score > 0 ? result.score - 10000 : result.score + 10000;
                const val = (raw / 100).toFixed(2);
                scoreStr = raw > 0 ? `+${val}` : val;
            }
        }

        // Format PV for display
        const pvStr = result.pv && result.pv.length > 0 ? result.pv.join(' ') : '';

        // Call search info callback with full details
        if (this.onSearchInfo) {
            const info = {
                depth: result.depth,
                score: result.score,
                scoreStr: scoreStr,
                nodes: result.nodes,
                nps: result.nps || 0,
                tbHits: result.tbHits,
                pv: result.pv || [],
                pvStr: pvStr
            };
            if (result.phase) info.phase = result.phase;
            if (result.rootMoves) info.rootMoves = result.rootMoves;
            if (result.bookMoves) info.bookMoves = result.bookMoves;
            this.onSearchInfo(info);
        }
    }

    /**
     * Undo last move
     */
    async undo() {
        if (this.history.length === 0) return;
        await this.abortSearch();
        this.currentPV = [];

        // Decrement position count for current position before undoing
        const currentBoard = await this.engine.getBoard();
        this._decrementPosition(currentBoard);

        // Pop last move and save it for redo
        const last = this.history.pop();
        this.redoStack.push(last);
        this.currentIndex = this.history.length - 1;

        // Restore board (including reversible move counter)
        await this.engine.setBoard(
            last.board.white,
            last.board.black,
            last.board.kings,
            last.board.whiteToMove,
            last.board.nReversible || 0
        );

        const board = await this.engine.getBoard();
        this._updateFromBoard(board);
        this._clearInputState();
        this.boardUI.setSelected(null);

        // Update legal moves (setLegalMoves clears lastMove, so set it after)
        await this._updateLegalMoves();

        // Restore last move highlight if there's still history
        if (this.history.length > 0) {
            const prevMove = this.history[this.history.length - 1].move;
            this.boardUI.setLastMove(prevMove.from, prevMove.to);
        }

        // Reset game over state
        this.gameOver = false;
        this.winner = null;

        // After taking back an engine move, the human "switches seats" and
        // takes over the side that is now to move.
        if (this.humanColor !== 'both') {
            this.humanColor = board.whiteToMove ? 'white' : 'black';
            if (this.onModeChange) this.onModeChange(this.humanColor);
        }

        this._updateStatus('Jugada deshecha');

        if (this.ponderEnabled && !this.gameOver) {
            await this._startPondering();
        }
    }

    /**
     * Redo last undone move
     */
    async redo() {
        if (this.redoStack.length === 0) return;
        if (this.state === 'pondering') {
            await this.abortSearch();
        } else if (this.state === 'thinking') {
            return;
        }

        this.currentPV = [];

        // Pop move from redo stack
        const entry = this.redoStack.pop();

        // Re-add to history
        this.history.push(entry);
        this.currentIndex = this.history.length - 1;

        // Make the move on the engine
        const newBoard = await this.engine.makeMove(entry.move);
        // Track position for repetition detection
        this._incrementPosition(newBoard);
        this._updateFromBoard(newBoard);
        this._clearInputState();
        this.boardUI.setSelected(null);

        // Update legal moves (setLegalMoves clears lastMove, so set it after)
        await this._updateLegalMoves();
        this.boardUI.setLastMove(entry.move.from, entry.move.to);

        // Switch mode so human can play the side that's now to move
        if (this.humanColor !== 'both') {
            this.humanColor = newBoard.whiteToMove ? 'white' : 'black';
            if (this.onModeChange) this.onModeChange(this.humanColor);
        }

        this._updateStatus('Jugada rehecha');

        // If pondering is enabled, start pondering the restored position
        if (this.ponderEnabled && !this.gameOver) {
            await this._startPondering();
        }
    }

    /**
     * Set up a custom board position
     * @param {number} white - White pieces bitboard
     * @param {number} black - Black pieces bitboard
     * @param {number} kings - Kings bitboard (0 for no kings)
     * @param {boolean} whiteToMove - True if white to move
     */
    async setPosition(white, black, kings, whiteToMove) {
        this.history = [];
        this.redoStack = [];
        this.currentIndex = -1;
        this.gameOver = false;
        this.winner = null;
        this.currentPV = [];
        this.positionCounts = new Map();
        this._clearInputState();
        this.boardUI.setSelected(null);
        this.boardUI.clearLastMove();

        // Set the position in the engine
        const board = await this.engine.setBoard(white, black, kings, whiteToMove);
        this._updateFromBoard(board);

        // Count initial position
        this._incrementPosition(board);

        // Get legal moves
        await this._updateLegalMoves();

        this._updateStatus('Posición establecida');

        // If engine's turn and autoPlay is on, start thinking
        if (!this.gameOver && this.autoPlay && !this._isHumanTurn(whiteToMove)) {
            await this._engineMove();
        }
    }

    /**
     * Force engine to move now and assign engine to current side
     * Behavior matches 'm' command in play.cpp:
     * - Engine takes over the current side
     * - Human plays the other side
     * - Auto-play continues from there
     */
    async engineMoveNow() {
        if (this.gameOver || this.isThinking) return;

        // Get current board to know whose turn it is
        const board = await this.engine.getBoard();

        // Assign engine to current side (like 'm' command in play.cpp)
        // humanColor becomes the OTHER side
        this.humanColor = board.whiteToMove ? 'black' : 'white';
        if (this.onModeChange) this.onModeChange(this.humanColor);

        // Now make the engine move with auto-play enabled
        // Since we've assigned engine to current side, it will continue playing that side
        await this._engineMove(true);
    }

    /**
     * Play the top PV move while pondering
     */
    async playPonderMove() {
        if (this.state !== 'pondering' || this.currentPV.length === 0) return false;
        const notation = this.currentPV[0];
        await this.abortSearch();

        // Switch human to the other side (engine takes over the current side)
        const board = await this.engine.getBoard();
        this.humanColor = board.whiteToMove ? 'black' : 'white';
        if (this.onModeChange) this.onModeChange(this.humanColor);

        return await this.inputMove(notation);
    }

    /**
     * Set seconds per move
     */
    setSecondsPerMove(seconds) {
        this.secondsPerMove = seconds;
    }

    setUseBook(useBook) {
        this.useBook = useBook;
        this.engine.setUseBook(useBook);
    }

    /**
     * Stop the current search (fire-and-forget, for the Stop button).
     * The search will finish with the best result found so far and play it.
     */
    stopSearch() {
        if (this.engine) {
            this.engine.stopSearch();
        }
    }

    /**
     * Set which side the human plays
     * If the human gives up their turn, trigger engine to play
     */
    async setHumanColor(color) {
        const previousColor = this.humanColor;
        this.humanColor = color;
        this.secondsLeft = 0;
        this._notifyTime();

        // Stop pondering if active (mode change may require engine to think)
        if (this.state === 'pondering') {
            await this.abortSearch();
        }

        // If it's now the engine's turn (human gave up their turn), make engine move
        if (!this.gameOver && this.state === 'idle' && this.autoPlay) {
            const board = await this.engine.getBoard();
            if (!this._isHumanTurn(board.whiteToMove)) {
                await this._engineMove();
            } else if (this.ponderEnabled) {
                await this._startPondering();
            }
        }
    }

    /**
     * Toggle board orientation
     */
    flipBoard() {
        this.boardUI.setFlipped(!this.boardUI.flipped);
    }

    /**
     * Get game notation (PGN-like)
     */
    getGameNotation() {
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
     * Get move history with redo information for display
     * Returns { history: string[], redo: string[] }
     */
    getMoveHistoryForDisplay() {
        const historyMoves = this.history.map(h => h.notation);
        // Redo stack is in reverse order (last undone is first to redo)
        const redoMoves = this.redoStack.slice().reverse().map(h => h.notation);
        return { history: historyMoves, redo: redoMoves };
    }

    /**
     * Check if undo/redo are available
     * Returns { canUndo: boolean, canRedo: boolean }
     */
    getUndoRedoState() {
        return {
            canUndo: this.history.length > 0 && (this.state !== 'thinking'),
            canRedo: this.redoStack.length > 0 && (this.state !== 'thinking')
        };
    }

    /**
     * Get DTM for current position
     */
    async getDTM() {
        return await this.engine.probeDTM();
    }

    /**
     * Notify listener of time bank change
     */
    _notifyTime() {
        if (this.onTimeUpdate) this.onTimeUpdate(this.secondsLeft);
    }

    /**
     * Update status display
     */
    _updateStatus(message) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
        }
        if (this.onStatusUpdate) {
            this.onStatusUpdate(message);
        }
    }

    /**
     * Resize board
     */
    resize(size) {
        this.boardUI.resize(size);
    }

    /**
     * Get current board state
     */
    async getBoard() {
        return await this.engine.getBoard();
    }

    /**
     * Load a recorded game for replay/analysis.
     * Replays all moves to build history, then rewinds to the start
     * so the user can step through the game with redo.
     * @param {string[]} moves - Array of move notations
     */
    async loadGame(moves) {
        await this.abortSearch();

        // Reset to initial position
        this.history = [];
        this.redoStack = [];
        this.currentIndex = -1;
        this.gameOver = false;
        this.winner = null;
        this.currentPV = [];
        this.positionCounts = new Map();
        this.secondsLeft = 0;
        this._notifyTime();
        this._clearInputState();
        this.boardUI.setSelected(null);
        this.boardUI.clearLastMove();

        const board = await this.engine.resetBoard();
        this._incrementPosition(board);

        // Replay all moves to build history
        for (const notation of moves) {
            const legalMoves = await this.engine.getLegalMoves();
            const parsed = await this.engine.parseMove(notation);
            if (!parsed) {
                console.warn('loadGame: could not parse move:', notation);
                break;
            }

            // Find matching legal move
            const move = legalMoves.find(m =>
                m.from_xor_to === parsed.from_xor_to && m.captures === parsed.captures
            );
            if (!move) {
                console.warn('loadGame: illegal move:', notation);
                break;
            }

            // Record in history
            const prevBoard = await this.engine.getBoard();
            this.history.push({
                board: { ...prevBoard },
                move: move,
                notation: move.notation
            });

            // Make the move on the engine
            const newBoard = await this.engine.makeMove(move);
            this._incrementPosition(newBoard);
        }

        // Now rewind: move entire history to redo stack and reset to initial position
        // Redo stack is in reverse order (last item is first to redo)
        this.redoStack = this.history.reverse();
        this.history = [];
        this.currentIndex = -1;

        // Reset engine to initial position
        await this.engine.resetBoard();
        this.positionCounts = new Map();
        const initialBoard = await this.engine.getBoard();
        this._incrementPosition(initialBoard);

        this._updateFromBoard(initialBoard);
        await this._updateLegalMoves();

        this._updateStatus('Partida cargada');
    }

    /**
     * Input move from notation
     */
    async inputMove(notation) {
        if (this.gameOver || this.isThinking) return false;

        const move = await this.engine.parseMove(notation);
        if (!move) return false;

        // Check if move is legal
        const legalMove = this.legalMoves.find(m =>
            m.from_xor_to === move.from_xor_to && m.captures === move.captures
        );

        if (!legalMove) return false;

        await this._makeMove(legalMove);
        return true;
    }
}
