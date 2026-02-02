/**
 * Game controller - manages game state, history, and coordinates UI with engine
 */

import { BoardUI } from './board-ui.js';
import { getEngine } from './engine-api.js';

export class GameController {
    constructor(canvas, statusElement = null) {
        this.boardUI = new BoardUI(canvas);
        this.engine = getEngine();
        this.statusElement = statusElement;

        // Game state
        this.history = [];           // Array of { board, move, notation }
        this.currentIndex = -1;      // Current position in history
        this.legalMoves = [];        // Current legal moves

        // Settings
        this.humanColor = 'white';   // 'white', 'black', or 'both' (human vs human)
        this.engineDepth = 20;
        this.engineNodes = 100000;
        this.autoPlay = true;        // Engine plays automatically

        // State flags
        this.isThinking = false;
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

        // Step-by-step capture state
        this.partialPath = [];       // Squares clicked so far in a multi-capture

        // Set up board click handler
        this.boardUI.onClick = (square) => this._handleSquareClick(square);
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
     * Start a new game
     */
    async newGame() {
        // If engine was playing white, switch so engine plays black
        // This prevents auto-start and lets the user make the first move
        if (this.humanColor === 'black') {
            this.humanColor = 'white';
            if (this.onModeChange) this.onModeChange(this.humanColor);
        }

        this.history = [];
        this.currentIndex = -1;
        this.gameOver = false;
        this.winner = null;
        this.partialPath = [];
        this.boardUI.setPartialPath([]);
        this.boardUI.setSelected(null);
        this.boardUI.clearLastMove();

        // Reset engine board
        const board = await this.engine.resetBoard();
        this._updateFromBoard(board);

        // Get legal moves
        await this._updateLegalMoves();

        this._updateStatus('New game started');

        // Engine never plays white at start (we switched above if needed)
        // So no need to trigger engine move here
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

        // Check for game over
        if (this.legalMoves.length === 0) {
            this.gameOver = true;
            const board = await this.engine.getBoard();
            this.winner = board.whiteToMove ? 'black' : 'white';
            this._updateStatus(`Game over: ${this.winner} wins!`);
            if (this.onGameOver) {
                this.onGameOver(this.winner, 'no moves');
            }
        }
    }

    /**
     * Handle square click - supports step-by-step capture input
     */
    async _handleSquareClick(square) {
        if (this.gameOver || this.isThinking) return;

        const board = await this.engine.getBoard();
        const isHumanTurn = this._isHumanTurn(board.whiteToMove);

        if (!isHumanTurn) return;

        // If we have a partial path, try to continue the capture sequence
        if (this.partialPath.length > 0) {
            const fromSquare = this.partialPath[0];

            // Find moves that match our partial path so far
            const matchingMoves = this._getMovesMatchingPartialPath();

            // Check if clicked square is a valid next step
            const validNextMoves = matchingMoves.filter(m => {
                const nextIdx = this.partialPath.length;
                return m.path && m.path.length > nextIdx && m.path[nextIdx] === square;
            });

            if (validNextMoves.length > 0) {
                // Extend the partial path
                this.partialPath.push(square);

                // Check if we've completed a unique move
                const completedMoves = validNextMoves.filter(m =>
                    m.path && m.path.length === this.partialPath.length
                );

                if (completedMoves.length === 1) {
                    // Unique move completed - make it
                    this.partialPath = [];
                    this.boardUI.setSelected(null);
                    await this._makeMove(completedMoves[0]);
                    return;
                }

                // Still multiple continuations or not at end yet - update highlights
                this._updatePartialPathHighlights();
                return;
            }

            // Clicked square is not a valid continuation
            // Check if clicking on a different piece to select
            if (this.boardUI.hasPieceToMove(square) && square !== fromSquare) {
                const movesFromSquare = this.legalMoves.filter(m => m.from === square);
                if (movesFromSquare.length > 0) {
                    this.partialPath = [square];
                    this.boardUI.setSelected(square);
                    this._updatePartialPathHighlights();
                    return;
                }
            }

            // Clear selection
            this._clearPartialPath();
            this.boardUI.setSelected(null);
            return;
        }

        // No partial path - try to select a piece
        if (this.boardUI.hasPieceToMove(square)) {
            const movesFromSquare = this.legalMoves.filter(m => m.from === square);
            if (movesFromSquare.length > 0) {
                this.partialPath = [square];
                this.boardUI.setSelected(square);
                this._updatePartialPathHighlights();
                return;
            }
        }

        // Clear selection
        this._clearPartialPath();
        this.boardUI.setSelected(null);
    }

    /**
     * Get moves that match the current partial path
     */
    _getMovesMatchingPartialPath() {
        if (this.partialPath.length === 0) return this.legalMoves;

        return this.legalMoves.filter(m => {
            if (!m.path || m.path.length < this.partialPath.length) return false;
            for (let i = 0; i < this.partialPath.length; i++) {
                if (m.path[i] !== this.partialPath[i]) return false;
            }
            return true;
        });
    }

    /**
     * Update board highlights for partial path - show next valid squares
     */
    _updatePartialPathHighlights() {
        const matchingMoves = this._getMovesMatchingPartialPath();
        const nextIdx = this.partialPath.length;

        // Create pseudo-moves for highlighting next valid squares
        const nextSquares = new Set();
        for (const m of matchingMoves) {
            if (m.path && m.path.length > nextIdx) {
                nextSquares.add(m.path[nextIdx]);
            }
        }

        // Create highlight moves with 'from' as current position and 'to' as next valid squares
        const currentSquare = this.partialPath[this.partialPath.length - 1];
        const highlightMoves = Array.from(nextSquares).map(sq => ({
            from: currentSquare,
            to: sq
        }));

        this.boardUI.setLegalMoves(highlightMoves);
        this.boardUI.setPartialPath(this.partialPath);
    }

    /**
     * Clear partial path and reset board highlights
     */
    _clearPartialPath() {
        this.partialPath = [];
        this.boardUI.setPartialPath([]);
        this.boardUI.setLegalMoves(this.legalMoves);
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
    async _makeMove(move, triggerAutoPlay = true) {
        // Store current position in history
        const prevBoard = await this.engine.getBoard();
        this.history.push({
            board: { ...prevBoard },
            move: move,
            notation: move.notation
        });
        this.currentIndex = this.history.length - 1;

        // Make the move
        const newBoard = await this.engine.makeMove(move);
        this._updateFromBoard(newBoard);
        this.partialPath = [];
        this.boardUI.setPartialPath([]);
        this.boardUI.setSelected(null);
        this.boardUI.setLastMove(move.from, move.to);

        // Callback
        if (this.onMove) {
            this.onMove(move, newBoard);
        }

        // Update legal moves
        await this._updateLegalMoves();

        // Update status
        const side = newBoard.whiteToMove ? 'White' : 'Black';
        this._updateStatus(`${side} to move`);

        // If game not over and it's engine's turn, make engine move
        if (triggerAutoPlay && !this.gameOver && this.autoPlay && !this._isHumanTurn(newBoard.whiteToMove)) {
            await this._engineMove();
        }
    }

    /**
     * Engine makes a move
     * @param {boolean} triggerAutoPlay - whether to auto-play after this move
     */
    async _engineMove(triggerAutoPlay = true) {
        if (this.gameOver || this.isThinking) return;

        this.isThinking = true;
        this._triggerAutoPlay = triggerAutoPlay;  // Store for use after search completes
        if (this.onThinkingStart) this.onThinkingStart();
        this._updateStatus('Engine thinking...');

        try {
            console.log('Starting search with depth:', this.engineDepth, 'nodes:', this.engineNodes);
            const result = await this.engine.search(this.engineDepth, this.engineNodes);
            console.log('Search result:', result);

            if (result.error) {
                console.error('Engine error:', result.error);
                this._updateStatus('Engine error: ' + result.error);
                return;
            }

            // Format score for display
            let scoreStr = '?';
            if (result.score !== undefined) {
                if (Math.abs(result.score) > 29000) {
                    const mateIn = Math.ceil((30000 - Math.abs(result.score)) / 2);
                    scoreStr = result.score > 0 ? `M${mateIn}` : `-M${mateIn}`;
                } else {
                    scoreStr = (result.score / 100).toFixed(2);
                }
            }

            // Format PV for display
            const pvStr = result.pv && result.pv.length > 0 ? result.pv.join(' ') : '';

            // Update status with basic info
            this._updateStatus(`Depth: ${result.depth || '?'}, Score: ${scoreStr}, Nodes: ${result.nodes || '?'}`);

            // Call search info callback with full details
            if (this.onSearchInfo) {
                this.onSearchInfo({
                    depth: result.depth,
                    score: result.score,
                    scoreStr: scoreStr,
                    nodes: result.nodes,
                    tbHits: result.tbHits,
                    pv: result.pv || [],
                    pvStr: pvStr
                });
            }

            // Make the move
            if (result.bestMove && result.bestMove.from_xor_to && result.bestMove.from_xor_to !== 0) {
                console.log('Making move:', result.bestMove);
                // Reset thinking state BEFORE making move, so recursive engine moves can proceed
                this.isThinking = false;
                if (this.onThinkingEnd) this.onThinkingEnd();
                await this._makeMove(result.bestMove, this._triggerAutoPlay);
                return;  // _makeMove handles the next engine move if needed
            } else {
                console.warn('No valid bestMove in result:', result.bestMove);
                this._updateStatus('No move found');
            }
        } catch (err) {
            console.error('Search exception:', err);
            this._updateStatus('Search error: ' + err.message);
        } finally {
            this.isThinking = false;
            if (this.onThinkingEnd) this.onThinkingEnd();
        }
    }

    /**
     * Undo last move
     */
    async undo() {
        if (this.history.length === 0 || this.isThinking) return;

        // Pop last move
        const last = this.history.pop();
        this.currentIndex = this.history.length - 1;

        // Restore board
        await this.engine.setBoard(
            last.board.white,
            last.board.black,
            last.board.kings,
            last.board.whiteToMove
        );

        const board = await this.engine.getBoard();
        this._updateFromBoard(board);
        this.partialPath = [];
        this.boardUI.setPartialPath([]);
        this.boardUI.setSelected(null);
        this.boardUI.clearLastMove();

        // Update last move highlight if there's still history
        if (this.history.length > 0) {
            const prevMove = this.history[this.history.length - 1].move;
            this.boardUI.setLastMove(prevMove.from, prevMove.to);
        }

        // Update legal moves
        await this._updateLegalMoves();

        // Reset game over state
        this.gameOver = false;
        this.winner = null;

        // Switch mode so human can play the side that's now to move
        // (unless in 2-player mode)
        if (this.humanColor !== 'both') {
            this.humanColor = board.whiteToMove ? 'white' : 'black';
            if (this.onModeChange) this.onModeChange(this.humanColor);
        }

        this._updateStatus('Move undone');
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
        this.currentIndex = -1;
        this.gameOver = false;
        this.winner = null;
        this.partialPath = [];
        this.boardUI.setPartialPath([]);
        this.boardUI.setSelected(null);
        this.boardUI.clearLastMove();

        // Set the position in the engine
        const board = await this.engine.setBoard(white, black, kings, whiteToMove);
        this._updateFromBoard(board);

        // Get legal moves
        await this._updateLegalMoves();

        this._updateStatus('Position set');

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
     * Set engine parameters
     */
    setEngineParams(depth, nodes) {
        this.engineDepth = depth;
        this.engineNodes = nodes;
    }

    /**
     * Set which side the human plays
     * If the human gives up their turn, trigger engine to play
     */
    async setHumanColor(color) {
        const previousColor = this.humanColor;
        this.humanColor = color;

        // If it's now the engine's turn (human gave up their turn), make engine move
        if (!this.gameOver && !this.isThinking && this.autoPlay) {
            const board = await this.engine.getBoard();
            if (!this._isHumanTurn(board.whiteToMove)) {
                await this._engineMove();
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
     * Get DTM for current position
     */
    async getDTM() {
        return await this.engine.probeDTM();
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
