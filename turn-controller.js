/**
 * Turn controller — the ONE place that decides "should the engine play now?"
 * Owns humanColor. Reacts to GameState and SearchManager events.
 *
 * Key invariant: humanColor is only mutated inside this class.
 * Key invariant: _evaluateTurn() is the single decision point.
 */

export class TurnController extends EventTarget {
    /**
     * @param {GameState} gameState
     * @param {SearchManager} searchManager
     */
    constructor(gameState, searchManager) {
        super();
        this.gameState = gameState;
        this.searchManager = searchManager;

        // State
        this.humanColor = 'white';  // 'white'|'black'|'both'
        this.autoPlay = true;
        this.ponderEnabled = false;
        this.drawDeclined = false;
    }

    /**
     * Wire event listeners. Call once after construction.
     */
    init() {
        this.gameState.addEventListener('newGame', (e) => this._onNewGame(e.detail));
        this.gameState.addEventListener('move', (e) => this._onMove(e.detail));
        this.gameState.addEventListener('undo', (e) => this._onUndo(e.detail));
        this.gameState.addEventListener('redo', (e) => this._onRedo(e.detail));
        this.gameState.addEventListener('positionChanged', (e) => this._onPositionChanged(e.detail));
        this.gameState.addEventListener('gameLoaded', (e) => this._onGameLoaded(e.detail));
    }

    /**
     * Set human color. Only external entry point for changing it.
     */
    setHumanColor(color) {
        this.humanColor = color;
        this.searchManager.resetTimeBank();
        this.dispatchEvent(new CustomEvent('humanColorChanged', {
            detail: { humanColor: color }
        }));
        this._evaluateTurn();
    }

    /**
     * Force engine to move now: assign engine to current side,
     * human gets the other side.
     */
    async engineMoveNow() {
        if (this.gameState.gameOver || this.searchManager.isSearching) return;

        const board = this.gameState.board;
        this.humanColor = board.whiteToMove ? 'black' : 'white';
        this.dispatchEvent(new CustomEvent('humanColorChanged', {
            detail: { humanColor: this.humanColor }
        }));

        await this._startEngineSearch();
    }

    /**
     * Play the top PV move while pondering.
     * Abort ponder, switch color, play the PV move.
     * @returns {boolean} whether a move was made
     */
    async playPonderMove() {
        if (this.searchManager.state !== 'pondering' || this.searchManager.currentPV.length === 0) {
            return false;
        }

        const notation = this.searchManager.currentPV[0];
        await this.searchManager.abort();

        // Resolve the notation to a legal move object
        const parsed = await this.gameState.engine.parseMove(this.gameState.board, notation);
        if (!parsed) return false;
        const legalMove = this.gameState.legalMoves.find(m =>
            m.from_xor_to === parsed.from_xor_to && m.captures === parsed.captures
        );
        if (!legalMove) return false;

        // Switch human to other side
        const board = this.gameState.board;
        this.humanColor = board.whiteToMove ? 'black' : 'white';
        this.dispatchEvent(new CustomEvent('humanColorChanged', {
            detail: { humanColor: this.humanColor }
        }));

        // Dispatch engineMove so main.js handles animation + sound
        await new Promise(resolve => {
            this.dispatchEvent(new CustomEvent('engineMove', {
                detail: { move: legalMove, resolve }
            }));
        });
        return true;
    }

    /**
     * Set pondering enabled/disabled
     */
    setPonderEnabled(enabled) {
        this.ponderEnabled = enabled;
        if (!enabled && this.searchManager.state === 'pondering') {
            this.searchManager.abort();
        } else if (enabled) {
            this._evaluateTurn();
        }
    }

    /**
     * Synchronous check: is it the human's turn?
     */
    isHumanTurn() {
        if (this.humanColor === 'both') return true;
        if (this.humanColor === 'white') return this.gameState.board.whiteToMove;
        return !this.gameState.board.whiteToMove;
    }

    // --- Event handlers ---

    _onNewGame({ board }) {
        this.drawDeclined = false;
        // Don't auto-evaluate here; main.js will call setHumanColor after newGame
    }

    _onMove({ move, board, gameOver }) {
        if (gameOver) return;
        this._evaluateTurn();
    }

    _onUndo({ board }) {
        this._switchSeat(board);
        this._evaluateTurn();
    }

    _onRedo({ board }) {
        this._switchSeat(board);
        this._evaluateTurn();
    }

    _onPositionChanged({ board }) {
        this._evaluateTurn();
    }

    _onGameLoaded({ board }) {
        // Game loaded for analysis — don't auto-play
    }

    // --- Core decision ---

    /**
     * Single decision point: should the engine start searching or pondering?
     */
    async _evaluateTurn() {
        if (this.gameState.gameOver) return;
        if (this.searchManager.isSearching) return;

        if (!this.isHumanTurn() && this.autoPlay) {
            await this._startEngineSearch();
        } else if (this.ponderEnabled) {
            await this._startPondering();
        }
    }

    /**
     * Start an engine search for the current position.
     */
    async _startEngineSearch() {
        const board = this.gameState.board;
        const searchStart = Date.now();
        const result = await this.searchManager.startSearch(board);

        if (!result || result.error) {
            if (result && result.error) {
                console.error('Engine error:', result.error);
            }
            return;
        }

        // Ensure minimum 200ms so user can see what happened
        const elapsed = Date.now() - searchStart;
        if (elapsed < 200) {
            await new Promise(r => setTimeout(r, 200 - elapsed));
        }

        // Check draw offer conditions
        if (!this.drawDeclined && !result.book) {
            if (this._shouldOfferDraw(result, board)) {
                const accepted = await this._offerDraw();
                if (accepted) {
                    this.gameState.setGameOver('draw', 'tablas aceptadas');
                    return;
                }
            }
        }

        // Dispatch engineMove event — main.js handles animation/sound then commits
        if (result.bestMove && result.bestMove.from_xor_to && result.bestMove.from_xor_to !== 0) {
            console.log('Engine move:', result.bestMove);
            // Use a Promise so we can await main.js finishing the animation+commit
            await new Promise(resolve => {
                this.dispatchEvent(new CustomEvent('engineMove', {
                    detail: { move: result.bestMove, resolve }
                }));
            });
            // _onMove will be called, which calls _evaluateTurn for the next move
        } else {
            console.warn('No valid bestMove in result:', result.bestMove);
        }
    }

    /**
     * Start pondering the current position
     */
    async _startPondering() {
        const board = this.gameState.board;
        const fullWindow = this.humanColor !== 'both';
        await this.searchManager.startPondering(board, { fullWindow });
    }

    /**
     * After undo/redo, switch humanColor to the side now to move
     */
    _switchSeat(board) {
        if (this.humanColor !== 'both') {
            this.humanColor = board.whiteToMove ? 'white' : 'black';
            this.dispatchEvent(new CustomEvent('humanColorChanged', {
                detail: { humanColor: this.humanColor }
            }));
        }
    }

    /**
     * Check if engine should offer a draw
     */
    _shouldOfferDraw(result, board) {
        const provenDraw = Math.abs(result.score) <= 10000 && board.pieceCount <= 4;
        const likelyDraw = Math.abs(result.score) <= 1000 && board.nReversible >= 10;
        return provenDraw || likelyDraw;
    }

    /**
     * Offer a draw to the human via event.
     * @returns {Promise<boolean>} whether the draw was accepted
     */
    _offerDraw() {
        return new Promise(resolve => {
            this.dispatchEvent(new CustomEvent('drawOffer', {
                detail: {
                    resolve: (accepted) => {
                        if (!accepted) this.drawDeclined = true;
                        resolve(accepted);
                    }
                }
            }));
        });
    }
}
