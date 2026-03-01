/**
 * Search lifecycle manager — owns thinking/pondering state, time bank, PV.
 * Never touches game state directly.
 * Uses a generation counter + promise-based abort (no polling).
 */

export class SearchManager extends EventTarget {
    constructor(engine) {
        super();
        this.engine = engine;

        // Search state
        this.state = 'idle';            // 'idle'|'thinking'|'pondering'
        this.currentPV = [];            // Current principal variation (notation strings)
        this._searchGeneration = 0;     // Monotonic counter for stale-result rejection
        this._currentSearchDone = null; // Resolves when current search finishes
        this._resolveDone = null;       // Resolver for _currentSearchDone

        // Time management
        this.secondsLeft = 0;
        this.secondsPerMove = 3.0;
        this._countdownTimer = null;
    }

    /**
     * Start a timed search for the best move.
     * Manages the time bank: adds secondsPerMove, computes soft/hard limits.
     * @param {Object} board - Board state to search
     * @param {Object} options
     * @returns {Promise<Object|null>} Search result, or null if aborted/stale
     */
    async startSearch(board, { analyzeMode = false } = {}) {
        if (this.state !== 'idle') {
            await this.abort();
        }

        const generation = ++this._searchGeneration;
        this.state = 'thinking';
        this._setupDonePromise();

        this.dispatchEvent(new CustomEvent('searchStart', { detail: { type: 'think' } }));

        // Time control
        this.secondsLeft += this.secondsPerMove;
        const softTime = this.secondsLeft / 8;
        const hardTime = this.secondsLeft;
        console.log('Starting search: softTime=', softTime.toFixed(2), 'hardTime=', hardTime.toFixed(2));
        const startTime = Date.now();
        const startSecondsLeft = this.secondsLeft;

        // Live countdown timer
        this._countdownTimer = setInterval(() => {
            const elapsed = (Date.now() - startTime) / 1000;
            const projected = Math.max(0, startSecondsLeft - elapsed);
            this.dispatchEvent(new CustomEvent('timeUpdate', { detail: { secondsLeft: projected } }));
        }, 100);

        const onProgress = (progressResult) => {
            if (generation !== this._searchGeneration) return;
            this._reportSearchInfo(progressResult);
        };

        let result;
        try {
            result = await this.engine.search(board, softTime, hardTime, onProgress, analyzeMode);
        } catch (err) {
            console.error('Search exception:', err);
            result = { error: err.message };
        } finally {
            clearInterval(this._countdownTimer);
            this._countdownTimer = null;
        }

        // Debit elapsed time
        const elapsedSeconds = (Date.now() - startTime) / 1000;
        this.secondsLeft -= elapsedSeconds;
        if (this.secondsLeft < 0.1) this.secondsLeft = 0.1;
        this.dispatchEvent(new CustomEvent('timeUpdate', { detail: { secondsLeft: this.secondsLeft } }));

        // Check if this result is stale
        if (generation !== this._searchGeneration) {
            this._finishDone();
            return null;
        }

        // Report final info
        if (result && !result.error) {
            this._reportSearchInfo(result);
        }

        this.state = 'idle';
        this._finishDone();
        this.dispatchEvent(new CustomEvent('searchEnd', { detail: {} }));

        if (result && !result.error) {
            this.dispatchEvent(new CustomEvent('searchComplete', {
                detail: { result, generation }
            }));
        }

        return result;
    }

    /**
     * Start pondering (open-ended background search).
     * Never plays a move — just reports info.
     * @param {Object} board - Board state to ponder
     * @param {Object} options
     */
    async startPondering(board, { fullWindow = false } = {}) {
        if (this.state !== 'idle') {
            await this.abort();
        }

        const generation = ++this._searchGeneration;
        this.state = 'pondering';
        this._setupDonePromise();

        this.dispatchEvent(new CustomEvent('searchStart', { detail: { type: 'ponder' } }));

        const onProgress = (progressResult) => {
            if (generation !== this._searchGeneration) return;
            this._reportSearchInfo(progressResult);
        };

        try {
            const result = await this.engine.search(
                board, 999999, 999999, onProgress, true, fullWindow
            );

            if (generation === this._searchGeneration && result && !result.error) {
                this._reportSearchInfo(result);
            }
        } catch (err) {
            if (generation === this._searchGeneration) {
                console.error('Ponder error:', err);
            }
        }

        // Only transition to idle if we're still the active search
        if (generation === this._searchGeneration) {
            this.state = 'idle';
            this.dispatchEvent(new CustomEvent('searchEnd', { detail: {} }));
        }
        this._finishDone();
    }

    /**
     * Fire-and-forget stop: sets the WASM stop flag.
     * The search will finish with the best result found so far.
     */
    stop() {
        this.engine.stopSearch();
    }

    /**
     * Stop the current search and wait for it to complete.
     * No polling — uses promise-based completion tracking.
     */
    async abort() {
        if (this.state === 'idle') return;

        // Increment generation so stale results are discarded
        this._searchGeneration++;
        this.state = 'idle';
        this.engine.stopSearch();

        // Wait for the search promise to settle
        if (this._currentSearchDone) {
            await this._currentSearchDone;
        }

        clearInterval(this._countdownTimer);
        this._countdownTimer = null;

        this.dispatchEvent(new CustomEvent('searchEnd', { detail: {} }));
    }

    /**
     * Reset time bank to zero
     */
    resetTimeBank() {
        this.secondsLeft = 0;
        this.dispatchEvent(new CustomEvent('timeUpdate', { detail: { secondsLeft: 0 } }));
    }

    /**
     * Set seconds per move
     */
    setSecondsPerMove(seconds) {
        this.secondsPerMove = seconds;
    }

    /**
     * Clear the current PV
     */
    clearPV() {
        this.currentPV = [];
    }

    /**
     * Whether a search is active
     */
    get isSearching() {
        return this.state !== 'idle';
    }

    // --- Internal ---

    _setupDonePromise() {
        this._currentSearchDone = new Promise(resolve => {
            this._resolveDone = resolve;
        });
    }

    _finishDone() {
        if (this._resolveDone) {
            this._resolveDone();
            this._resolveDone = null;
            this._currentSearchDone = null;
        }
    }

    /**
     * Format and dispatch search info
     */
    _reportSearchInfo(result) {
        this.currentPV = result.pv || [];

        // Format score for display
        let scoreStr = '?';
        if (result.score !== undefined) {
            if (Math.abs(result.score) > 29000) {
                const mateIn = Math.ceil((30000 - Math.abs(result.score)) / 2);
                scoreStr = result.score > 0 ? `M${mateIn}` : `-M${mateIn}`;
            } else if (Math.abs(result.score) <= 10000) {
                const val = (result.score / 100).toFixed(2);
                scoreStr = (result.score >= 0 ? '+' : '') + val + '(tablas)';
            } else {
                const raw = result.score > 0 ? result.score - 10000 : result.score + 10000;
                const val = (raw / 100).toFixed(2);
                scoreStr = raw > 0 ? `+${val}` : val;
            }
        }

        const pvStr = result.pv && result.pv.length > 0 ? result.pv.join(' ') : '';

        const info = {
            depth: result.depth,
            score: result.score,
            scoreStr,
            nodes: result.nodes,
            nps: result.nps || 0,
            tbHits: result.tbHits,
            pv: result.pv || [],
            pvStr
        };
        if (result.phase) info.phase = result.phase;
        if (result.rootMoves) info.rootMoves = result.rootMoves;
        if (result.bookMoves) info.bookMoves = result.bookMoves;

        this.dispatchEvent(new CustomEvent('searchInfo', { detail: info }));
    }
}
