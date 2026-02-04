/**
 * Main thread API for communicating with the engine Web Worker
 */

export class EngineAPI {
    constructor() {
        this.worker = null;
        this.pendingRequests = new Map();
        this.requestId = 0;
        this.isReady = false;
        this.onReady = null;
        this.onError = null;
        // SharedArrayBuffer for stop flag (legacy approach)
        this.stopFlag = null;
        this.stopFlagView = null;
        // Direct WASM memory access (preferred)
        this.wasmMemory = null;
        this.wasmStopFlagAddress = 0;
    }

    /**
     * Initialize the engine worker
     * @param {string} workerPath - Path to engine-worker.js
     * @returns {Promise<void>}
     */
    async init(workerPath = './engine-worker.js') {
        // Create SharedArrayBuffer for stop flag if available
        if (typeof SharedArrayBuffer !== 'undefined') {
            this.stopFlag = new SharedArrayBuffer(4);
            this.stopFlagView = new Int32Array(this.stopFlag);
        }

        return new Promise((resolve, reject) => {
            this.worker = new Worker(workerPath);

            // Send stop flag buffer to worker
            if (this.stopFlag) {
                this.worker.postMessage({ type: 'setStopFlag', buffer: this.stopFlag });
            }

            this.worker.onmessage = (e) => {
                const { id, type, ...data } = e.data;

                // Handle WASM memory sharing for direct stop flag access
                if (type === 'wasmMemory') {
                    this.wasmMemory = new Uint8Array(data.buffer);
                    this.wasmStopFlagAddress = data.stopFlagAddress;
                    console.log(`Received WASM memory, stop flag at ${this.wasmStopFlagAddress}`);
                    return;
                }

                // Handle ready message
                if (type === 'ready') {
                    this.isReady = true;
                    if (this.onReady) this.onReady();
                    resolve();
                    return;
                }

                // Handle error message
                if (type === 'error') {
                    if (this.onError) this.onError(data.message);
                    reject(new Error(data.message));
                    return;
                }

                // Handle search progress update
                if (type === 'searchProgress' && id !== undefined && this.pendingRequests.has(id)) {
                    const { onProgress } = this.pendingRequests.get(id);
                    if (onProgress) {
                        onProgress(data.result);
                    }
                    return; // Don't resolve yet, wait for final result
                }

                // Handle response to a request
                if (id !== undefined && this.pendingRequests.has(id)) {
                    const { resolve, reject } = this.pendingRequests.get(id);
                    this.pendingRequests.delete(id);

                    if (data.error) {
                        reject(new Error(data.error));
                    } else {
                        resolve(data);
                    }
                }
            };

            this.worker.onerror = (err) => {
                if (this.onError) this.onError(err.message);
                reject(err);
            };
        });
    }

    /**
     * Send a request to the worker and wait for response
     */
    async request(type, data = {}) {
        if (!this.worker) {
            throw new Error('Worker not initialized');
        }

        return new Promise((resolve, reject) => {
            const id = ++this.requestId;
            this.pendingRequests.set(id, { resolve, reject });
            this.worker.postMessage({ id, type, data });
        });
    }

    /**
     * Load a neural network model
     * @param {Uint8Array} data - Raw model data
     * @param {boolean} isDTMModel - True if this is the DTM specialist model
     */
    async loadNNModel(data, isDTMModel = false) {
        return this.request('loadNNModel', { data, isDTMModel });
    }

    /**
     * Get legal moves for the current position
     * @returns {Promise<Array>} Array of move objects
     */
    async getLegalMoves() {
        const response = await this.request('getLegalMoves');
        return response.moves;
    }

    /**
     * Make a move on the board
     * @param {Object} move - Move object with from_xor_to and captures
     * @returns {Promise<Object>} New board state
     */
    async makeMove(move) {
        const response = await this.request('makeMove', move);
        return response.board;
    }

    /**
     * Set the board position
     * @param {number} white - White pieces bitboard
     * @param {number} black - Black pieces bitboard
     * @param {number} kings - Kings bitboard
     * @param {boolean} whiteToMove - True if white to move
     */
    async setBoard(white, black, kings, whiteToMove) {
        const response = await this.request('setBoard', { white, black, kings, whiteToMove });
        return response.board;
    }

    /**
     * Reset to initial position
     */
    async resetBoard() {
        const response = await this.request('resetBoard');
        return response.board;
    }

    /**
     * Get current board state
     */
    async getBoard() {
        const response = await this.request('getBoard');
        return response.board;
    }

    /**
     * Search for the best move
     * @param {number} maxDepth - Maximum search depth
     * @param {number} maxNodes - Maximum nodes to search (0 = unlimited)
     * @param {number} gamePly - Current game ply (for opening variety)
     * @param {number} varietyMode - Variety mode: 0=none, 1=safe, 2=curious, 3=wild
     * @param {Function} onProgress - Optional callback for progress updates
     * @returns {Promise<Object>} Search result with bestMove, score, depth, nodes
     */
    async search(maxDepth = 20, maxNodes = 0, gamePly = 100, varietyMode = 0, onProgress = null) {
        // Reset stop flag before starting search
        if (this.wasmMemory && this.wasmStopFlagAddress) {
            this.wasmMemory[this.wasmStopFlagAddress] = 0;
        } else if (this.stopFlagView) {
            Atomics.store(this.stopFlagView, 0, 0);
        }
        const response = await this.requestWithProgress('search', { maxDepth, maxNodes, gamePly, varietyMode }, onProgress);
        return response.result;
    }

    /**
     * Stop the current search
     */
    stopSearch() {
        // Write directly to WASM memory (immediate, works even when worker is blocked)
        if (this.wasmMemory && this.wasmStopFlagAddress) {
            this.wasmMemory[this.wasmStopFlagAddress] = 1;
            console.log('Set stop flag via WASM memory');
        }
        // Fallback: SharedArrayBuffer approach
        else if (this.stopFlagView) {
            Atomics.store(this.stopFlagView, 0, 1);
        }
        // Also send message as fallback (processed after search completes)
        if (this.worker) {
            this.worker.postMessage({ type: 'stop' });
        }
    }

    /**
     * Send a request that may receive progress updates
     */
    async requestWithProgress(type, data = {}, onProgress = null) {
        if (!this.worker) {
            throw new Error('Worker not initialized');
        }

        return new Promise((resolve, reject) => {
            const id = ++this.requestId;
            this.pendingRequests.set(id, { resolve, reject, onProgress });
            this.worker.postMessage({ id, type, data });
        });
    }

    /**
     * Probe DTM tablebase for current position
     * @returns {Promise<number|null>} DTM value or null if not available
     */
    async probeDTM() {
        const response = await this.request('probeDTM');
        return response.dtm;
    }

    /**
     * Parse a move from notation string
     * @param {string} notation - Move in notation (e.g., "9-13" or "9x14x23")
     * @returns {Promise<Object|null>} Move object or null if invalid
     */
    async parseMove(notation) {
        const response = await this.request('parseMove', { notation });
        return response.move;
    }

    /**
     * Get status of loaded resources
     */
    async getStatus() {
        const response = await this.request('getStatus');
        return response;
    }

    /**
     * Terminate the worker
     */
    terminate() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
            this.isReady = false;
        }
    }
}

// Singleton instance
let engineInstance = null;

/**
 * Get the singleton engine instance
 */
export function getEngine() {
    if (!engineInstance) {
        engineInstance = new EngineAPI();
    }
    return engineInstance;
}
