/**
 * Web Worker for the checkers engine
 * Handles WASM module initialization and search requests
 */

// Import the WASM module
importScripts('./engine.js');

let engine = null;
let board = null;
let isReady = false;

// Tablebase lazy loading support
const CHUNK_SIZE = 16384;  // 16 KB chunks
let tbSyncHandles = new Map();  // materialKey -> FileSystemSyncAccessHandle
let tbFileSizes = new Map();    // materialKey -> file size
let chunkCache = new Map();     // materialKey -> Map<chunkIdx, Int8Array>
let tablebasesAvailable = false;

/**
 * Initialize sync access handles for all tablebase files
 * This is fast - just opens handles without reading data
 */
async function initTablebaseSyncHandles() {
    try {
        if (!('storage' in navigator) || !('getDirectory' in navigator.storage)) {
            console.log('OPFS not available in worker');
            return;
        }

        const opfsRoot = await navigator.storage.getDirectory();
        let tbDir;
        try {
            tbDir = await opfsRoot.getDirectoryHandle('tablebases', { create: false });
        } catch (e) {
            console.log('No tablebases directory found');
            return;
        }

        let count = 0;
        for await (const entry of tbDir.values()) {
            if (entry.kind === 'file' && entry.name.startsWith('dtm_') && entry.name.endsWith('.bin')) {
                const materialKey = entry.name.slice(4, 10);  // Extract XXXXXX from dtm_XXXXXX.bin
                try {
                    const fileHandle = await tbDir.getFileHandle(entry.name);
                    const syncHandle = await fileHandle.createSyncAccessHandle();
                    const fileSize = syncHandle.getSize();

                    tbSyncHandles.set(materialKey, syncHandle);
                    tbFileSizes.set(materialKey, fileSize);
                    count++;
                } catch (e) {
                    console.warn(`Failed to open sync handle for ${entry.name}:`, e);
                }
            }
        }

        if (count > 0) {
            tablebasesAvailable = true;
            console.log(`Opened sync handles for ${count} tablebase files`);
        }
    } catch (err) {
        console.warn('Failed to init tablebase sync handles:', err);
    }
}

/**
 * Load a specific chunk from a tablebase file
 * Called from WASM via globalThis.loadTablebaseChunk
 */
globalThis.loadTablebaseChunk = function(materialKey, chunkIdx) {
    // Check cache first
    if (!chunkCache.has(materialKey)) {
        chunkCache.set(materialKey, new Map());
    }
    const fileCache = chunkCache.get(materialKey);
    if (fileCache.has(chunkIdx)) {
        return fileCache.get(chunkIdx);
    }

    // Load from OPFS
    const handle = tbSyncHandles.get(materialKey);
    if (!handle) {
        return null;
    }

    const fileSize = tbFileSizes.get(materialKey);
    const offset = chunkIdx * CHUNK_SIZE;
    const readSize = Math.min(CHUNK_SIZE, fileSize - offset);

    if (readSize <= 0) {
        return null;
    }

    const buffer = new Uint8Array(readSize);
    handle.read(buffer, { at: offset });

    // Convert to Int8Array for signed interpretation
    const signedBuffer = new Int8Array(buffer.buffer);

    // Cache it
    fileCache.set(chunkIdx, signedBuffer);

    return signedBuffer;
};

/**
 * Check if tablebases are available (have sync handles open)
 */
globalThis.tablebasesAvailable = function() {
    return tablebasesAvailable;
};

/**
 * Initialize the WASM module
 */
async function init() {
    try {
        // First, init tablebase sync handles (fast)
        await initTablebaseSyncHandles();

        // Then init WASM engine
        engine = await CheckersEngine();

        // Log engine version for debugging cache issues
        const version = engine.getEngineVersion();
        console.log(`Engine version: ${version}`);

        board = engine.getInitialBoard();
        isReady = true;
        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', message: `Failed to initialize engine: ${err.message}` });
    }
}

/**
 * Load neural network model
 */
function loadNNModel(data, isDTMModel) {
    if (!engine) return;
    engine.loadNNModel(data, isDTMModel);
}

/**
 * Get legal moves for the current position
 */
function getLegalMoves() {
    if (!engine || !board) return [];
    const moves = engine.getLegalMoves(board);
    return moves;
}

/**
 * Make a move on the board
 */
function makeMove(moveData) {
    if (!engine || !board) return null;
    board = engine.makeMove(board, moveData);
    return getBoardState();
}

/**
 * Set board position
 */
function setBoard(white, black, kings, whiteToMove) {
    if (!engine) return;
    board = engine.Board.fromBitboards(white, black, kings, whiteToMove);
}

/**
 * Reset to initial position
 */
function resetBoard() {
    if (!engine) return;
    board = engine.getInitialBoard();
    return getBoardState();
}

/**
 * Get current board state
 */
function getBoardState() {
    if (!board) return null;
    return {
        white: board.getWhite(),
        black: board.getBlack(),
        kings: board.getKings(),
        whiteToMove: board.isWhiteToMove(),
        pieceCount: board.pieceCount()
    };
}

// Current search request ID for progress updates
let currentSearchId = null;

/**
 * Perform search with progress updates
 * @param {number} maxDepth - Maximum search depth
 * @param {number} maxNodes - Maximum nodes to search
 * @param {number} gamePly - Current game ply (for opening variety)
 * @param {number} varietyMode - Variety mode: 0=none, 1=safe, 2=curious, 3=wild
 * @param {number} requestId - Request ID for progress updates
 */
function search(maxDepth, maxNodes, gamePly, varietyMode, requestId) {
    if (!engine || !board) {
        return { error: 'Engine not ready' };
    }

    currentSearchId = requestId;

    try {
        // Progress callback - sends updates as search progresses
        const progressCallback = (result) => {
            if (currentSearchId !== null) {
                postMessage({
                    id: currentSearchId,
                    type: 'searchProgress',
                    result: {
                        bestMove: result.best_move,
                        score: result.score,
                        depth: result.depth,
                        nodes: result.nodes,
                        tbHits: result.tb_hits,
                        pv: result.pv || []
                    }
                });
            }
        };

        // Use searchWithCallback with game ply and variety mode
        let result;
        if (engine.searchWithCallback) {
            result = engine.searchWithCallback(
                board,
                maxDepth || 20,
                maxNodes || 0,
                gamePly || 0,
                varietyMode || 0,
                progressCallback
            );
        } else {
            result = engine.search(board, maxDepth || 20, maxNodes || 0);
        }

        currentSearchId = null;
        return {
            bestMove: result.best_move,
            score: result.score,
            depth: result.depth,
            nodes: result.nodes,
            tbHits: result.tb_hits,
            pv: result.pv || []
        };
    } catch (err) {
        console.error('Worker: search error:', err);
        currentSearchId = null;
        return { error: err.message || 'Search failed' };
    }
}

/**
 * Probe DTM tablebase for current position
 */
function probeDTM() {
    if (!engine || !board) return null;
    const dtm = engine.probeDTM(board);
    return dtm;
}

/**
 * Parse move from notation string
 */
function parseMove(notation) {
    if (!engine || !board) return null;
    const move = engine.parseMove(board, notation);
    return move;
}

/**
 * Check if tablebases/models are loaded
 */
function getLoadedStatus() {
    if (!engine) return { tablebases: false, nnModel: false, dtmNNModel: false };
    return {
        tablebases: tablebasesAvailable,
        nnModel: engine.hasNNModel(),
        dtmNNModel: engine.hasDTMNNModel()
    };
}

// Message handler
self.onmessage = function(e) {
    const { id, type, data } = e.data;

    let response = { id, type };

    try {
        switch (type) {
            case 'init':
                // Init is async, will post 'ready' message when done
                init();
                return;

            case 'loadNNModel':
                loadNNModel(data.data, data.isDTMModel);
                response.success = true;
                break;

            case 'getLegalMoves':
                response.moves = getLegalMoves();
                break;

            case 'makeMove':
                response.board = makeMove(data);
                break;

            case 'setBoard':
                setBoard(data.white, data.black, data.kings, data.whiteToMove);
                response.board = getBoardState();
                break;

            case 'resetBoard':
                response.board = resetBoard();
                break;

            case 'getBoard':
                response.board = getBoardState();
                break;

            case 'search':
                response.result = search(data.maxDepth, data.maxNodes, data.gamePly, data.varietyMode, id);
                break;

            case 'stop':
                if (engine) {
                    engine.stopSearch();
                }
                response.success = true;
                break;

            case 'probeDTM':
                response.dtm = probeDTM();
                break;

            case 'parseMove':
                response.move = parseMove(data.notation);
                break;

            case 'getStatus':
                response.status = getLoadedStatus();
                response.ready = isReady;
                break;

            default:
                response.error = `Unknown message type: ${type}`;
        }
    } catch (err) {
        response.error = err.message;
    }

    postMessage(response);
};

// Start initialization
init();
