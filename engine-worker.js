/**
 * Web Worker for the checkers engine (stateless — board passed to every call)
 * Handles WASM module initialization and search requests
 */

// Import the WASM module (propagate cache-busting version from worker URL)
const _workerV = new URL(self.location.href).searchParams.get('v') || '';
importScripts(_workerV ? `./engine.js?v=${_workerV}` : './engine.js');

let engine = null;
let isReady = false;

// SharedArrayBuffer for stop flag (legacy approach)
let stopFlagBuffer = null;
let stopFlagView = null;

// Direct WASM memory access for stop flag
let wasmStopFlagAddress = 0;
let wasmMemory = null;

// Tablebase lazy loading support
const CHUNK_SIZE = 16384;  // 16 KB chunks
let tbSyncHandles = new Map();  // materialKey -> FileSystemSyncAccessHandle
let tbFileSizes = new Map();    // materialKey -> file size
let chunkCache = new Map();     // materialKey -> Map<chunkIdx, Int8Array>
let tablebasesAvailable = false;

// CWDL (compressed WDL) tablebase support
let cwdlSyncHandles = new Map();  // materialKey -> FileSystemSyncAccessHandle
let cwdlFileSizes = new Map();    // materialKey -> file size
let cwdlAvailable = false;

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

        let dtmCount = 0;
        let cwdlCount = 0;
        for await (const entry of tbDir.values()) {
            if (entry.kind !== 'file' || !entry.name.endsWith('.bin')) continue;

            if (entry.name.startsWith('dtm_')) {
                const materialKey = entry.name.slice(4, 10);  // Extract XXXXXX from dtm_XXXXXX.bin
                try {
                    const fileHandle = await tbDir.getFileHandle(entry.name);
                    const syncHandle = await fileHandle.createSyncAccessHandle();
                    const fileSize = syncHandle.getSize();

                    tbSyncHandles.set(materialKey, syncHandle);
                    tbFileSizes.set(materialKey, fileSize);
                    dtmCount++;
                } catch (e) {
                    console.warn(`Failed to open sync handle for ${entry.name}:`, e);
                }
            } else if (entry.name.startsWith('cwdl_')) {
                const materialKey = entry.name.slice(5, 11);  // Extract XXXXXX from cwdl_XXXXXX.bin
                try {
                    const fileHandle = await tbDir.getFileHandle(entry.name);
                    const syncHandle = await fileHandle.createSyncAccessHandle();
                    const fileSize = syncHandle.getSize();

                    cwdlSyncHandles.set(materialKey, syncHandle);
                    cwdlFileSizes.set(materialKey, fileSize);
                    cwdlCount++;
                } catch (e) {
                    console.warn(`Failed to open sync handle for ${entry.name}:`, e);
                }
            }
        }

        if (dtmCount > 0) {
            tablebasesAvailable = true;
            console.log(`Opened sync handles for ${dtmCount} DTM tablebase files`);
        }
        if (cwdlCount > 0) {
            cwdlAvailable = true;
            console.log(`Opened sync handles for ${cwdlCount} CWDL tablebase files`);
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
 * Check if CWDL tablebases are available
 */
globalThis.cwdlAvailable = function() {
    return cwdlAvailable;
};

/**
 * Load an entire CWDL file by material key
 * Called from WASM to load compressed WDL tablebase data
 * Returns Uint8Array or null if not found
 */
globalThis.loadCWDLFile = function(materialKey) {
    const handle = cwdlSyncHandles.get(materialKey);
    if (!handle) return null;

    const size = cwdlFileSizes.get(materialKey);
    const buffer = new Uint8Array(size);
    handle.read(buffer, { at: 0 });
    return buffer;
};

/**
 * Reconstruct a JSBoard from board data object
 */
function boardFromData(data) {
    return engine.Board.fromBitboards(
        data.white, data.black, data.kings,
        data.whiteToMove, data.nReversible || 0
    );
}

/**
 * Extract plain data from a JSBoard object
 */
function boardToData(jsBoard) {
    return {
        white: jsBoard.getWhite(),
        black: jsBoard.getBlack(),
        kings: jsBoard.getKings(),
        whiteToMove: jsBoard.isWhiteToMove(),
        pieceCount: jsBoard.pieceCount(),
        nReversible: jsBoard.getNReversible()
    };
}

/**
 * Initialize the WASM module
 */
async function init() {
    try {
        // First, init tablebase sync handles (fast)
        await initTablebaseSyncHandles();

        // Then init WASM engine (with cache busting for .wasm and pthread worker)
        engine = await CheckersEngine({
            locateFile: (path) => _workerV ? `./${path}?v=${_workerV}` : `./${path}`
        });

        // Log engine version for debugging cache issues
        const version = engine.getEngineVersion();
        console.log(`Engine version: ${version}`);

        // Get stop flag address for direct memory access
        if (engine.getStopFlagAddress && engine.wasmMemory) {
            wasmStopFlagAddress = engine.getStopFlagAddress();
            wasmMemory = engine.HEAPU8;
            console.log(`Stop flag address: ${wasmStopFlagAddress}`);
            // Send WASM memory buffer and stop flag address to main thread
            postMessage({
                type: 'wasmMemory',
                buffer: engine.wasmMemory.buffer,
                stopFlagAddress: wasmStopFlagAddress
            });
        }

        isReady = true;
        postMessage({ type: 'ready' });
    } catch (err) {
        postMessage({ type: 'error', message: `Failed to initialize engine: ${err.message}` });
    }
}

/**
 * Load neural network model
 */
function loadNNModel(data) {
    if (!engine) return;
    engine.loadNNModel(data);
}

/**
 * Load opening book from .cbook text
 */
function loadOpeningBook(text) {
    if (!engine) return;
    engine.loadOpeningBook(text);
}

/**
 * Get the initial board position
 */
function getInitialBoard() {
    if (!engine) return null;
    const board = engine.getInitialBoard();
    return boardToData(board);
}

/**
 * Get legal moves for a given position
 */
function getLegalMoves(boardData) {
    if (!engine) return [];
    const board = boardFromData(boardData);
    return engine.getLegalMoves(board);
}

/**
 * Make a move on a given board, return the new board state
 */
function makeMove(boardData, moveData) {
    if (!engine) return null;
    const board = boardFromData(boardData);
    const newBoard = engine.makeMove(board, moveData);
    return boardToData(newBoard);
}

// Current search request ID for progress updates
let currentSearchId = null;

/**
 * Perform search with progress updates
 * @param {Object} boardData - Board state
 * @param {number} softTime - Soft time limit in seconds
 * @param {number} hardTime - Hard time limit in seconds
 * @param {number} requestId - Request ID for progress updates
 * @param {boolean} analyzeMode - If true, search even with only one legal move
 * @param {boolean} ponderMode - If true, use full window for all root moves
 */
function search(boardData, softTime, hardTime, requestId, analyzeMode, ponderMode) {
    if (!engine) {
        return { error: 'Engine not ready' };
    }

    const board = boardFromData(boardData);
    currentSearchId = requestId;

    try {
        const searchStart = performance.now();

        // Progress callback - sends updates as search progresses
        const progressCallback = (result) => {
            // Check stop flag from SharedArrayBuffer
            if (stopFlagView && Atomics.load(stopFlagView, 0) !== 0) {
                engine.stopSearch();
            }

            if (currentSearchId !== null) {
                const elapsedMs = performance.now() - searchStart;
                const nodes = result.nodes || 0;
                const nps = elapsedMs > 0 ? Math.round(nodes / (elapsedMs / 1000)) : 0;
                const progress = {
                    bestMove: result.best_move,
                    score: result.score,
                    depth: result.depth,
                    nodes: nodes,
                    nps: nps,
                    tbHits: result.tb_hits,
                    pv: result.pv || []
                };
                if (result.phase) progress.phase = result.phase;
                if (result.rootMoves) progress.rootMoves = result.rootMoves;
                if (result.bookMoves) progress.bookMoves = result.bookMoves;
                postMessage({
                    id: currentSearchId,
                    type: 'searchProgress',
                    result: progress
                });
            }
        };

        let result;
        if (engine.searchWithCallback) {
            result = engine.searchWithCallback(
                board, 100, softTime || 3, hardTime || 10, progressCallback, !!analyzeMode || !!ponderMode, !!ponderMode
            );
        } else {
            result = engine.search(board, 100, softTime || 3, hardTime || 10);
        }

        const totalElapsedMs = performance.now() - searchStart;
        const totalNodes = result.nodes || 0;
        const nps = totalElapsedMs > 0 ? Math.round(totalNodes / (totalElapsedMs / 1000)) : 0;

        currentSearchId = null;
        const searchResult = {
            bestMove: result.best_move,
            score: result.score,
            depth: result.depth,
            nodes: totalNodes,
            nps: nps,
            tbHits: result.tb_hits,
            pv: result.pv || []
        };
        if (result.phase) searchResult.phase = result.phase;
        if (result.book) searchResult.book = true;
        if (result.rootMoves) searchResult.rootMoves = result.rootMoves;
        if (result.bookMoves) searchResult.bookMoves = result.bookMoves;
        return searchResult;
    } catch (err) {
        console.error('Worker: search error:', err);
        currentSearchId = null;
        return { error: err.message || 'Search failed' };
    }
}

/**
 * Probe DTM tablebase for a given position
 */
function probeDTM(boardData) {
    if (!engine) return null;
    const board = boardFromData(boardData);
    return engine.probeDTM(board);
}

/**
 * Parse move from notation string for a given position
 */
function parseMove(boardData, notation) {
    if (!engine) return null;
    const board = boardFromData(boardData);
    return engine.parseMove(board, notation);
}

/**
 * Check if tablebases/models are loaded
 */
function getLoadedStatus() {
    if (!engine) return { tablebases: false, cwdl: false, nnModel: false };
    return {
        tablebases: tablebasesAvailable,
        cwdl: cwdlAvailable,
        nnModel: engine.hasNNModel(),
        openingBook: engine.hasOpeningBook()
    };
}

// Message handler
self.onmessage = function(e) {
    const { id, type, data, buffer } = e.data;

    // Handle stop flag setup (no response needed)
    if (type === 'setStopFlag') {
        stopFlagBuffer = buffer;
        stopFlagView = new Int32Array(buffer);
        return;
    }

    let response = { id, type };

    try {
        switch (type) {
            case 'init':
                // Init is async, will post 'ready' message when done
                init();
                return;

            case 'loadNNModel':
                loadNNModel(data.data);
                response.success = true;
                break;

            case 'loadOpeningBook':
                loadOpeningBook(data.text);
                response.success = true;
                break;

            case 'getInitialBoard':
                response.board = getInitialBoard();
                break;

            case 'getLegalMoves':
                response.moves = getLegalMoves(data);
                break;

            case 'makeMove':
                response.board = makeMove(data.board, data.move);
                break;

            case 'search':
                response.result = search(data.board, data.softTime, data.hardTime, id, data.analyzeMode, data.ponderMode);
                break;

            case 'setUseBook':
                if (engine && engine.setUseBook) engine.setUseBook(data.useBook);
                response.success = true;
                break;

            case 'stop':
                if (engine) {
                    engine.stopSearch();
                }
                response.success = true;
                break;

            case 'probeDTM':
                response.dtm = probeDTM(data);
                break;

            case 'parseMove':
                response.move = parseMove(data.board, data.notation);
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
