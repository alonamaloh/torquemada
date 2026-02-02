/**
 * Tablebase loader using OPFS (Origin Private File System)
 * Handles downloading, storing, and loading DTM tablebases
 */

// Base URL for tablebase files (configure as needed)
const TABLEBASE_BASE_URL = './tablebases/';

// List of DTM tablebase files (up to 5 pieces)
const DTM_FILES = [
    'dtm_000011.bin', 'dtm_000012.bin', 'dtm_000013.bin', 'dtm_000014.bin',
    'dtm_000021.bin', 'dtm_000022.bin', 'dtm_000023.bin', 'dtm_000031.bin',
    'dtm_000032.bin', 'dtm_000041.bin', 'dtm_000110.bin', 'dtm_000111.bin',
    'dtm_000112.bin', 'dtm_000113.bin', 'dtm_000120.bin', 'dtm_000121.bin',
    'dtm_000122.bin', 'dtm_000130.bin', 'dtm_000131.bin', 'dtm_000140.bin',
    'dtm_000210.bin', 'dtm_000211.bin', 'dtm_000212.bin', 'dtm_000220.bin',
    'dtm_000221.bin', 'dtm_000230.bin', 'dtm_000310.bin', 'dtm_000311.bin',
    'dtm_000320.bin', 'dtm_000410.bin', 'dtm_001001.bin', 'dtm_001002.bin',
    'dtm_001003.bin', 'dtm_001004.bin', 'dtm_001011.bin', 'dtm_001012.bin',
    'dtm_001013.bin', 'dtm_001021.bin', 'dtm_001022.bin', 'dtm_001031.bin',
    'dtm_001100.bin', 'dtm_001101.bin', 'dtm_001102.bin', 'dtm_001103.bin',
    'dtm_001110.bin', 'dtm_001111.bin', 'dtm_001112.bin', 'dtm_001120.bin',
    'dtm_001121.bin', 'dtm_001130.bin', 'dtm_001200.bin', 'dtm_001201.bin',
    'dtm_001202.bin', 'dtm_001210.bin', 'dtm_001211.bin', 'dtm_001220.bin',
    'dtm_001300.bin', 'dtm_001301.bin', 'dtm_001310.bin', 'dtm_001400.bin',
    'dtm_002001.bin', 'dtm_002002.bin', 'dtm_002003.bin', 'dtm_002011.bin',
    'dtm_002012.bin', 'dtm_002021.bin', 'dtm_002100.bin', 'dtm_002101.bin',
    'dtm_002102.bin', 'dtm_002110.bin', 'dtm_002111.bin', 'dtm_002120.bin',
    'dtm_002200.bin', 'dtm_002201.bin', 'dtm_002210.bin', 'dtm_002300.bin',
    'dtm_003001.bin', 'dtm_003002.bin', 'dtm_003011.bin', 'dtm_003100.bin',
    'dtm_003101.bin', 'dtm_003110.bin', 'dtm_003200.bin', 'dtm_004001.bin',
    'dtm_004100.bin', 'dtm_010010.bin', 'dtm_010011.bin', 'dtm_010012.bin',
    'dtm_010013.bin', 'dtm_010020.bin', 'dtm_010021.bin', 'dtm_010022.bin',
    'dtm_010030.bin', 'dtm_010031.bin', 'dtm_010040.bin', 'dtm_010110.bin',
    'dtm_010111.bin', 'dtm_010112.bin', 'dtm_010120.bin', 'dtm_010121.bin',
    'dtm_010130.bin', 'dtm_010210.bin', 'dtm_010211.bin', 'dtm_010220.bin',
    'dtm_010310.bin', 'dtm_011000.bin', 'dtm_011001.bin', 'dtm_011002.bin',
    'dtm_011003.bin', 'dtm_011010.bin', 'dtm_011011.bin', 'dtm_011012.bin',
    'dtm_011020.bin', 'dtm_011021.bin', 'dtm_011030.bin', 'dtm_011100.bin',
    'dtm_011101.bin', 'dtm_011102.bin', 'dtm_011110.bin', 'dtm_011111.bin',
    'dtm_011120.bin', 'dtm_011200.bin', 'dtm_011201.bin', 'dtm_011210.bin',
    'dtm_011300.bin', 'dtm_012000.bin', 'dtm_012001.bin', 'dtm_012002.bin',
    'dtm_012010.bin', 'dtm_012011.bin', 'dtm_012020.bin', 'dtm_012100.bin',
    'dtm_012101.bin', 'dtm_012110.bin', 'dtm_012200.bin', 'dtm_013000.bin',
    'dtm_013001.bin', 'dtm_013010.bin', 'dtm_013100.bin', 'dtm_014000.bin',
    'dtm_020010.bin', 'dtm_020011.bin', 'dtm_020012.bin', 'dtm_020020.bin',
    'dtm_020021.bin', 'dtm_020030.bin', 'dtm_020110.bin', 'dtm_020111.bin',
    'dtm_020120.bin', 'dtm_020210.bin', 'dtm_021000.bin', 'dtm_021001.bin',
    'dtm_021002.bin', 'dtm_021010.bin', 'dtm_021011.bin', 'dtm_021020.bin',
    'dtm_021100.bin', 'dtm_021101.bin', 'dtm_021110.bin', 'dtm_021200.bin',
    'dtm_022000.bin', 'dtm_022001.bin', 'dtm_022010.bin', 'dtm_022100.bin',
    'dtm_023000.bin', 'dtm_030010.bin', 'dtm_030011.bin', 'dtm_030020.bin',
    'dtm_030110.bin', 'dtm_031000.bin', 'dtm_031001.bin', 'dtm_031010.bin',
    'dtm_031100.bin', 'dtm_032000.bin', 'dtm_040010.bin', 'dtm_041000.bin',
    'dtm_100001.bin', 'dtm_100002.bin', 'dtm_100003.bin', 'dtm_100004.bin',
    'dtm_100011.bin', 'dtm_100012.bin', 'dtm_100013.bin', 'dtm_100021.bin',
    'dtm_100022.bin', 'dtm_100031.bin', 'dtm_100100.bin', 'dtm_100101.bin',
    'dtm_100102.bin', 'dtm_100103.bin', 'dtm_100110.bin', 'dtm_100111.bin',
    'dtm_100112.bin', 'dtm_100120.bin', 'dtm_100121.bin', 'dtm_100130.bin',
    'dtm_100200.bin', 'dtm_100201.bin', 'dtm_100202.bin', 'dtm_100210.bin',
    'dtm_100211.bin', 'dtm_100220.bin', 'dtm_100300.bin', 'dtm_100301.bin',
    'dtm_100310.bin', 'dtm_100400.bin', 'dtm_101001.bin', 'dtm_101002.bin',
    'dtm_101003.bin', 'dtm_101011.bin', 'dtm_101012.bin', 'dtm_101021.bin',
    'dtm_101100.bin', 'dtm_101101.bin', 'dtm_101102.bin', 'dtm_101110.bin',
    'dtm_101111.bin', 'dtm_101120.bin', 'dtm_101200.bin', 'dtm_101201.bin',
    'dtm_101210.bin', 'dtm_101300.bin', 'dtm_102001.bin', 'dtm_102002.bin',
    'dtm_102011.bin', 'dtm_102100.bin', 'dtm_102101.bin', 'dtm_102110.bin',
    'dtm_102200.bin', 'dtm_103001.bin', 'dtm_103100.bin', 'dtm_110000.bin',
    'dtm_110001.bin', 'dtm_110002.bin', 'dtm_110003.bin', 'dtm_110010.bin',
    'dtm_110011.bin', 'dtm_110012.bin', 'dtm_110020.bin', 'dtm_110021.bin',
    'dtm_110030.bin', 'dtm_110100.bin', 'dtm_110101.bin', 'dtm_110102.bin',
    'dtm_110110.bin', 'dtm_110111.bin', 'dtm_110120.bin', 'dtm_110200.bin',
    'dtm_110201.bin', 'dtm_110210.bin', 'dtm_110300.bin', 'dtm_111000.bin',
    'dtm_111001.bin', 'dtm_111002.bin', 'dtm_111010.bin', 'dtm_111011.bin',
    'dtm_111020.bin', 'dtm_111100.bin', 'dtm_111101.bin', 'dtm_111110.bin',
    'dtm_111200.bin', 'dtm_112000.bin', 'dtm_112001.bin', 'dtm_112010.bin',
    'dtm_112100.bin', 'dtm_113000.bin', 'dtm_120000.bin', 'dtm_120001.bin',
    'dtm_120002.bin', 'dtm_120010.bin', 'dtm_120011.bin', 'dtm_120020.bin',
    'dtm_120100.bin', 'dtm_120101.bin', 'dtm_120110.bin', 'dtm_120200.bin',
    'dtm_121000.bin', 'dtm_121001.bin', 'dtm_121010.bin', 'dtm_121100.bin',
    'dtm_122000.bin', 'dtm_130000.bin', 'dtm_130001.bin', 'dtm_130010.bin',
    'dtm_130100.bin', 'dtm_131000.bin', 'dtm_140000.bin', 'dtm_200001.bin',
    'dtm_200002.bin', 'dtm_200003.bin', 'dtm_200011.bin', 'dtm_200012.bin',
    'dtm_200021.bin', 'dtm_200100.bin', 'dtm_200101.bin', 'dtm_200102.bin',
    'dtm_200110.bin', 'dtm_200111.bin', 'dtm_200120.bin', 'dtm_200200.bin',
    'dtm_200201.bin', 'dtm_200210.bin', 'dtm_200300.bin', 'dtm_201001.bin',
    'dtm_201002.bin', 'dtm_201011.bin', 'dtm_201100.bin', 'dtm_201101.bin',
    'dtm_201110.bin', 'dtm_201200.bin', 'dtm_202001.bin', 'dtm_202100.bin',
    'dtm_210000.bin', 'dtm_210001.bin', 'dtm_210002.bin', 'dtm_210010.bin',
    'dtm_210011.bin', 'dtm_210020.bin', 'dtm_210100.bin', 'dtm_210101.bin',
    'dtm_210110.bin', 'dtm_210200.bin', 'dtm_211000.bin', 'dtm_211001.bin',
    'dtm_211010.bin', 'dtm_211100.bin', 'dtm_212000.bin', 'dtm_220000.bin',
    'dtm_220001.bin', 'dtm_220010.bin', 'dtm_220100.bin', 'dtm_221000.bin',
    'dtm_230000.bin', 'dtm_300001.bin', 'dtm_300002.bin', 'dtm_300011.bin',
    'dtm_300100.bin', 'dtm_300101.bin', 'dtm_300110.bin', 'dtm_300200.bin',
    'dtm_301001.bin', 'dtm_301100.bin', 'dtm_310000.bin', 'dtm_310001.bin',
    'dtm_310010.bin', 'dtm_310100.bin', 'dtm_311000.bin', 'dtm_320000.bin',
    'dtm_400001.bin', 'dtm_400100.bin', 'dtm_410000.bin',
];

/**
 * TablebaseLoader class - manages tablebase storage and retrieval
 */
export class TablebaseLoader {
    constructor() {
        this.opfsRoot = null;
        this.tbDirectory = null;
        this.loadedFiles = new Set();
        this.onProgress = null;
        this.isInitialized = false;
    }

    /**
     * Initialize OPFS access
     */
    async init() {
        if (!('storage' in navigator) || !('getDirectory' in navigator.storage)) {
            throw new Error('OPFS not supported in this browser');
        }

        this.opfsRoot = await navigator.storage.getDirectory();
        this.tbDirectory = await this.opfsRoot.getDirectoryHandle('tablebases', { create: true });
        this.isInitialized = true;
    }

    /**
     * Check if OPFS is available and initialized
     */
    isAvailable() {
        return this.isInitialized && this.tbDirectory !== null;
    }

    /**
     * Check which tablebases are already stored in OPFS
     */
    async checkStoredTablebases() {
        if (!this.isAvailable()) {
            return [];
        }
        const stored = [];
        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file' && entry.name.startsWith('dtm_')) {
                stored.push(entry.name);
            }
        }
        return stored;
    }

    /**
     * Download missing tablebases from server
     * @param {Function} onProgress - Callback(loaded, total, currentFile)
     */
    async downloadTablebases(onProgress = null) {
        if (!this.isAvailable()) {
            throw new Error('OPFS not available. Tablebase storage requires a browser with Origin Private File System support (Chrome, Edge, Firefox). Safari does not support OPFS.');
        }

        this.onProgress = onProgress;

        // Check what's already stored
        const stored = await this.checkStoredTablebases();
        const storedSet = new Set(stored);

        // Filter to only missing files
        const missing = DTM_FILES.filter(f => !storedSet.has(f));

        if (missing.length === 0) {
            if (onProgress) onProgress(DTM_FILES.length, DTM_FILES.length, 'Complete');
            return { downloaded: 0, total: DTM_FILES.length };
        }

        let downloaded = stored.length;
        const total = DTM_FILES.length;

        for (const filename of missing) {
            try {
                if (onProgress) onProgress(downloaded + 1, total, filename);

                const response = await fetch(TABLEBASE_BASE_URL + filename);
                if (!response.ok) {
                    console.warn(`Failed to download ${filename}: ${response.status}`);
                    continue;
                }

                const data = await response.arrayBuffer();

                // Store in OPFS
                const fileHandle = await this.tbDirectory.getFileHandle(filename, { create: true });
                const writable = await fileHandle.createWritable();
                await writable.write(data);
                await writable.close();

                downloaded++;
            } catch (err) {
                console.warn(`Error downloading ${filename}:`, err);
            }
        }

        if (onProgress) onProgress(downloaded, total, 'Complete');
        return { downloaded: downloaded - stored.length, total };
    }

    /**
     * Load a tablebase file from OPFS (sync access for Worker)
     * Returns Uint8Array or null if not found
     */
    async loadTablebase(filename) {
        if (!this.isAvailable()) {
            return null;
        }
        try {
            const fileHandle = await this.tbDirectory.getFileHandle(filename);
            const file = await fileHandle.getFile();
            const buffer = await file.arrayBuffer();
            return new Uint8Array(buffer);
        } catch (err) {
            return null;
        }
    }

    /**
     * Load all tablebases and return them as a map
     * @returns {Map<string, Uint8Array>}
     */
    async loadAllTablebases() {
        const tablebases = new Map();

        if (!this.isAvailable()) {
            return tablebases;
        }

        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file' && entry.name.startsWith('dtm_')) {
                const data = await this.loadTablebase(entry.name);
                if (data) {
                    // Extract material key from filename (dtm_XXXXXX.bin -> XXXXXX)
                    const materialKey = entry.name.slice(4, 10);
                    tablebases.set(materialKey, data);
                }
            }
        }

        return tablebases;
    }

    /**
     * Get total size of stored tablebases
     */
    async getStoredSize() {
        if (!this.isAvailable()) {
            return 0;
        }
        let total = 0;
        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file') {
                const file = await (await this.tbDirectory.getFileHandle(entry.name)).getFile();
                total += file.size;
            }
        }
        return total;
    }

    /**
     * Clear all stored tablebases
     */
    async clearTablebases() {
        if (!this.isAvailable()) {
            return;
        }
        for await (const entry of this.tbDirectory.values()) {
            if (entry.kind === 'file') {
                await this.tbDirectory.removeEntry(entry.name);
            }
        }
    }
}

/**
 * Load NN model file from server
 */
export async function loadNNModelFile(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load model: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

export { DTM_FILES };
