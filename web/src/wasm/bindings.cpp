// Emscripten bindings for the checkers engine
// Exposes Board, Move, Searcher, and helper functions to JavaScript

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <memory>
#include <vector>
#include <cstdint>
#include <bit>
#include <chrono>

#include "../../../core/board.hpp"
#include "../../../core/movegen.hpp"
#include "../../../core/notation.hpp"
#include "../../../core/random.hpp"
#include "../../../search/search.hpp"
#include "../../../tablebase/tablebase.hpp"
#include "../../../nn/mlp.hpp"

using namespace emscripten;

// Engine version - update this when making changes to help debug caching issues
#define ENGINE_VERSION __DATE__ " " __TIME__

namespace {
    // Constants for chunk-based tablebase loading
    constexpr int CHUNK_SIZE = 16384;  // 16 KB chunks
    constexpr int HEADER_SIZE = 33;    // DTM file header size

    // Global RNG for evaluation noise, seeded with system clock
    RandomBits g_rng(std::chrono::steady_clock::now().time_since_epoch().count());

    // Global stop flag for interrupting search
    std::atomic<bool> g_stop_flag{false};

    // Helper to build notation string from path
    std::string buildNotation(const std::vector<int>& path, bool is_capture, bool white_view) {
        std::string result;
        for (size_t i = 0; i < path.size(); ++i) {
            if (i > 0) result += is_capture ? "x" : "-";
            int sq = white_view ? (path[i] + 1) : (32 - path[i]);
            result += std::to_string(sq);
        }
        return result;
    }

    // Helper to convert Material to string key
    std::string material_key(const Material& m) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d%d%d%d%d%d",
                 m.back_white_pawns, m.back_black_pawns,
                 m.other_white_pawns, m.other_black_pawns,
                 m.white_queens, m.black_queens);
        return buf;
    }

    // Lazy-loading DTM tablebase manager
    // Calls into JS to load chunks on demand
    class WasmDTMTablebaseManager {
    public:
        tablebase::DTM lookup_dtm(const Board& board) const {
            // Check if tablebases are available in JS
            val tablebasesAvailable = val::global("tablebasesAvailable");
            if (tablebasesAvailable.isUndefined() || !tablebasesAvailable().as<bool>()) {
                return tablebase::DTM_UNKNOWN;
            }

            Material m = get_material(board);
            std::string key = material_key(m);
            std::size_t idx = board_to_index(board, m);

            // Compute chunk index and offset
            // File layout: 33-byte header + DTM values (1 byte each)
            int file_offset = HEADER_SIZE + static_cast<int>(idx);
            int chunk_idx = file_offset / CHUNK_SIZE;
            int offset_in_chunk = file_offset % CHUNK_SIZE;

            // Call JS to get the chunk (it handles caching)
            val loadChunk = val::global("loadTablebaseChunk");
            val chunk = loadChunk(key, chunk_idx);

            if (chunk.isNull() || chunk.isUndefined()) {
                return tablebase::DTM_UNKNOWN;
            }

            // Check if offset is within chunk bounds
            int chunk_length = chunk["length"].as<int>();
            if (offset_in_chunk >= chunk_length) {
                return tablebase::DTM_UNKNOWN;
            }

            // Get the DTM value (signed byte)
            int dtm_value = chunk[offset_in_chunk].as<int>();
            return static_cast<tablebase::DTM>(dtm_value);
        }

        // Find best move using DTM lookup
        bool find_best_move(const Board& board, Move& best_move, tablebase::DTM& best_dtm) const {
            MoveList moves;
            generateMoves(board, moves);

            if (moves.empty()) {
                return false;
            }

            tablebase::DTM current_dtm = lookup_dtm(board);
            tablebase::DTM best_opp_dtm = tablebase::DTM_UNKNOWN;
            bool found = false;

            for (const Move& move : moves) {
                Board child = makeMove(board, move);

                Material child_m = get_material(child);
                tablebase::DTM opp_dtm;
                if (child_m.white_pieces() == 0) {
                    opp_dtm = tablebase::DTM_LOSS_TERMINAL;
                } else {
                    opp_dtm = lookup_dtm(child);
                }

                if (opp_dtm == tablebase::DTM_UNKNOWN) {
                    continue;
                }

                bool dominated = false;
                if (!found) {
                    dominated = false;
                } else if (current_dtm > 0) {
                    if (best_opp_dtm < 0 && opp_dtm >= 0) {
                        dominated = true;
                    } else if (best_opp_dtm >= 0 && opp_dtm < 0) {
                        dominated = false;
                    } else if (best_opp_dtm < 0 && opp_dtm < 0) {
                        dominated = (opp_dtm <= best_opp_dtm);
                    } else {
                        dominated = true;
                    }
                } else if (current_dtm < 0) {
                    dominated = (opp_dtm <= best_opp_dtm);
                } else {
                    dominated = (opp_dtm >= best_opp_dtm);
                }

                if (!dominated) {
                    best_opp_dtm = opp_dtm;
                    best_move = move;
                    found = true;
                }
            }

            if (found) {
                if (best_opp_dtm == tablebase::DTM_LOSS_TERMINAL) {
                    best_dtm = 1;
                } else if (best_opp_dtm < 0) {
                    best_dtm = static_cast<tablebase::DTM>(-best_opp_dtm);
                } else if (best_opp_dtm == 0) {
                    best_dtm = 0;
                } else {
                    best_dtm = static_cast<tablebase::DTM>(-best_opp_dtm);
                }
            }

            return found;
        }
    };

    WasmDTMTablebaseManager g_tb_manager;

    // Neural network models
    std::unique_ptr<nn::MLP> g_nn_model;
    std::unique_ptr<nn::MLP> g_dtm_nn_model;
}

// Load neural network model from typed array
void loadNNModel(val typed_array, bool is_dtm_model) {
    unsigned int length = typed_array["length"].as<unsigned int>();

    // Copy data to a temporary file (Emscripten's virtual filesystem)
    std::vector<uint8_t> buffer(length);
    for (unsigned int i = 0; i < length; ++i) {
        buffer[i] = typed_array[i].as<uint8_t>();
    }

    // Write to virtual filesystem
    const char* path = is_dtm_model ? "/tmp/dtm_model.bin" : "/tmp/nn_model.bin";
    FILE* f = fopen(path, "wb");
    if (f) {
        fwrite(buffer.data(), 1, buffer.size(), f);
        fclose(f);

        try {
            if (is_dtm_model) {
                g_dtm_nn_model = std::make_unique<nn::MLP>(path);
            } else {
                g_nn_model = std::make_unique<nn::MLP>(path);
            }
        } catch (...) {
            // Failed to load model
        }
    }
}

// Check if tablebases are available (asks JS)
bool hasTablebases() {
    val tablebasesAvailable = val::global("tablebasesAvailable");
    if (tablebasesAvailable.isUndefined()) {
        return false;
    }
    return tablebasesAvailable().as<bool>();
}

// Check if NN models are loaded
bool hasNNModel() {
    return g_nn_model != nullptr;
}

bool hasDTMNNModel() {
    return g_dtm_nn_model != nullptr;
}

// Board wrapper for JS
struct JSBoard {
    Board board;
    bool white_to_move;  // Track which side is to move in the game

    JSBoard() : white_to_move(true) {}
    JSBoard(const Board& b, bool wtm) : board(b), white_to_move(wtm) {}

    // Create from bitboards
    static JSBoard fromBitboards(uint32_t white, uint32_t black, uint32_t kings, bool white_to_move) {
        Board b;
        if (white_to_move) {
            b.white = white;
            b.black = black;
            b.kings = kings;
        } else {
            // Flip to internal representation
            b.white = flip(black);
            b.black = flip(white);
            b.kings = flip(kings);
        }
        return JSBoard(b, white_to_move);
    }

    // Get bitboards (always from white's perspective for UI)
    uint32_t getWhite() const {
        return white_to_move ? board.white : flip(board.black);
    }
    uint32_t getBlack() const {
        return white_to_move ? board.black : flip(board.white);
    }
    uint32_t getKings() const {
        return white_to_move ? board.kings : flip(board.kings);
    }
    bool isWhiteToMove() const {
        return white_to_move;
    }

    // Get piece count
    int pieceCount() const {
        return std::popcount(board.allPieces());
    }
};

// Move wrapper for JS
struct JSMove {
    uint32_t from_xor_to;
    uint32_t captures;
    std::vector<int> path;  // Full path for notation

    JSMove() : from_xor_to(0), captures(0) {}
    JSMove(const Move& m, const std::vector<int>& p) : from_xor_to(m.from_xor_to), captures(m.captures), path(p) {}

    bool isCapture() const { return captures != 0; }

    // Get from/to squares (1-32 indexed)
    int getFrom() const {
        if (path.size() >= 1) return path[0];
        return 0;
    }
    int getTo() const {
        if (path.size() >= 2) return path[path.size() - 1];
        return 0;
    }

    // Get notation string
    std::string notation() const {
        if (path.empty()) return "";
        std::string result = std::to_string(path[0]);
        for (size_t i = 1; i < path.size(); ++i) {
            result += (captures ? "x" : "-");
            result += std::to_string(path[i]);
        }
        return result;
    }

    // Get path as JS array
    val getPath() const {
        val arr = val::array();
        for (size_t i = 0; i < path.size(); ++i) {
            arr.call<void>("push", path[i]);
        }
        return arr;
    }
};

// Get legal moves for a position - returns plain JS objects
val getLegalMoves(const JSBoard& jsboard) {
    // Generate moves with full path info, keeping all paths for UI selection
    std::vector<FullMove> full_moves;
    generateFullMoves(jsboard.board, full_moves, true);  // keepAllPaths = true

    val result = val::array();
    for (const auto& fm : full_moves) {
        // Adjust path for perspective
        val path = val::array();
        for (int sq : fm.path) {
            if (jsboard.white_to_move) {
                path.call<void>("push", sq + 1);
            } else {
                path.call<void>("push", 32 - sq);
            }
        }

        // Create plain JS object instead of wrapped C++ object
        val move = val::object();
        move.set("from_xor_to", fm.move.from_xor_to);
        move.set("captures", fm.move.captures);
        move.set("path", path);
        move.set("from", path[0]);
        move.set("to", path[path["length"].as<int>() - 1]);
        move.set("isCapture", fm.move.isCapture());

        // Build notation string
        move.set("notation", buildNotation(fm.path, fm.move.isCapture(), jsboard.white_to_move));

        result.call<void>("push", move);
    }
    return result;
}

// Make a move - accepts plain JS object with from_xor_to and captures
JSBoard makeJSMove(const JSBoard& jsboard, val jsmove) {
    Move m;
    m.from_xor_to = jsmove["from_xor_to"].as<uint32_t>();
    m.captures = jsmove["captures"].as<uint32_t>();

    Board new_board = makeMove(jsboard.board, m);
    return JSBoard(new_board, !jsboard.white_to_move);
}

// Helper to create a move JS object from Move and path
val createMoveObject(const Move& m, const std::vector<int>& path, bool white_to_move) {
    val jsPath = val::array();
    for (int sq : path) {
        if (white_to_move) {
            jsPath.call<void>("push", sq + 1);
        } else {
            jsPath.call<void>("push", 32 - sq);
        }
    }

    val move = val::object();
    move.set("from_xor_to", m.from_xor_to);
    move.set("captures", m.captures);
    move.set("path", jsPath);
    if (jsPath["length"].as<int>() > 0) {
        move.set("from", jsPath[0]);
        move.set("to", jsPath[jsPath["length"].as<int>() - 1]);
    } else {
        move.set("from", 0);
        move.set("to", 0);
    }
    move.set("isCapture", m.isCapture());
    move.set("notation", buildNotation(path, m.isCapture(), white_to_move));

    return move;
}

// Forward declaration
val doSearchWithCallback(const JSBoard& jsboard, int max_depth, double max_nodes_d,
                         int game_ply, int variety_mode, val progress_callback);

// Perform search without callback - wrapper for backwards compatibility
val doSearch(const JSBoard& jsboard, int max_depth, double max_nodes_d) {
    return doSearchWithCallback(jsboard, max_depth, max_nodes_d, 100, 0, val::null());
}

// Helper to build a result val from SearchResult and board
val buildSearchResultVal(const search::SearchResult& sr, const Board& board, bool white_to_move) {
    val result = val::object();

    // Generate full moves for path resolution
    std::vector<FullMove> full_moves;
    generateFullMoves(board, full_moves);

    if (sr.best_move.from_xor_to != 0) {
        for (const auto& fm : full_moves) {
            if (fm.move.from_xor_to == sr.best_move.from_xor_to &&
                fm.move.captures == sr.best_move.captures) {
                result.set("best_move", createMoveObject(fm.move, fm.path, white_to_move));
                break;
            }
        }
    }

    result.set("score", sr.score);
    result.set("depth", sr.depth);
    result.set("nodes", static_cast<double>(sr.nodes));
    result.set("tb_hits", static_cast<double>(sr.tb_hits));

    // Include variety selection info if present
    if (!sr.variety_candidates.empty()) {
        val candidates = val::array();
        for (const auto& vc : sr.variety_candidates) {
            // Find the notation for this move
            std::string notation = "?";
            for (const auto& fm : full_moves) {
                if (fm.move.from_xor_to == vc.move.from_xor_to &&
                    fm.move.captures == vc.move.captures) {
                    notation = buildNotation(fm.path, vc.move.isCapture(), white_to_move);
                    break;
                }
            }

            val candidate = val::object();
            candidate.set("notation", notation);
            candidate.set("score", vc.score);
            candidate.set("probability", vc.probability);
            candidate.set("selected", vc.selected);
            candidates.call<void>("push", candidate);
        }
        result.set("varietyCandidates", candidates);
    }

    // Build PV with full paths for captures
    val pv = val::array();
    if (!sr.pv.empty()) {
        Board pos = board;
        bool white_view = white_to_move;
        for (const Move& m : sr.pv) {
            // Generate full moves to get the complete path for captures
            std::vector<FullMove> full_moves;
            generateFullMoves(pos, full_moves);

            std::string notation;
            bool found = false;
            for (const auto& fm : full_moves) {
                if (fm.move.from_xor_to == m.from_xor_to &&
                    fm.move.captures == m.captures) {
                    notation = buildNotation(fm.path, m.isCapture(), white_view);
                    found = true;
                    break;
                }
            }

            // Fallback to simple from-to notation if not found
            if (!found) {
                Bb occupied = pos.white | pos.black;
                int from_bit = __builtin_ctz(m.from_xor_to & occupied);
                int to_bit = __builtin_ctz(m.from_xor_to ^ (1u << from_bit));
                int disp_from = white_view ? (from_bit + 1) : (32 - from_bit);
                int disp_to = white_view ? (to_bit + 1) : (32 - to_bit);
                notation = std::to_string(disp_from) +
                           (m.isCapture() ? "x" : "-") +
                           std::to_string(disp_to);
            }

            pv.call<void>("push", notation);
            pos = makeMove(pos, m);
            white_view = !white_view;
        }
    }
    result.set("pv", pv);

    return result;
}

// Perform search with optional progress callback
// variety_mode: 0=none, 1=safe, 2=curious, 3=wild
val doSearchWithCallback(const JSBoard& jsboard, int max_depth, double max_nodes_d,
                         int game_ply, int variety_mode, val progress_callback) {
    uint64_t max_nodes = static_cast<uint64_t>(max_nodes_d);
    val result = val::object();
    int piece_count = jsboard.pieceCount();

    // Check if tablebases are available
    bool tb_available = hasTablebases();

    // Try tablebase lookup first for endgame positions
    if (piece_count <= 5 && tb_available) {
        Move best_move;
        tablebase::DTM best_dtm;
        if (g_tb_manager.find_best_move(jsboard.board, best_move, best_dtm)) {
            std::vector<FullMove> full_moves;
            generateFullMoves(jsboard.board, full_moves);

            for (const auto& fm : full_moves) {
                if (fm.move.from_xor_to == best_move.from_xor_to &&
                    fm.move.captures == best_move.captures) {
                    result.set("best_move", createMoveObject(fm.move, fm.path, jsboard.white_to_move));
                    break;
                }
            }

            int score = 0;
            if (best_dtm > 0) {
                score = 30000 - 2 * best_dtm + 1;
            } else if (best_dtm < 0) {
                score = -30000 + 2 * (-best_dtm);
            }
            result.set("score", score);
            result.set("depth", 1);
            result.set("nodes", 1);
            result.set("tb_hits", 1);
            result.set("pv", val::array());
            return result;
        }
    }

    // Set up evaluation function (no noise - variety handled by search)
    auto eval_func = [piece_count](const Board& board, int /*ply*/) -> int {
        if (piece_count <= 7 && g_dtm_nn_model) {
            return g_dtm_nn_model->evaluate(board, 0);
        } else if (g_nn_model) {
            return g_nn_model->evaluate(board, 0);
        } else {
            int white_men = std::popcount(board.whitePawns());
            int white_kings = std::popcount(board.whiteQueens());
            int black_men = std::popcount(board.blackPawns());
            int black_kings = std::popcount(board.blackQueens());
            return (white_men - black_men) * 100 + (white_kings - black_kings) * 300;
        }
    };

    search::Searcher searcher("", 0, "", "");
    searcher.set_eval(eval_func);

    // Reset and set stop flag for this search
    g_stop_flag.store(false, std::memory_order_relaxed);
    searcher.set_stop_flag(&g_stop_flag);

    // Set variety mode
    switch (variety_mode) {
        case 1: searcher.set_variety_mode(search::VarietyMode::SAFE); break;
        case 2: searcher.set_variety_mode(search::VarietyMode::CURIOUS); break;
        case 3: searcher.set_variety_mode(search::VarietyMode::WILD); break;
        default: searcher.set_variety_mode(search::VarietyMode::NONE); break;
    }

    // Set RNG for variety selection
    searcher.set_rng(&g_rng);

    // Set up DTM probe function if tablebases are available
    if (tb_available) {
        searcher.set_dtm_probe([](const Board& b) {
            return g_tb_manager.lookup_dtm(b);
        }, 5);  // 5-piece tablebases
    }

    // Set progress callback if provided
    bool has_callback = !progress_callback.isNull() && !progress_callback.isUndefined();
    if (has_callback) {
        searcher.set_progress_callback([&](const search::SearchResult& sr) {
            val update = buildSearchResultVal(sr, jsboard.board, jsboard.white_to_move);
            progress_callback.call<void>("call", val::null(), update);
        });
    }

    search::SearchResult sr;
    try {
        sr = searcher.search(jsboard.board, max_depth, max_nodes, game_ply);
    } catch (const search::SearchInterrupted&) {
        // Search was interrupted - return minimal result
        val result = val::object();
        result.set("error", "Search interrupted");
        return result;
    } catch (const std::exception& e) {
        val result = val::object();
        result.set("error", std::string("Search exception: ") + e.what());
        return result;
    } catch (...) {
        val result = val::object();
        result.set("error", "Search threw unknown exception");
        return result;
    }
    return buildSearchResultVal(sr, jsboard.board, jsboard.white_to_move);
}

// Probe tablebase for current position (returns DTM or null)
val probeDTM(const JSBoard& jsboard) {
    if (!hasTablebases()) {
        return val::null();
    }

    tablebase::DTM dtm = g_tb_manager.lookup_dtm(jsboard.board);
    if (dtm == tablebase::DTM_UNKNOWN) {
        return val::null();
    }

    return val(static_cast<int>(dtm));
}

// Get initial board position
JSBoard getInitialBoard() {
    return JSBoard(Board(), true);
}

// Parse a move from notation (e.g., "9-13" or "9x14x23")
val doParseMove(const JSBoard& jsboard, const std::string& notation) {
    auto result = ::parseMove(jsboard.board, notation, !jsboard.white_to_move);
    if (!result.has_value()) {
        return val::null();
    }

    return createMoveObject(result->move, result->path, jsboard.white_to_move);
}

// Get engine version string
std::string getEngineVersion() {
    return ENGINE_VERSION;
}

// Stop ongoing search
void stopSearch() {
    g_stop_flag.store(true, std::memory_order_relaxed);
}

// Get address of stop flag for direct memory access from JavaScript
std::uintptr_t getStopFlagAddress() {
    return reinterpret_cast<std::uintptr_t>(&g_stop_flag);
}

// Emscripten bindings
EMSCRIPTEN_BINDINGS(checkers_engine) {
    // JSBoard
    class_<JSBoard>("Board")
        .constructor<>()
        .function("getWhite", &JSBoard::getWhite)
        .function("getBlack", &JSBoard::getBlack)
        .function("getKings", &JSBoard::getKings)
        .function("isWhiteToMove", &JSBoard::isWhiteToMove)
        .function("pieceCount", &JSBoard::pieceCount)
        .class_function("fromBitboards", &JSBoard::fromBitboards);

    // Free functions
    function("getInitialBoard", &getInitialBoard);
    function("getLegalMoves", &getLegalMoves);
    function("makeMove", &makeJSMove);
    function("search", &doSearch);
    function("searchWithCallback", &doSearchWithCallback);  // (board, depth, nodes, ply, variety, callback)
    function("probeDTM", &probeDTM);
    function("parseMove", &doParseMove);
    function("loadNNModel", &loadNNModel);
    function("hasTablebases", &hasTablebases);
    function("hasNNModel", &hasNNModel);
    function("getEngineVersion", &getEngineVersion);
    function("hasDTMNNModel", &hasDTMNNModel);
    function("stopSearch", &stopSearch);
    function("getStopFlagAddress", &getStopFlagAddress);
}
