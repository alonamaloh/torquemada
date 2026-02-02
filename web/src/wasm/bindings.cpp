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

// Global tablebase data storage
// Tablebases are loaded from JS and stored here as raw bytes
namespace {
    // Global RNG for evaluation noise, seeded with system clock
    RandomBits g_rng(std::chrono::steady_clock::now().time_since_epoch().count());

    // Map from material config string to DTM data
    std::unordered_map<std::string, std::vector<tablebase::DTM>> g_tablebase_data;

    // Simple DTM tablebase manager for WASM that uses preloaded data
    class WasmDTMTablebaseManager {
    public:
        tablebase::DTM lookup_dtm(const Board& board) const {
            Material m = get_material(board);
            std::string key = material_key(m);

            auto it = g_tablebase_data.find(key);
            if (it == g_tablebase_data.end() || it->second.empty()) {
                return tablebase::DTM_UNKNOWN;
            }

            std::size_t idx = board_to_index(board, m);
            if (idx >= it->second.size()) {
                return tablebase::DTM_UNKNOWN;
            }
            return it->second[idx];
        }

        bool has_dtm(const Board& board) const {
            Material m = get_material(board);
            std::string key = material_key(m);
            auto it = g_tablebase_data.find(key);
            return it != g_tablebase_data.end() && !it->second.empty();
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

    private:
        static std::string material_key(const Material& m) {
            char buf[16];
            snprintf(buf, sizeof(buf), "%d%d%d%d%d%d",
                     m.back_white_pawns, m.back_black_pawns,
                     m.other_white_pawns, m.other_black_pawns,
                     m.white_queens, m.black_queens);
            return buf;
        }
    };

    WasmDTMTablebaseManager g_tb_manager;

    // Neural network models
    std::unique_ptr<nn::MLP> g_nn_model;
    std::unique_ptr<nn::MLP> g_dtm_nn_model;
}

// Load tablebase data from a typed array (called from JS)
void loadTablebaseData(const std::string& material_key, val typed_array) {
    // Get the length and copy data
    unsigned int length = typed_array["length"].as<unsigned int>();

    // DTM files have a header: version (1) + material (24) + count (8) = 33 bytes
    if (length < 33) {
        return;
    }

    // Copy data from JS typed array
    std::vector<uint8_t> buffer(length);
    val memory = val::module_property("HEAPU8");
    val memview = typed_array.call<val>("slice");
    for (unsigned int i = 0; i < length; ++i) {
        buffer[i] = memview[i].as<uint8_t>();
    }

    // Parse header
    uint8_t version = buffer[0];
    if (version != 1) {
        return;
    }

    // Skip material verification (we trust JS to pass correct key)
    // Read count (at offset 25)
    std::size_t count;
    std::memcpy(&count, buffer.data() + 25, sizeof(count));

    // Parse DTM values (1 byte each, signed)
    std::vector<tablebase::DTM> dtm_data(count);
    for (std::size_t i = 0; i < count && 33 + i < length; ++i) {
        dtm_data[i] = static_cast<tablebase::DTM>(static_cast<int8_t>(buffer[33 + i]));
    }

    g_tablebase_data[material_key] = std::move(dtm_data);
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

// Check if tablebases are loaded
bool hasTablebases() {
    return !g_tablebase_data.empty();
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
        std::string notation;
        for (size_t i = 0; i < fm.path.size(); ++i) {
            if (i > 0) notation += fm.move.isCapture() ? "x" : "-";
            int sq = jsboard.white_to_move ? (fm.path[i] + 1) : (32 - fm.path[i]);
            notation += std::to_string(sq);
        }
        move.set("notation", notation);

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

// Search result - now returns plain JS object
// (JSSearchResult struct kept for internal use but we return val)

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

    std::string notation;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i > 0) notation += m.isCapture() ? "x" : "-";
        int sq = white_to_move ? (path[i] + 1) : (32 - path[i]);
        notation += std::to_string(sq);
    }
    move.set("notation", notation);

    return move;
}

// Forward declaration
val doSearchWithCallback(const JSBoard& jsboard, int max_depth, double max_nodes_d, val progress_callback);

// Perform search without callback - wrapper for backwards compatibility
val doSearch(const JSBoard& jsboard, int max_depth, double max_nodes_d) {
    return doSearchWithCallback(jsboard, max_depth, max_nodes_d, val::null());
}

// Helper to build a result val from SearchResult and board
val buildSearchResultVal(const search::SearchResult& sr, const Board& board, bool white_to_move) {
    val result = val::object();

    if (sr.best_move.from_xor_to != 0) {
        std::vector<FullMove> full_moves;
        generateFullMoves(board, full_moves);

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

    // Build PV
    val pv = val::array();
    if (!sr.pv.empty()) {
        Board pos = board;
        bool white_view = white_to_move;
        for (const Move& m : sr.pv) {
            Bb occupied = pos.white | pos.black;
            int from_bit = __builtin_ctz(m.from_xor_to & occupied);
            int to_bit = __builtin_ctz(m.from_xor_to ^ (1u << from_bit));
            int disp_from = white_view ? (from_bit + 1) : (32 - from_bit);
            int disp_to = white_view ? (to_bit + 1) : (32 - to_bit);
            std::string notation = std::to_string(disp_from) +
                                   (m.isCapture() ? "x" : "-") +
                                   std::to_string(disp_to);
            pv.call<void>("push", notation);
            pos = makeMove(pos, m);
            white_view = !white_view;
        }
    }
    result.set("pv", pv);

    return result;
}

// Perform search with optional progress callback
val doSearchWithCallback(const JSBoard& jsboard, int max_depth, double max_nodes_d, val progress_callback) {
    uint64_t max_nodes = static_cast<uint64_t>(max_nodes_d);
    val result = val::object();
    int piece_count = jsboard.pieceCount();

    // Try tablebase lookup first for endgame positions
    if (piece_count <= 5 && !g_tablebase_data.empty()) {
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

    // Set up evaluation function with small random noise for variety
    auto eval_func = [piece_count](const Board& board, int /*ply*/) -> int {
        int score;
        if (piece_count <= 7 && g_dtm_nn_model) {
            score = g_dtm_nn_model->evaluate(board, 0);
        } else if (g_nn_model) {
            score = g_nn_model->evaluate(board, 0);
        } else {
            int white_men = std::popcount(board.whitePawns());
            int white_kings = std::popcount(board.whiteQueens());
            int black_men = std::popcount(board.blackPawns());
            int black_kings = std::popcount(board.blackQueens());
            score = (white_men - black_men) * 100 + (white_kings - black_kings) * 300;
        }
        // Add small noise for variety: -5 to +5
        int noise = -5 + static_cast<int>(g_rng() % 11);
        return score + noise;
    };

    search::Searcher searcher("", 0, "", "");
    searcher.set_eval(eval_func);

    // Set progress callback if provided
    bool has_callback = !progress_callback.isNull() && !progress_callback.isUndefined();
    if (has_callback) {
        searcher.set_progress_callback([&](const search::SearchResult& sr) {
            val update = buildSearchResultVal(sr, jsboard.board, jsboard.white_to_move);
            progress_callback(update);
        });
    }

    search::SearchResult sr = searcher.search(jsboard.board, max_depth, max_nodes);
    return buildSearchResultVal(sr, jsboard.board, jsboard.white_to_move);
}

// Probe tablebase for current position (returns DTM or null)
val probeDTM(const JSBoard& jsboard) {
    if (g_tablebase_data.empty()) {
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

    // Moves and SearchResults are returned as plain JS objects, not wrapped C++ classes

    // Free functions
    function("getInitialBoard", &getInitialBoard);
    function("getLegalMoves", &getLegalMoves);
    function("makeMove", &makeJSMove);
    function("search", &doSearch);
    function("searchWithCallback", &doSearchWithCallback);
    function("probeDTM", &probeDTM);
    function("parseMove", &doParseMove);
    function("loadTablebaseData", &loadTablebaseData);
    function("loadNNModel", &loadNNModel);
    function("hasTablebases", &hasTablebases);
    function("hasNNModel", &hasNNModel);
    function("hasDTMNNModel", &hasDTMNNModel);
}
