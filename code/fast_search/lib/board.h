#pragma once
#include <cinttypes>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace chess {
struct Piece {
  enum class Type : char {
    Pawn,
    Bishop,
    Knight,
    Rook,
    Queen,
    King,
  };
  enum class Color : char { Black, White };

  Type type;
  Color color;
};

// should this be a giant enum?
struct Location {
  uint8_t col;
  uint8_t row;
};

struct Move {
  Location from;
  Location to;
};

class Board {
  static constexpr size_t  boardSize = 8;
  static constexpr std::array<char, boardSize> colNames = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  static constexpr std::array<char, boardSize> rowNames = {'1', '2', '3', '4', '5', '6', '7', '8'};

  std::array<std::optional<Piece>, boardSize * boardSize> state;
  std::optional<Piece::Color> winner;

  static unsigned int getAbsoluteLocation(Location loc);
  void setPiece(Location loc, Piece p);

 public:

  [[nodiscard]] std::optional<Piece> getPiece(Location) const;

  [[nodiscard]] static bool isLegalMove(Move, Piece);

  void apply(Move);

  [[nodiscard]] bool isCheckMate() const;

  [[nodiscard]] std::optional<Piece::Color> getWinner() const;

  void reset();

  void print() const;
};

}  // namespace chess