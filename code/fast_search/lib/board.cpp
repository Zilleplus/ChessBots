#include "board.h"

#include <cassert>
#include <cstdint>
#include <iostream>

#include "ranges"
#include <array>
#include <optional>

namespace chess {

bool Board::isLegalMove(Move m, Piece p){
  switch (p.type) {
    case Piece::Type::Pawn:
      return true;

    case Piece::Type::Bishop:
      return true;

    case Piece::Type::Knight:
      return true;

    case Piece::Type::Rook:
      return true;

    case Piece::Type::Queen:
      return true;

    case Piece::Type::King:
      return true;
    default:
      assert(false); // this is in unsupported piece
      return false;
  }
  return false;
}

void Board::apply(Move move) {
  auto p = getPiece(move.from);
  assert(p.has_value());
  assert(isLegalMove(move, p.value()));
  setPiece(move.to, p.value());
}

unsigned int Board::getAbsoluteLocation(Location loc) {
  return loc.row * boardSize + loc.col;
}

void Board::setPiece(Location loc, Piece p) {
  const auto absLoc = getAbsoluteLocation(loc);
  state[absLoc] = p;

  // TODO::Check for checkmate
}

[[nodiscard]] bool Board::isCheckMate() const {
  return winner.has_value();
}

void Board::print() const{
  auto symbol = [](Piece p) -> char
  {
    switch (p.type) {
      case Piece::Type::Pawn:
        return 'p';
      case Piece::Type::Bishop:
        return 'B';
      case Piece::Type::Knight:
        return 'K';
      case Piece::Type::Rook:
        return 'R';
      case Piece::Type::Queen:
        return 'Q';
      case Piece::Type::King:
        return 'K';
      default:
        assert(false);
        return '?';
    }
  };
  for(uint8_t iCol=0; iCol<Board::boardSize; iCol++){
    std::cout << colNames[iCol];
  }

  for(uint8_t iRow=0; iRow<Board::boardSize; iRow++){
    std::cout << rowNames[iRow];
    for(uint8_t iCol=0; iCol<Board::boardSize; iCol++){
      std::cout << getPiece({.col=iCol, .row=iRow})
         .transform(symbol)
         .or_else([]-> std::optional<char>{return {'.'};})
         .value();
    }
    std::cout << std::endl;
  }
}

}  // namespace chess