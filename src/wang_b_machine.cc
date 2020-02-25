#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>

#include <Eigen/Dense>

#include <util.h>
#include <model.h>

// demonstration of model turing completeness

enum class comm{ left, right, star, c_n };

std::string comm_to_string(const comm c){
  switch(c){
    case comm::left: return "left";
    case comm::right: return "right";
    case comm::star: return "star";
    case comm::c_n: return "c_n";
  }
  assert(false); return "?";
}

comm comm_from_string(const std::string& str){
  if(comm_to_string(comm::left) == str) return comm::left;
  if(comm_to_string(comm::right) == str) return comm::right;
  if(comm_to_string(comm::star) == str) return comm::star;
  if(comm_to_string(comm::c_n) == str) return comm::c_n;
  assert(false); return comm::left;
}


struct wang_node{
  comm command;
  size_t jump{0};
  
  wang_node(const comm c){
    assert(c != comm::c_n);
    command = c;
  }
  
  wang_node(const comm c, const size_t j){
    assert(c == comm::c_n);
    command = c;
    jump = j;
  }
};

std::ostream& operator<<(std::ostream& os, const wang_node& node){
  os << comm_to_string(node.command);
  if(node.command == comm::c_n){
    os << ' ' << node.jump;
  }
  return os;
}

struct wang_program{
  std::vector<wang_node> instructions{};
};

std::istream& operator>>(std::istream& is, wang_program& wp){
  wp.instructions.clear();
  while(true){
    std::string type_str;
    if(!(is >> type_str)) break;
    const comm type = comm_from_string(type_str);
    if(type == comm::c_n){
      size_t jump; is >> jump;
      wp.instructions.push_back(wang_node(type, jump));
    }else{
      wp.instructions.push_back(wang_node(type));
    }
  }
  return is;
}

std::ostream& operator<<(std::ostream& os, const wang_program& wp){
  for(const auto& node : wp.instructions){
    os << node << ' ';
  }
  return os;
}

void execute_display(const wang_program& program, const size_t iter = 100){
  constexpr size_t tape_size = 25;
  std::vector<bool> tape(tape_size, false);
  size_t head_pos = tape_size / 2;
  size_t comm_idx = 0;

  auto present_tape = [&](){
    for(bool val : tape){
      std::cout << val << ' ';
    }
    std::cout << std::endl;
    for(size_t i(0); i < tape.size(); ++i){
      if(i == head_pos){ std::cout << "^_"; }
      else{ std::cout << "__"; }
    }
    std::cout << std::endl;
  };

  for(size_t i(0); i < iter; ++i){
    std::cout << program.instructions[comm_idx] << std::endl;
    switch(program.instructions[comm_idx].command){
      case comm::left: { --head_pos; ++comm_idx; break; }
      case comm::right: { ++head_pos; ++comm_idx; break; }
      case comm::star: { tape[head_pos] = true; ++comm_idx; break; }
      case comm::c_n: {
        if(tape[head_pos]){
          comm_idx = program.instructions[comm_idx].jump;
        }else{
          ++comm_idx;
        }
        break;
      }
    }
    present_tape();
  }
}

int main(){
  std::cout << "file name: ";
  std::string file_name; std::cin >> file_name; 
  auto file_handler = std::fstream(file_name);
  wang_program wp; file_handler >> wp;
  std::cout << "loaded: " << std::endl;
  std::cout << wp << std::endl;
  std::cout << std::endl << std::endl << std::endl;
  execute_display(wp);
}
