#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
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

template<int tape_size, int instruction_count>
struct get_info{
  using type = util::info<double, 1, 1, 6 * tape_size + 2 * instruction_count>;
};

template<int tape_size, int instruction_count>
auto latent_init(){
  //instruction-head, tape-head, tape, tape-head&tape, tape-head-left, tape-head-right, tape-head-copy
  typename get_info<tape_size, instruction_count>::type::latent_vec_t result{};
  result.setZero();
  //init instruction-head
  result(0) = 1.0;
  //init tape-head
  result(2 * instruction_count + tape_size / 2) = 1.0;
  return result;
}

template<int tape_size, int instruction_count>
dyn::model<typename get_info<tape_size, instruction_count>::type> compile_model(const wang_program& program){
  assert(instruction_count == program.instructions.size());
  dyn::model<typename get_info<tape_size, instruction_count>::type> result{1.0};
  result.w.m_latent.setIdentity(); result.w.m_latent *= -1.0;
  for(int i(0); i < static_cast<int>(program.instructions.size()); ++i){
    if(comm::left == program.instructions[i].command){
      result.w.m_latent_1(2*i+1, std::min(2*i+2, 2*instruction_count)) = 1.0;
      result.w.m_latent_2(2*i+1, std::min(2*i+2, 2*instruction_count)) = 1.0;
      for(int j(2*instruction_count + 3*tape_size); j < 2*instruction_count + 4*tape_size; ++j){
        result.w.m_latent_1(2*i, j) = 1.0;
      }
    }else if(comm::right == program.instructions[i].command){
      result.w.m_latent_1(2*i+1, std::min(2*i+2, 2*instruction_count)) = 1.0;
      result.w.m_latent_2(2*i+1, std::min(2*i+2, 2*instruction_count)) = 1.0;
      for(int j(2*instruction_count + 4*tape_size); j < 2*instruction_count + 5*tape_size; ++j){
        result.w.m_latent_1(2*i, j) = 1.0;
      }
    }else if(comm::star == program.instructions[i].command){
      result.w.m_latent_1(2*i+1, std::min(2*i+2, 2*instruction_count)) = 1.0;
      result.w.m_latent_2(2*i+1, std::min(2*i+2, 2*instruction_count)) = 1.0;
      for(int j(2*instruction_count + 5*tape_size); j < 2*instruction_count + 6*tape_size; ++j){
        result.w.m_latent_1(2*i, j) = 1.0;
      }
      for(int j(2*instruction_count); j < 2*instruction_count + tape_size; ++j){
        result.w.m_latent_1(2*i+1, tape_size + j) = 1.0;
        result.w.m_latent_1(2*tape_size + j, tape_size + j) = -1.0;
        result.w.m_latent_2(5*tape_size + j, tape_size + j) = 1.0;
      }
    }else if(comm::c_n == program.instructions[i].command){
      result.w.m_latent_1(2*i+1, 2*program.instructions[i].jump) = 1.0;
      result.w.m_latent_1(2*i+1, std::min(2*i+2, 2*instruction_count)) = 1.0;
      result.w.m_latent_2(2*i+1, std::min(2*i+2, 2*instruction_count)) = 1.0;
      for(int j(2*instruction_count); j < 2*instruction_count + tape_size; ++j){
        result.w.m_latent_2(j, j) = 1.0;
        result.w.m_latent_1(2*i+1, j) = 1.0;
      }
      
      for(int j(2*instruction_count); j < 2*instruction_count + tape_size; ++j){
        result.w.m_latent_2(j + 2*tape_size, program.instructions[i].jump) = 1.0;
        result.w.m_latent_1(j + 2*tape_size, std::min(2*i+2, 2*instruction_count)) = -1.0;
      }
    }
  }
  for(int i(0); i < static_cast<int>(instruction_count); ++i){
    result.w.m_latent_1(2*i, 2*i+1) = 1.0;
    result.w.m_latent_2(2*i, 2*i+1) = 1.0;
    for(int j(2*instruction_count); j < 2*instruction_count + tape_size; ++j){
      result.w.m_latent_1(2*i, j) = 1.0;
      result.w.m_latent_2(j, j) = 1.0;
    }
  }
  for(int i(2*instruction_count); i < 2*instruction_count + tape_size; ++i){
    result.w.m_latent_1(i, 2*tape_size + i) = 1.0;
    result.w.m_latent_2(i + tape_size , 2*tape_size + i) = 1.0;
    result.w.m_latent_2(i, std::max(3*tape_size + i - 1, 3*tape_size + 2*instruction_count)) = 1.0;
    result.w.m_latent_2(i, std::min(4*tape_size + i + 1, 5*tape_size + 2*instruction_count-1)) = 1.0;
    result.w.m_latent_2(i, 5*tape_size + i) = 1.0;
  }
  
  for(int i(2*instruction_count); i < 2*instruction_count + tape_size; ++i){
    result.w.m_latent(i, 3*tape_size + i) = 1.0;
    result.w.m_latent(i, 4*tape_size + i) = 1.0;
    result.w.m_latent(i, 5*tape_size + i) = 1.0;
    result.w.m_latent(i + tape_size, i + tape_size) = 0.0;
  }
  
  
  result.w.m_latent_1 = result.w.m_latent_1.transpose().eval();
  result.w.m_latent_2 = result.w.m_latent_2.transpose().eval();
  return result;
}

void execute_display(const wang_program& program, const size_t iter = 100){
  constexpr size_t tape_size = 7;
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
  
  constexpr int tape_size = 11;
  constexpr int instruction_count = 4;
  
  const auto compiled = compile_model<tape_size, instruction_count>(wp);
  std::cout << std::endl << std::endl;
  std::cout << compiled << std::endl;
  std::cout << std::endl << std::endl;
  
  auto latent = latent_init<tape_size, instruction_count>();
  std::cout << "latent_init: " << std::endl;
  std::cout << latent.transpose() << std::endl << std::endl;
  std::cout << "======================================================================================================" << std::endl;
  typename get_info<tape_size, instruction_count>::type::in_vec_t input{}; input.setZero(); 
  
  for(size_t i(0); i < 100; ++i){
    const auto [next_input, next_latent] = compiled.forward(decltype(compiled)::backward_t(input, latent));
    input = next_input;
    latent = next_latent;
    std::cout << i << ':' << std::endl;
    std::cout << latent.transpose() << std::endl;
  }
}
