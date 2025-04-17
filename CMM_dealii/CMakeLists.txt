cmake_minimum_required(VERSION 3.16)

project(cmm_project)

find_package(deal.II REQUIRED HINTS /usr/local/dealii)

deal_ii_initialize_cached_variables()

add_executable(cmm cmm.cc)

deal_ii_setup_target(cmm)

